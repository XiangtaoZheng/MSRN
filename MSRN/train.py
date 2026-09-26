"""Train reconstruction, loss prediction and Swin V2-B classification."""

import argparse
from datetime import datetime
import json
from pathlib import Path

from eva1.config import DEFAULT_CONFIG, load_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Folder containing class subfolders")
    parser.add_argument("--dataset", default="HSC-XMS")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to a new timestamped runs/ folder")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--epochs", type=int, default=51)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--meta-steps", type=int, default=200, help="Maximum support/query pairs per epoch; 0 disables meta phase")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--step-size", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--freeze-epochs", type=int, default=10)
    parser.add_argument("--coef-int", type=float, default=1.0)
    parser.add_argument("--coef-spec", type=float, default=0.2)
    parser.add_argument("--coef-grad", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=None, help="Optional square resize; default preserves original resolution")
    parser.add_argument("--base-ch", type=int, default=128, help="Reconstruction width; retain 128 for the original architecture")
    parser.add_argument("--lpn-dim", type=int, default=64, help="Must be a positive multiple of 8")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True,
                        help="ImageNet Swin weights; --no-pretrained avoids downloads")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.step_size, args.base_ch, args.save_every) < 1:
        parser.error("epochs, batch-size, step-size, base-ch and save-every must be positive")
    if min(args.num_workers, args.meta_steps, args.freeze_epochs) < 0 or args.lr <= 0 or not 0 < args.gamma <= 1:
        parser.error("invalid worker count, meta steps, freeze epochs, learning rate or gamma")
    if args.lpn_dim < 8 or args.lpn_dim % 8:
        parser.error("lpn-dim must be a positive multiple of 8")
    if min(args.coef_int, args.coef_spec, args.coef_grad) < 0:
        parser.error("loss coefficients must be non-negative")
    if args.image_size is not None and args.image_size < 8:
        parser.error("image-size must be >= 8")
    return args


def main():
    args = parse_args()
    import torch
    from torch.utils.data import DataLoader
    from eva1.data import HSIFolder
    from eva1.engine import meta_step, train_step
    from eva1.models import LPN, Reconstruction
    from eva1.runtime import create_classifier, seed_everything, select_device

    seed_everything(args.seed)
    device = select_device(args.device)
    config = load_config(args.dataset, args.config)
    dataset = HSIFolder(args.data_root, config, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")
    if args.meta_steps and len(loader) < 2:
        raise ValueError("Meta-learning requires at least two batches; lower --batch-size or use --meta-steps 0")
    # Check TIFF shape/layout before allocating the full networks or downloading weights.
    dataset[0]
    output = args.output_dir or Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("latest.pt", "config.json", "metrics.jsonl")):
        raise FileExistsError(f"Existing run in {output}; choose a new --output-dir")
    reconstruction = Reconstruction(inc=config.channels, base_ch=args.base_ch, rgb_band=config.rgb_bands).to(device)
    lpn = LPN(inc=config.channels, dim=args.lpn_dim).to(device)
    classifier = create_classifier(config.num_classes, args.pretrained).to(device)
    for parameter in classifier.features.parameters():
        parameter.requires_grad_(args.freeze_epochs == 0)
    optimizer_r = torch.optim.Adam(reconstruction.parameters(), lr=args.lr, weight_decay=1e-6)
    optimizer_c = torch.optim.Adam((p for p in classifier.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-6)
    scheduler_r = torch.optim.lr_scheduler.StepLR(optimizer_r, step_size=args.step_size, gamma=args.gamma)
    scheduler_c = torch.optim.lr_scheduler.StepLR(optimizer_c, step_size=args.step_size, gamma=args.gamma)
    coefficients = (args.coef_int, args.coef_spec, args.coef_grad)
    metadata = {"format_version": 1, "dataset": args.dataset, "dataset_config": config.to_dict(),
                "classes": dataset.classes, "image_size": args.image_size,
                "base_ch": args.base_ch, "lpn_dim": args.lpn_dim}
    run_config = {**metadata, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    (output / "config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Device: {device}; samples: {len(dataset)}; output: {output}", flush=True)

    def to_device(batch):
        return tuple(value.to(device, non_blocking=True) for value in batch)

    for epoch in range(args.epochs):
        if epoch == args.freeze_epochs and args.freeze_epochs > 0:
            for parameter in classifier.features.parameters():
                parameter.requires_grad_(True)
            optimizer_c = torch.optim.Adam((p for p in classifier.parameters() if p.requires_grad), lr=args.lr, weight_decay=0)
            scheduler_c = torch.optim.lr_scheduler.StepLR(optimizer_c, step_size=args.step_size, gamma=args.gamma)
        reconstruction.train()
        classifier.train()
        lpn.train()
        meta_losses = []
        batches = iter(loader)
        for _ in range(min(args.meta_steps, len(loader) // 2)):
            meta_losses.append(meta_step(reconstruction, classifier, lpn, optimizer_c,
                                         to_device(next(batches)), to_device(next(batches)), config.rgb_bands,
                                         coefficients, optimizer_r.param_groups[0]["lr"]))
        lpn.eval()
        losses = [train_step(reconstruction, classifier, lpn, optimizer_r, optimizer_c,
                             to_device(batch), config.rgb_bands, coefficients) for batch in loader]
        report = {"epoch": epoch + 1, "reconstruction_loss": sum(v[0] for v in losses) / len(losses),
                  "classification_loss": sum(v[1] for v in losses) / len(losses),
                  "meta_pairs": len(meta_losses),
                  "meta_task_loss": sum(v[1] for v in meta_losses) / len(meta_losses) if meta_losses else None}
        print(json.dumps(report), flush=True)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(report) + "\n")
        scheduler_r.step()
        scheduler_c.step()
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            checkpoint = {**metadata, "epoch": epoch + 1, "reconstruction": reconstruction.state_dict(),
                          "classifier": classifier.state_dict(), "lpn": lpn.state_dict()}
            temporary = output / "latest.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(output / "latest.pt")


if __name__ == "__main__":
    main()
