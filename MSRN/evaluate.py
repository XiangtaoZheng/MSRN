"""Evaluate a trained checkpoint and save metrics plus per-image predictions."""

import argparse
import csv
import json
from pathlib import Path
import time

from eva1.config import DEFAULT_CONFIG, DatasetConfig, load_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Test folder containing class subfolders")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="latest.pt produced by train.py")
    source.add_argument("--legacy-model-dir", type=Path, help="Folder with original Reconstruction_*_spec.pth and CLS_*_spec.pth")
    parser.add_argument("--dataset", help="Required only for legacy checkpoints")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Dataset settings for legacy checkpoints")
    parser.add_argument("--image-size", type=int, help="Optional resize for legacy checkpoints")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup-batches", type=int, default=10, help="Initial batches excluded from timing, still included in accuracy")
    args = parser.parse_args()
    if args.legacy_model_dir and not args.dataset:
        parser.error("--dataset is required with --legacy-model-dir")
    if args.checkpoint and (args.dataset is not None or args.image_size is not None or args.config != DEFAULT_CONFIG):
        parser.error("New checkpoints already contain dataset settings and image size; omit overrides")
    if args.batch_size < 1 or min(args.num_workers, args.warmup_batches) < 0:
        parser.error("batch-size must be positive; num-workers and warmup-batches must be non-negative")
    if args.image_size is not None and args.image_size < 8:
        parser.error("image-size must be >= 8")
    return args


def main():
    args = parse_args()
    import torch
    from torch.utils.data import DataLoader
    from eva1.data import HSIFolder
    from eva1.models import Reconstruction
    from eva1.runtime import classification_metrics, create_classifier, imagenet_norm, select_device

    device = select_device(args.device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format")
        config = DatasetConfig(**checkpoint["dataset_config"])
        classes, image_size, base_ch = checkpoint["classes"], checkpoint["image_size"], checkpoint["base_ch"]
        reconstruction_state, classifier_state = checkpoint["reconstruction"], checkpoint["classifier"]
    else:
        config = load_config(args.dataset, args.config)
        classes, image_size, base_ch = None, args.image_size, 128
        reconstruction_state = torch.load(args.legacy_model_dir / f"Reconstruction_{args.dataset}_spec.pth", map_location="cpu", weights_only=True)
        classifier_state = torch.load(args.legacy_model_dir / f"CLS_{args.dataset}_spec.pth", map_location="cpu", weights_only=True)
        print("Legacy evaluation assumes all class folders are present in the original alphabetical order.", flush=True)
    dataset = HSIFolder(args.data_root, config, classes=classes, image_size=image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    reconstruction = Reconstruction(inc=config.channels, base_ch=base_ch, rgb_band=config.rgb_bands).to(device)
    classifier = create_classifier(config.num_classes, pretrained=False).to(device)
    reconstruction.load_state_dict(reconstruction_state, strict=True)
    classifier.load_state_dict(classifier_state, strict=True)
    reconstruction.eval()
    classifier.eval()
    # Release CPU checkpoint tensors (including the unused LPN).
    del reconstruction_state, classifier_state
    if args.checkpoint:
        del checkpoint
    targets, predictions = [], []
    elapsed, timed_samples = 0.0, 0
    with torch.inference_mode():
        for step, (images, labels) in enumerate(loader):
            images = images.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            rgb, _ = reconstruction(images)
            prediction = classifier(imagenet_norm(rgb)).argmax(1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            duration = time.perf_counter() - started
            if step >= args.warmup_batches:
                elapsed += duration
                timed_samples += len(labels)
            targets.extend(labels.tolist())
            predictions.extend(prediction.cpu().tolist())
    metrics = classification_metrics(targets, predictions, dataset.classes)
    metrics.update({"timed_samples": timed_samples, "mean_inference_ms_per_image":
                    1000 * elapsed / timed_samples if timed_samples else None})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file", "target", "prediction"])
        for (path, _), target, prediction in zip(dataset.samples, targets, predictions):
            writer.writerow([path.relative_to(dataset.root).as_posix(), dataset.classes[target], dataset.classes[prediction]])
    print(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
