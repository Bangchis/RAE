from __future__ import annotations

import argparse
from pathlib import Path

from utils.fid_utils import (
    calculate_stats_from_array,
    calculate_stats_from_image_folder,
    load_image_array,
    write_reference_stats,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reference FID statistics from an image folder or image archive.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to an image folder, .npy, or .npz used as the reference distribution.",
    )
    parser.add_argument("--output", type=str, required=True, help="Destination .npz file for mu/sigma statistics.")
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device used for Inception feature extraction.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for Inception feature extraction.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for image-folder input.")
    parser.add_argument("--num-threads", type=int, default=None, help="Optional torch CPU thread count for FID stats.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        mu, sigma, num_samples = calculate_stats_from_image_folder(
            input_path,
            batch_size=args.batch_size,
            device=args.device,
            num_workers=args.num_workers,
            num_threads=args.num_threads,
        )
    else:
        arr = load_image_array(input_path)
        mu, sigma, num_samples = calculate_stats_from_array(
            arr,
            batch_size=args.batch_size,
            device=args.device,
            num_threads=args.num_threads,
        )

    output_path = write_reference_stats(
        args.output,
        mu=mu,
        sigma=sigma,
        num_samples=num_samples,
        source=input_path,
    )
    print(f"Saved reference FID statistics to {output_path} [num_samples={num_samples}]")


if __name__ == "__main__":
    main()
