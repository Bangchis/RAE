from __future__ import annotations

import argparse
from pathlib import Path

from utils.fid_utils import calculate_fid_from_path, write_fid_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute FID for a generated sample archive against reference stats.")
    parser.add_argument("--samples", type=str, required=True, help="Path to a generated .npy or .npz image archive.")
    parser.add_argument(
        "--ref",
        type=str,
        required=True,
        help="Path to a .npz file containing reference statistics with mu/sigma arrays.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device used for FID feature extraction.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for Inception feature extraction.")
    parser.add_argument("--num-threads", type=int, default=None, help="Optional torch CPU thread count for FID.")
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save the computed FID result as JSON.",
    )
    args = parser.parse_args()

    fid_value = calculate_fid_from_path(
        args.samples,
        args.ref,
        batch_size=args.batch_size,
        device=args.device,
        num_threads=args.num_threads,
    )
    print(f"FID: {fid_value:.6f}")

    if args.output_json is not None:
        output_path = write_fid_result(
            args.output_json,
            fid=fid_value,
            sample_path=args.samples,
            reference_path=args.ref,
            batch_size=args.batch_size,
            device=args.device,
            num_threads=args.num_threads,
        )
        print(f"Saved FID result to {output_path}")


if __name__ == "__main__":
    main()
