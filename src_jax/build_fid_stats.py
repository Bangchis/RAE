from __future__ import annotations

import argparse

try:
    from .backend_fid import calculate_backend_reference_stats, write_backend_reference_stats
except ImportError:
    from backend_fid import calculate_backend_reference_stats, write_backend_reference_stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build reference FID statistics with the same Flax Inception detector used by the JAX backend."
    )
    parser.add_argument("--input", type=str, required=True, help="ImageFolder root used as the real-image distribution.")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Destination .pkl/.pickle/.npz file for reference mu/sigma statistics.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Optional center-crop size before feature extraction. Leave unset to use the stored pixels as-is.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Host batch size before local-device sharding.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for the image folder.")
    args = parser.parse_args()

    mu, sigma, num_samples = calculate_backend_reference_stats(
        args.input,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    output_path = write_backend_reference_stats(
        args.output,
        mu=mu,
        sigma=sigma,
        num_samples=num_samples,
        source=args.input,
    )
    print(f"Saved backend JAX FID statistics to {output_path} [num_samples={num_samples}]")


if __name__ == "__main__":
    main()
