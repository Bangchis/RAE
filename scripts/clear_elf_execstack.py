#!/usr/bin/env python3
from __future__ import annotations

import argparse
import site
import struct
import sys
from pathlib import Path


ELF_MAGIC = b"\x7fELF"
PT_GNU_STACK = 0x6474E551
PF_X = 0x1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clear the executable-stack flag from ELF binaries such as jaxlib/xla_extension.so."
    )
    parser.add_argument("paths", nargs="*", help="Files or directories to scan.")
    parser.add_argument(
        "--package",
        dest="packages",
        action="append",
        default=[],
        help="Package directory to scan inside the current interpreter site-packages.",
    )
    parser.add_argument(
        "--quiet-unchanged",
        action="store_true",
        help="Only print files that were patched.",
    )
    return parser


def _site_roots() -> list[Path]:
    roots: list[Path] = []
    for raw in site.getsitepackages():
        roots.append(Path(raw))
    try:
        roots.append(Path(site.getusersitepackages()))
    except AttributeError:
        pass
    return roots


def _resolve_package_targets(packages: list[str]) -> list[Path]:
    targets: list[Path] = []
    missing: list[str] = []
    roots = _site_roots()
    for package in packages:
        found = False
        for root in roots:
            candidate = root / package
            if candidate.exists():
                targets.append(candidate)
                found = True
                break
        if not found:
            missing.append(package)
    if missing:
        joined = ", ".join(sorted(missing))
        raise FileNotFoundError(f"Package path not found in site-packages: {joined}")
    return targets


def _iter_elf_files(targets: list[Path]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for target in targets:
        resolved = target.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Path not found: {resolved}")
        candidates = [resolved] if resolved.is_file() else [path for path in resolved.rglob("*") if path.is_file()]
        for candidate in candidates:
            if candidate in seen:
                continue
            try:
                with candidate.open("rb") as handle:
                    if handle.read(4) != ELF_MAGIC:
                        continue
            except OSError:
                continue
            seen.add(candidate)
            files.append(candidate)
    return files


def _elf_layout(header: bytes) -> tuple[str, int, int, int]:
    if len(header) < 64 or header[:4] != ELF_MAGIC:
        raise ValueError("Not an ELF file.")
    elf_class = header[4]
    data_encoding = header[5]
    if data_encoding == 1:
        endian = "<"
    elif data_encoding == 2:
        endian = ">"
    else:
        raise ValueError("Unsupported ELF data encoding.")

    if elf_class == 1:
        e_phoff = struct.unpack_from(f"{endian}I", header, 28)[0]
        e_phentsize = struct.unpack_from(f"{endian}H", header, 42)[0]
        e_phnum = struct.unpack_from(f"{endian}H", header, 44)[0]
        return endian, e_phoff, e_phentsize, e_phnum
    if elf_class == 2:
        e_phoff = struct.unpack_from(f"{endian}Q", header, 32)[0]
        e_phentsize = struct.unpack_from(f"{endian}H", header, 54)[0]
        e_phnum = struct.unpack_from(f"{endian}H", header, 56)[0]
        return endian, e_phoff, e_phentsize, e_phnum
    raise ValueError(f"Unsupported ELF class: {elf_class}")


def clear_execstack_flag(path: Path) -> bool:
    with path.open("r+b") as handle:
        header = handle.read(64)
        endian, e_phoff, e_phentsize, e_phnum = _elf_layout(header)
        elf_class = header[4]
        for index in range(e_phnum):
            ph_offset = e_phoff + index * e_phentsize
            handle.seek(ph_offset)
            program_header = handle.read(e_phentsize)
            if len(program_header) < e_phentsize:
                raise ValueError(f"Truncated program header in {path}")

            p_type = struct.unpack_from(f"{endian}I", program_header, 0)[0]
            if p_type != PT_GNU_STACK:
                continue

            flags_offset = 24 if elf_class == 1 else 4
            p_flags = struct.unpack_from(f"{endian}I", program_header, flags_offset)[0]
            if not (p_flags & PF_X):
                return False

            handle.seek(ph_offset + flags_offset)
            handle.write(struct.pack(f"{endian}I", p_flags & ~PF_X))
            return True

    return False


def main() -> int:
    args = build_parser().parse_args()
    targets = [Path(raw) for raw in args.paths]
    targets.extend(_resolve_package_targets(args.packages))
    if not targets:
        raise SystemExit("Provide at least one path or --package target.")

    elf_files = _iter_elf_files(targets)
    if not elf_files:
        raise SystemExit("No ELF files found to inspect.")

    patched = 0
    for elf_path in elf_files:
        changed = clear_execstack_flag(elf_path)
        if changed:
            patched += 1
            print(f"patched executable-stack flag: {elf_path}")
        elif not args.quiet_unchanged:
            print(f"already clear: {elf_path}")

    if patched == 0 and args.quiet_unchanged:
        return 0
    print(f"checked {len(elf_files)} ELF file(s); patched {patched}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
