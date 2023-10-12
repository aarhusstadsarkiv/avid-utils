from argparse import ArgumentParser
from datetime import timedelta
from math import ceil
from math import log10
from pathlib import Path
from time import perf_counter
from typing import BinaryIO

from .utils import print_log

text_bytes: set[int] = {7, 8, 9, 10, 12, 13, 27, *range(0x20, 0x7f), *range(0x80, 0x100)}
control_bytes: set[int] = set(range(0, 32)) - text_bytes


def is_binary(fh: BinaryIO) -> bool:
    data: bytes = fh.read(1024)
    fh.seek(0)
    return bool(data.translate(None, bytes(text_bytes)))


def main(file: Path, commit: bool, keep: bool, log_file: Path):
    echo = print_log(log_file)
    file_new: Path = file.with_name("." + file.name).with_suffix(".tmp")

    t1: float = perf_counter()

    try:
        with file.open("rb") as fi, file_new.open("wb") as fo:
            if is_binary(fi):
                echo(f"{file.name}/is binary")
                return

            index: int = 0
            chunk_size: int = 1_000_000
            index_max: int = file.stat().st_size
            index_power: int = max(2, ceil(abs(log10(chunk_size / index_max))) - 2) if chunk_size < index_max else 2
            line: str = ""
            chunk: bytes = bytes([0] if index_max else [])

            while chunk:
                chunk: bytes = fi.read(chunk_size)
                index += len(chunk)

                line = f"{file.name}/reading/{(index / index_max) * 100:.0{index_power}f}%"
                print("\r" + line, end="", flush=True)

                if set(chunk) & control_bytes:
                    print("\r" + (" " * len(line)) + "\r", end="", flush=True)
                    for n, byte in filter(lambda c: c[1] in control_bytes, enumerate(chunk)):
                        echo(f"{file.name}/{index - len(chunk) + n}/{byte:02x}")
                    chunk = chunk.translate(None, bytes(control_bytes))
                    print(line, end="", flush=True)

                if commit:
                    fo.write(chunk)

        print("\r" + (" " * len(line)) + "\r", end="", flush=True)
    except (Exception, BaseException):
        file_new.unlink(missing_ok=True)
        raise

    if commit and file.stat().st_size != file_new.stat().st_size:
        size, old_size = file_new.stat().st_size, file.stat().st_size
        if keep:
            file_keep = file.replace(file.with_stem(file.stem + ".old"))
            echo(f"\r{file.name}/preserved {file_keep.name}")
        file_new.replace(file)
        echo(f"\r{file.name}/saved {size}B")
        echo(f"\r{file.name}/removed {old_size - size}B")
    else:
        file_new.unlink(missing_ok=True)

    echo(f"{file.name}/time/{timedelta(seconds=perf_counter() - t1)}")


def cli():
    """
    Remove control characters from a text file.

    Removed characters are: 00, 01, 02, 03, 04, 05, 06, 0b, 0e, 0f, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 1a, 1c, 1d, 1e, 1f.
    """

    parser = ArgumentParser("remove-control-characters", description=cli.__doc__)
    parser.add_argument("file", type=Path, nargs="+", help="the path to the file")
    parser.add_argument("--commit", action="store_true", required=False, help="commit changes to file")
    parser.add_argument("--keep", action="store_true", required=False, help="keep original file")
    parser.add_argument("--log-file", type=Path, required=True, help="write change events to log file")

    args = parser.parse_args()

    for file in args.file:
        main(file, args.commit, args.keep, args.log_file)
