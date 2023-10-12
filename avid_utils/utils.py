from datetime import datetime
from pathlib import Path
from shutil import get_terminal_size
from sys import stdout
from typing import Callable
from typing import Optional
from typing import TextIO

from click import Context
from click import Parameter


def get_parameter(ctx: Context, param_name: str) -> Parameter | None:
    return next((p for p in ctx.command.params if p.name == param_name), None)


def clear_line():
    print("\r" + (" " * (get_terminal_size((0, 0)).columns - 1)) + "\r", end="", flush=True)


def print_log(log_file: Optional[Path]):
    if log_file:
        def inner(*args, **kwargs):
            print(*args, **kwargs)
            print(datetime.now().isoformat().strip(), (kwargs.get("sep", " ").join(map(str, args)).strip()),
                  file=log_file.open("a"))
    else:
        def inner(*args, **kwargs):
            print(*args, **kwargs)

    return inner


def rmdir(path: Path):
    if not path.is_dir():
        return path.unlink(missing_ok=True)

    for item in path.iterdir():
        rmdir(item)

    path.rmdir()
