from dataclasses import asdict
from dataclasses import dataclass
from datetime import timedelta
from functools import cached_property
from hashlib import algorithms_available
from hashlib import new as new_hash
from json import loads
from pathlib import Path
from sqlite3 import Connection
from struct import calcsize
from struct import pack
from struct import unpack
from timeit import timeit
from typing import Any
from typing import BinaryIO
from typing import Generator
from typing import Optional

from click import argument
from click import Choice
from click import group
from click import IntRange
from click import option
from click import Path as ClickPath
from orjson import dumps
from orjson import loads

sql_types_int: dict[str, int] = {
    "int": 0,
    "integer": 0,
    "text": 1,
    "blob": 2,
    "real": 3,
    "numeric": 4,
}


@dataclass
class ColInfo:
    cid: int
    name: str
    type: str
    notnull: bool
    dflt_value: Optional[Any]
    pk: bool

    @property
    def byte_type(self) -> bytes:
        return bytes([sql_types_int[self.type.lower()]])


@dataclass
class TableInfo:
    name: str
    rows: int
    columns: list[ColInfo]


@dataclass
class Header:
    hash_algorithm: str
    preserve_types: bool
    tables: list[TableInfo]
    bytes_data: Optional[bytes] = None

    @property
    def length(self) -> int:
        return len(self.bytes_data or self.bytes)

    @property
    def total_length(self) -> int:
        return calcsize("<L") + self.length

    @property
    def bytes(self) -> bytes:
        return dumps(asdict(self))

    @classmethod
    def from_handle(cls, handle: BinaryIO) -> 'Header':
        handle.seek(0)
        length: int = unpack("<L", handle.read(4))[0]
        bytes_data: bytes = handle.read(length)
        data: dict = loads(bytes_data)
        tables: list[TableInfo] = [
            TableInfo(name=t["name"], rows=t["rows"], columns=[ColInfo(**c) for c in t["columns"]])
            for t in data["tables"]
        ]

        return Header(
            hash_algorithm=data["hash_algorithm"], preserve_types=data["preserve_types"],
            tables=tables, bytes_data=bytes_data
        )

    def to_bytes(self):
        return pack("<L", self.length) + self.bytes


def get_columns(conn: Connection, table: str) -> list[ColInfo]:
    return [ColInfo(*c) for c in conn.execute(f'pragma table_info("{table}")').fetchall()]


# noinspection SqlNoDataSourceInspection,SqlResolve
def count_rows(conn: Connection, table: str, sample: Optional[int]) -> int:
    rows: int = conn.execute(f'select count(*) from "{table}"').fetchone()[0]
    return min(sample, rows) if sample else rows


def encode_table_column(value: Any, type_byte: bytes, hash_algorithm: str, preserve_types: bool) -> bytes:
    type_byte = type_byte if preserve_types else bytes([1])
    value = value if preserve_types else str(value)
    return (
            type_byte +
            new_hash(hash_algorithm, value if isinstance(value, bytes) else dumps(value, default=str)).digest()
    )


# noinspection SqlNoDataSourceInspection,SqlResolve
def encode_table_rows(
        conn: Connection, table: str, hash_algorithm: str, preserve_types: bool, sample: Optional[int]
) -> Generator[bytes, None, None]:
    columns: list[ColInfo] = get_columns(conn, table)
    sql: str = f'select * from "{table}" limit "{sample}"' if sample else f'select * from "{table}"'

    return (
        encode_table_column(row[col.cid], col.byte_type, hash_algorithm, preserve_types)
        for row in conn.execute(sql)
        for col in columns
    )


# noinspection SqlNoDataSourceInspection,SqlResolve
def encode_database(conn: Connection, file: Path, hash_algorithm: str, preserve_types: bool, sample: Optional[int]):
    tables: list[str] = [t for [t] in conn.execute("select name from sqlite_master where type = 'table'")]

    header: Header = Header(hash_algorithm=hash_algorithm, tables=[], preserve_types=preserve_types)

    for i, table in enumerate(tables):
        print(f"Getting header for table {i + 1} '{table}' ...", end=" ", flush=True)
        header.tables.append(
            TableInfo(
                name=table,
                rows=count_rows(conn, table, sample),
                columns=get_columns(conn, table)
            )
        )
        print("Done")

    with file.open("wb") as fh:
        print("Writing header ...", end=" ", flush=True)
        fh.write(header.to_bytes())
        print("Done")

        for i, table_info in enumerate(header.tables):
            print(f"Writing table {i + 1} rows '{table_info.name}' ...", end=" ", flush=True)
            rows = encode_table_rows(conn, table_info.name, hash_algorithm, preserve_types, sample)
            for j, row in enumerate(rows):
                fh.write(row)
                if (j % 1000) == 0:
                    print(
                        f"\rWriting table {i + 1} rows '{table_info.name}' ... {j / table_info.rows:02.01f}%",
                        end=" ", flush=True
                    )
            print(f"\rWriting table {i + 1} rows '{table_info.name}' ... Done  ")


class Database:
    def __init__(self, file: Path):
        self.file: Path = file
        self.handle: BinaryIO = file.open("rb")
        self.header = Header.from_handle(self.handle)
        self.tables: dict[str, TableInfo] = {t.name.lower(): t for t in self.header.tables}

    def seek(self, offset: int) -> int:
        return self.handle.seek(offset)

    def read(self, size: int) -> bytes:
        return self.handle.read(size)

    def seek_read(self, offset: int, size: int) -> bytes:
        self.seek(offset)
        return self.read(size)

    def encode_value(self, value_type: str, value: Any) -> bytes:
        return encode_table_column(
            value, bytes([sql_types_int[value_type]]), self.header.hash_algorithm,
            self.header.preserve_types
        )

    @cached_property
    def null_hash(self) -> bytes:
        return encode_table_column(None, bytes([0]), self.header.hash_algorithm, self.header.preserve_types)[1:]

    @cached_property
    def hash_length(self) -> int:
        return new_hash(self.header.hash_algorithm, bytes(1)).digest_size + 1

    @cached_property
    def data_start(self) -> int:
        return self.header.total_length

    @property
    def data_size(self) -> int:
        return self.file.stat().st_size - self.data_start

    def table_size(self, table: str) -> int:
        return self.tables[table.lower()].rows * len(self.tables[table.lower()].columns) * self.hash_length

    def table_offset_start(self, table: str, row: int = 0, column: int = 0) -> int:
        table_info: TableInfo = self.tables[table.lower()]
        table_index: int = [t.name.lower() for t in self.header.tables].index(table_info.name)
        offset_tables: list[TableInfo] = self.header.tables[:table_index]
        rows_offset: int = sum((self.table_size(t.name) for t in offset_tables), 0)
        rows_offset += row * len(table_info.columns) * self.hash_length
        columns_offset: int = column * self.hash_length
        return self.data_start + rows_offset + columns_offset

    def table_offset_end(self, table: str) -> int:
        return self.table_offset_start(table) + self.table_size(table) - 1


def print_all_results(results: list[tuple[TableInfo, list[int]]]):
    for table, blocks in results:
        print(
            f"Found match in '{table.name}'", f"{(blocks[0] // len(table.columns)) + 1}:"
                                              f"{','.join(str((b % len(table.columns)) + 1) for b in blocks)}",
            ' '.join(f"'{table.columns[b % len(table.columns)].name}'" for b in blocks)
        )

    print(
        f"{len(results)} matches found",
        f"across {len(set(t.name for [t, _] in results))} tables." if results else ""
    )


def print_aggregated_results(results: list[tuple[TableInfo, list[int]]]):
    tables: dict[str, TableInfo] = {}
    tables_results: dict[(str, str), int] = {}

    for table, blocks in results:
        columns: tuple[str, ...] = tuple(
            sorted(
                (table.columns[block % len(table.columns)].name for block in blocks),
                key=[c.name for c in table.columns].index
            )
        )
        tables[table.name] = table
        tables_results[(table.name, columns)] = tables_results.get((table.name, columns), 0) + 1

    sorter = (lambda tc: (tc[0][0], [c.name for c in tables[tc[0][0]].columns].index(tc[0][1][0])))

    for [table, columns], count in sorted(tables_results.items(), key=sorter):
        print(f"Found {count} matches in '{table}' in columns", ' '.join(f"'{c}'" for c in columns))

    print(f"Found {len(results)} matches", f"across {len(set(t.name for [t, _] in results))} tables" if results else "")


def sort_results(output: list[tuple[TableInfo, list[int]]]) -> list[tuple[TableInfo, list[int]]]:
    return sorted(output, key=lambda r: (r[0].name.lower(), min(r[1])))


def find_value_in_region(
        file: Path, value_hash: bytes, table: TableInfo, start: int, end: int, max_results: int = 0
) -> list[int]:
    if not table.columns:
        return []

    results: list[int] = []

    line: str = f"Searching table '{table.name}' ... "
    print(line, end="", flush=True)

    with file.open("rb") as fh:
        fh.seek(start)
        hash_length: int = len(value_hash)
        blocks: int = (end - start) // hash_length
        for block_number in range(blocks):
            if fh.read(hash_length) == value_hash:
                results.append(block_number)
                if max_results and len(results) >= max_results:
                    break

    print("\r" + (" " * len(line)) + "\r", end="", flush=True)

    return results


def find_values_in_region(
        file: Path, value_hashes: set[bytes], table: TableInfo, start: int, end: int, _max_results: int = 0
) -> list[list[int]]:
    if len(value_hashes) > len(table.columns):
        return []

    line: str = f"Searching table '{table.name}' ... "
    results: list[list[int]] = []

    print(line, end="", flush=True)

    with file.open("rb") as fh:
        fh.seek(start)
        hash_length: int = len(list(value_hashes).pop())
        columns: int = len(table.columns)
        blocks: int = (end - start) // hash_length
        for block_number in range(0, blocks, columns):
            value_hashes_copy = value_hashes.copy()
            match_blocks = []
            for column in range(columns):
                block = fh.read(hash_length)
                if block in value_hashes_copy:
                    value_hashes_copy.remove(block)
                    match_blocks.append(block_number + column)
                    if not value_hashes_copy:
                        results.append(match_blocks)

    print("\r" + (" " * len(line)) + "\r", end="", flush=True)

    return results


def find_value_parent(
        db: Database, value_hash: bytes, exclude: list[str], max_results: int
) -> list[tuple[TableInfo, list[int]]]:
    exclude = [e.lower() for e in exclude or []]
    output: list[tuple[TableInfo, list[int]]] = []

    for table in db.header.tables:
        if table.name.lower() in exclude:
            continue
        table_output = find_value_in_region(
            db.file, value_hash, table, db.table_offset_start(table.name),
            db.table_offset_end(table.name), max_results
        )
        if table_output:
            output.extend(((table, [b]) for b in table_output))

    return sort_results(output)


def find_values_parent(
        db: Database, value_hashes: set[bytes], exclude: list[str], max_results: int
) -> list[tuple[TableInfo, list[int]]]:
    exclude = [e.lower() for e in exclude or []]
    output: list[tuple[TableInfo, list[int]]] = []

    for table in db.header.tables:
        if table.name.lower() in exclude:
            continue
        table_output = find_values_in_region(
            db.file, value_hashes, table, db.table_offset_start(table.name),
            db.table_offset_end(table.name), max_results
        )
        output.extend((table, xs) for xs in table_output if xs)

    return sort_results(output)


# noinspection DuplicatedCode
def find_value(
        db: Database, value_type: str, value_serialised: str, *, max_results: int, exclude_null: bool
) -> list[tuple[TableInfo, list[int]]]:
    value_hash: bytes

    if not db.header.preserve_types or value_type == "text":
        value_hash = db.encode_value("text", value_serialised)
    elif value_type == "blob":
        value_hash = bytes([int(value_serialised[n:n + 2], base=16) for n in range(0, len(value_serialised), 2)])
    else:
        value_hash = db.encode_value(value_type, loads(value_serialised))

    if exclude_null and value_hash == db.null_hash:
        print("Value is null")
        return []

    print(f"Searching for {value_type.upper()} value {value_serialised}")
    print(*(f"{b:02x}" for b in value_hash), end="\n\n")

    return find_value_parent(db, value_hash, [], max_results)


# noinspection DuplicatedCode
def find_values(
        db: Database, values: tuple[tuple[str, str]], *, max_results: int, exclude_null: bool
) -> list[tuple[TableInfo, list[int]]]:
    value_hashes: set[bytes] = set()

    for value_type, value_serialised in values:
        if not db.header.preserve_types or value_type == "text":
            value_hash = db.encode_value("text", value_serialised)
        elif value_type == "blob":
            value_hash = bytes([int(value_serialised[n:n + 2], base=16) for n in range(0, len(value_serialised), 2)])
        else:
            value_hash = db.encode_value(value_type, loads(value_serialised))

        if exclude_null and value_hash == db.null_hash:
            continue

        value_hashes.add(value_hash)

        print(f"Searching for {value_type.upper()} value {value_serialised}")
        print(*(f"{b:02x}" for b in value_hash))

    if not value_hashes:
        print("No Values to search")
        return []

    print()

    return find_values_parent(db, value_hashes, [], max_results)


def find_cell(
        db: Database, table: str, row: int, column: int, *, max_results: int, exclude_null: bool
) -> list[tuple[TableInfo, list[int]]]:
    value_hash: bytes = db.seek_read(db.table_offset_start(table, row - 1, column - 1), db.hash_length)

    print(f"Searching for '{table}' R{row}C{column}")

    if exclude_null and value_hash == db.null_hash:
        print("Skipping null value.")
        return []

    print(*(f"{b:02x}" for b in value_hash), end="\n\n")

    return find_value_parent(db, value_hash, [], max_results)


def find_column(
        db: Database, table: str, column: int, *, max_results: int, exclude_null: bool
) -> list[tuple[TableInfo, list[int]]]:
    table_info: TableInfo = db.tables[table.lower()]

    assert 0 < column <= len(table_info.columns), f"Column {column} does not exist (max {len(table_info.columns)})"

    values_hashes: set[bytes] = set()
    results: list[tuple[TableInfo, list[int]]] = []

    for row in range(table_info.rows):
        value_hash: bytes = db.seek_read(db.table_offset_start(table, row, column - 1), db.hash_length)

        print(f"Searching for '{table}' R{row + 1}C{column}")

        if exclude_null and value_hash == db.null_hash:
            print("Skipping null value")
            continue
        elif value_hash in values_hashes:
            print("Skipping searched value")
            continue

        print(*(f"{b:02x}" for b in value_hash), end="\n\n")

        row_results = find_value_parent(db, value_hash, [table_info.name], max_results - len(results))

        print(f"Found {len(row_results)} matches", end="\n\n" if row < table_info.rows - 1 else "\n")

        results.extend(row_results)
        values_hashes.add(value_hash)

        if len(results) >= max_results:
            break

    return results


def timer(text: str):
    def inner(func):
        def func_new(*args, **kwargs):
            t = timeit(lambda: func(*args, **kwargs), number=1)
            print(f"\n{text}", timedelta(seconds=t))

        return func_new

    return inner


@group("search-encoded-db")
def main():
    """
    This program converts SQLite databases into specially encoded files
    for faster search of relationships between tables.

    The first step is to encode the database using the 'encode' command.

    Once the encoded file is ready, the 'search' commands can look for specific values.
    """


# noinspection GrazieInspection
@main.command("encode", short_help="Encode a database.")
@argument("file", required=True, type=ClickPath(exists=True, dir_okay=False, resolve_path=True, path_type=Path))
@argument(
    "output", required=False, default=None,
    type=ClickPath(exists=False, dir_okay=False, resolve_path=True, path_type=Path)
)
@option(
    "--hash", "hash_algo", metavar="NAME", type=Choice(sorted(algorithms_available)), default="md5",
    show_default=True, help="The hash algorithm to use."
)
@option(
    "--sample", metavar="ROWS", type=IntRange(1), default=None,
    help="Encode a random sample of ROWS rows for each table."
)
@option("--ignore-types", is_flag=True, default=False, help="Do not encode type information.")
@timer("Converted database in")
def encode(file: Path, output: Path, hash_algo: str, ignore_types: bool, sample: Optional[int]):
    """
    Encode a SQLite database FILE into a searchable format containing the hashes of each cell's value.

    The result file will be saved as OUTPUT or as FILE.dat.

    Always available hash algorithms are: blake2b, blake2s, md5, sha1, sha224, sha256, sha384, sha3_224, sha3_256,
    sha3_384, sha3_512, sha512, shake_128, shake_256.
    """
    conn: Connection = Connection(file)
    encode_database(conn, output or file.with_suffix(file.suffix + ".dat"), hash_algo, not ignore_types, sample)


@main.command("search", short_help="Search an encoded database.")
@argument("file", required=1, type=ClickPath(exists=True, dir_okay=False, resolve_path=True, path_type=Path))
@option(
    "--value", metavar="<SQL-TYPE JSON-VALUE>...", type=(Choice(list(sql_types_int.keys())), str),
    multiple=True, help="Search for specific values."
)
@option(
    "--cell", metavar="<TABLE ROW COLUMN>", type=(str, IntRange(1), IntRange(1)),
    help="Search for the value in a cell."
)
@option(
    "--column", metavar="<TABLE COLUMN>", type=(str, IntRange(1)),
    help="Search for all values in column."
)
@option("--max-results", metavar="INTEGER", type=IntRange(1), help="Stop after INTEGER results.")
@option("--include-null", is_flag=True, default=False, help="Do not skip null values.")
@option("--show-all-results", is_flag=True, default=False, help="Do not aggregate results.")
@timer("Search completed in")
def find(
        file: Path, value: tuple[tuple[str, str]], cell: Optional[tuple[str, int, int]],
        column: Optional[tuple[str, int]], max_results: Optional[int], include_null: bool, show_all_results: bool
):
    """
    Search for specific values, cells, or columns inside an encoded FILE.

    The '--max-results' option cannot be used when searching for multiple values with the '--value' option.

    See 'find-relations encode' for help on encoding a database.
    """
    db: Database = Database(file)
    t1: float
    t2: float
    results: list[tuple[TableInfo, list[int]]]

    if cell:
        results = find_cell(db, *cell, max_results=max_results or 0, exclude_null=not include_null)
    elif len(value) == 1:
        results = find_value(db, *value[0], max_results=max_results or 0, exclude_null=not include_null)
    elif value:
        if max_results:
            print("--max-results cannot be used when searching multiple values")
        results = find_values(db, value, max_results=max_results or 0, exclude_null=not include_null)
    elif column:
        results = find_column(db, *column, max_results=max_results or 0, exclude_null=not include_null)
    else:
        raise NotImplemented()

    if show_all_results:
        print_all_results(results)
    else:
        print_aggregated_results(results)
