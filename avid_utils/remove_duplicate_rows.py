from os import environ
from pathlib import Path
from sqlite3 import Connection

from .clean_empty_columns import sqlite_get_tables
from .utils import print_log


def has_primary_keys(conn: Connection, table: str) -> bool:
    return any(c[5] for c in conn.execute(f"pragma table_info({table})").fetchall())


# noinspection SqlNoDataSourceInspection
def count_rows(conn: Connection, table: str) -> int:
    return conn.execute(f"select count(ROWID) from {table}").fetchone()[0]


# noinspection SqlNoDataSourceInspection
def count_unique_rows(conn: Connection, table: str) -> int:
    return conn.execute(f"select count(*) from (select distinct * from {table})").fetchone()[0]


# noinspection SqlNoDataSourceInspection
def remove_duplicates(conn: Connection, table: str):
    tables: list[str] = sqlite_get_tables(conn)

    table_tmp: str = table
    while table_tmp in tables:
        table_tmp = "_" + table_tmp

    conn.execute(f"create table {table_tmp} as select distinct * from {table}")
    conn.execute(f"drop table {table}")
    conn.execute(f"alter table {table_tmp} rename to {table}")


def main(file: Path, commit: bool, log_file: Path):
    echo = print_log(log_file)

    environ["SQLITE_TMPDIR"] = str(file.parent.resolve())
    conn = Connection(file)
    duplicate_tables: list[tuple[str, int]] = []

    for table in sqlite_get_tables(conn):
        line = f"{file.name}/{table}/counting... "
        print(line, end="", flush=True)

        if has_primary_keys(conn, table):
            print("\r" + (" " * len(line)) + "\r", end="", flush=True)
            continue

        rows, unique_rows = count_rows(conn, table), count_unique_rows(conn, table)
        print("\r" + (" " * len(line)) + "\r", end="", flush=True)

        if rows != unique_rows:
            echo(f"{file.name}/{table}/duplicates: {rows - unique_rows} ({rows}, {unique_rows})")
            duplicate_tables.append((table, rows - unique_rows))

    if commit and duplicate_tables:
        try:
            for table, duplicates in duplicate_tables:
                print(f"{file.name}/{table}/cleaning... ", end="", flush=True)
                remove_duplicates(conn, table)
                echo(f"\r{file.name}/{table}/removed {duplicates} duplicates")

            line = f"{file.name}/vacuuming... "
            print(line, end="", flush=True)
            conn.commit()
            conn.execute("vacuum")
            print("\r" + (" " * len(line)) + "\r", end="", flush=True)
        finally:
            conn.commit()
