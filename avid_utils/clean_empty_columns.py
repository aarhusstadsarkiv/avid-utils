from argparse import ArgumentParser
from copy import deepcopy
from functools import reduce
from os import environ
from pathlib import Path
from sqlite3 import Connection
from sqlite3 import connect
from typing import Optional

from xmltodict import ParsingInterrupted
from xmltodict import parse as parse_xml
from xmltodict import unparse as unparse_xml

from .utils import print_log
from .utils import rmdir


# noinspection SqlNoDataSourceInspection,SqlResolve
def sqlite_get_tables(conn: Connection) -> list[str]:
    """
    Get the names of all the tables in a database.
    """
    return [
        t for [t] in
        conn.execute("select name from sqlite_master where type = 'table' and name != 'sqlite_master'")
    ]


def sqlite_get_columns(conn: Connection, table: str) -> list[str]:
    """
    Get the names of all the columns in a table that are not primary keys.
    """
    return [c[1] for c in conn.execute(f'pragma table_info("{table}")').fetchall() if not c[5]]


# noinspection SqlNoDataSourceInspection,SqlResolve
def sqlite_has_value(conn: Connection, table: str, column: str) -> bool:
    """
    Check if any row in the table has a non-empty value in a specific column.
    """
    return conn.execute(
        f'select "{column}" from "{table}" where "{column}" is not null and "{column}" != \'\' limit 1'
    ).fetchone() is not None


# noinspection SqlNoDataSourceInspection,SqlResolve
def sqlite_drop_column(conn: Connection, table: str, column: str):
    """
    Drop a column from a table.
    """
    return conn.execute(f'alter table "{table}" drop column "{column}"')


# noinspection SqlNoDataSourceInspection,SqlResolve
def sqlite_drop_table(conn: Connection, table: str):
    """
    Drop a table from a database.
    """
    return conn.execute(f'drop table "{table}"')


def table_index_update(path: Path, remove_columns: list[tuple[int, set[str]]], remove_tables: list[int],
                       out_path: Optional[Path] = None) -> Path:
    out_path = out_path or path.with_suffix(".new" + path.suffix)

    tables_index = parse_xml(path.read_text("utf-8"), force_list=True)
    new_table_index = deepcopy(tables_index)
    new_table_index["siardDiark"][0]["tables"][0]["table"] = []

    for table in deepcopy(tables_index["siardDiark"][0]["tables"][0]["table"]):
        index: int = int(table["folder"][0].removeprefix("table"))
        if index in remove_tables:
            continue
        index -= reduce(lambda p, c: (p + 1) if c < index else p, remove_tables, 0)
        table["folder"][0] = f"table{index}"
        _remove_columns: set[str] = next((cs for t, cs in remove_columns if t == index), set())
        columns: list[dict] = table["columns"][0]["column"]
        table["columns"][0]["column"] = [c for c in columns if c["columnID"][0] not in _remove_columns]
        for column in table["columns"][0]["column"]:
            col_id: int = int(column["columnID"][0].removeprefix("c"))
            col_id -= reduce(lambda p, c: (p + 1) if int(c.removeprefix("c")) < col_id else p, _remove_columns, 0)
            column["columnID"][0] = f"c{col_id}"
        new_table_index["siardDiark"][0]["tables"][0]["table"].append(table)

    with out_path.open("w") as fh:
        unparse_xml(new_table_index, fh, "utf-8")

    return out_path


# noinspection HttpUrlsUsage
def table_xml_update(path: Path, index: int, remove_columns: list[str], out_path: Optional[Path] = None) -> Path:
    remove_columns_indices = [int(c.removeprefix("c")) for c in remove_columns]
    out_path = out_path or path.with_suffix(".new" + path.suffix)

    with path.open("rb") as fi:
        if remove_columns:
            with out_path.open("w", encoding="utf-8") as fo:
                fo.write('<?xml version="1.0" encoding="UTF-8" ?>\n')
                fo.write(
                    f'<table '
                    f'xsi:schemaLocation="http://www.sa.dk/xmlns/siard/1.0/schema0/table{index}.xsd '
                    f'./table{index}.xsd" '
                    f'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                    f'xmlns="http://www.sa.dk/xmlns/siard/1.0/schema0/table{index}.xsd">\n')

                def callback(_, row: dict):
                    new_row: dict = {}
                    for col_id, col in row.items():
                        if col_id in remove_columns:
                            continue
                        col_index: int = int(col_id.removeprefix("c"))
                        col_index -= reduce(lambda p, c: (p + 1) if c < col_index else p, remove_columns_indices, 0)
                        new_row[f"c{col_index}"] = col
                    unparse_xml({"row": new_row}, fo, "utf-8", full_document=False)
                    fo.write("\n")
                    return True

                parse_xml(fi, item_depth=2, item_callback=callback, force_list=True)
                fo.write('</table>')
        else:
            with out_path.open("wb") as fo:
                fo.write('<?xml version="1.0" encoding="UTF-8" ?>\n'.encode())
                fo.write(
                    (
                        f'<table '
                        f'xsi:schemaLocation="http://www.sa.dk/xmlns/siard/1.0/schema0/table{index}.xsd '
                        f'./table{index}.xsd" '
                        f'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                        f'xmlns="http://www.sa.dk/xmlns/siard/1.0/schema0/table{index}.xsd">\n'
                    ).encode()
                )

                chunk: bytes = bytes(5)
                while chunk != b"<row>":
                    new_byte: bytes = fi.read(1)
                    if not new_byte:
                        chunk = "</table>".encode()
                        break
                    chunk = chunk[1:] + new_byte

                while chunk:
                    fo.write(chunk)
                    chunk = fi.read(10_000_000)

    return out_path


# noinspection HttpUrlsUsage
def table_xsd_update(path: Path, table_index: int, remove_columns: list[str], out_path: Optional[Path] = None):
    remove_columns_indices = [int(c.removeprefix("c")) for c in remove_columns]
    out_path = out_path or path.with_suffix(".new" + path.suffix)

    xsd = parse_xml(path.read_bytes(), "utf-8", force_list=True)
    xsd["xs:schema"][0]["@xmlns"] = f"http://www.sa.dk/xmlns/siard/1.0/schema0/table{table_index}.xsd"
    xsd["xs:schema"][0]["@targetNamespace"] = f"http://www.sa.dk/xmlns/siard/1.0/schema0/table{table_index}.xsd"
    xsd["xs:schema"][0]["xs:complexType"][0]["xs:sequence"][0]["xs:element"] = [
        column
        for column in xsd["xs:schema"][0]["xs:complexType"][0]["xs:sequence"][0]["xs:element"]
        if column['@name'] not in remove_columns
    ]
    for column in xsd["xs:schema"][0]["xs:complexType"][0]["xs:sequence"][0]["xs:element"]:
        column_index: int = int(column["@name"].removeprefix("c"))
        if column_index < min(remove_columns_indices, default=-1):
            continue
        column_index_diff: int = reduce(lambda p, c: (p + 1) if c < column_index else p, remove_columns_indices, 0)
        column["@name"] = f"c{column_index - column_index_diff}"

    with out_path.open("w") as fh:
        unparse_xml(xsd, fh, "utf-8")

    return out_path


# noinspection SqlNoDataSourceInspection
def clean_sqlite(file: Path, commit: bool, log_file: Optional[Path]):
    echo = print_log(log_file)

    print(file.name)

    # Connect to the database
    environ["SQLITE_TMPDIR"] = str(file.parent.resolve())
    conn: Connection = connect(file)

    columns_to_remove: dict[str, list[str]] = {}

    for table in sqlite_get_tables(conn):
        for column in sqlite_get_columns(conn, table):
            # Prepare output string
            line = f"{file.name}/{table}/{column}"
            print(line, end="... ", flush=True)

            if sqlite_has_value(conn, table, column):
                # If the column is not empty, clear the output line
                print("\r" + (" " * (len(line) + 4)) + "\r", end="", flush=True)
            else:
                echo(f"\r{line}/empty")
                if commit:
                    columns_to_remove[table] = columns_to_remove.get(table, []) + [column]

    if columns_to_remove and commit:
        try:
            for table, columns in columns_to_remove.items():
                if set(columns) == set(sqlite_get_columns(conn, table)):
                    # If all columns are empty, remove table
                    print(f"{file.name}/{table}/removing...", end="", flush=True)
                    sqlite_drop_table(conn, table)
                    echo(f"\r{file.name}/{table}/removed    ")
                else:
                    # Remove one column at a time
                    for column in columns:
                        print(f"{file.name}/{table}/{column}/removing...", end="", flush=True)
                        sqlite_drop_column(conn, table, column)
                        echo(f"\r{file.name}/{table}/{column}/removed    ")

            # Show temporary message during cleanup
            line = f"{file.name}/cleaning..."
            print(line, end="", flush=True)

            # Commit all changes and clean the database with vacuum
            conn.commit()
            conn.execute("vacuum")

            print("\r" + (" " * len(line)) + "\r", end="", flush=True)
        except Exception as err:
            echo(f"ERROR: {err!r}")
            echo("ERROR: Changes interrupted before committing")
            raise

    conn.close()


# noinspection DuplicatedCode
def clean_xml(archive: Path, commit: bool, log_file: Optional[Path]):
    echo = print_log(log_file)

    print(archive.name)

    tables_index_path: Path = archive.joinpath("Indices", "tableIndex.xml")
    tables_index: dict = parse_xml(tables_index_path.read_text(), force_list=True)

    tables: list[dict] = tables_index["siardDiark"][0]["tables"][0]["table"]
    tables_to_remove: list[int] = []
    columns_to_remove: list[tuple[int, set[str]]] = []

    for table in tables:
        line: str = f"{archive.name}/{table['folder'][0]}/{table['name'][0]}..."
        print(line, end="", flush=True)
        xml_path: Path = archive.joinpath("tables", table["folder"][0], table["folder"][0]).with_suffix(".xml")
        columns: list[dict] = table["columns"][0]["column"]
        empty_columns: set[str] = {c["columnID"][0] for c in columns}

        def callback(_, row):
            _empty_columns: list[str] = []

            for col_id in empty_columns:
                value = row[col_id]
                if isinstance(value, dict):
                    if value.get("@xsi:nil", None) != "true":
                        _empty_columns.append(col_id)
                elif value:
                    _empty_columns.append(col_id)

            empty_columns.difference_update(_empty_columns)

            return len(empty_columns) > 0

        try:
            parse_xml(xml_path.open("rb"), item_depth=2, item_callback=callback)
        except ParsingInterrupted:
            pass

        if len(empty_columns) == len(columns):
            tables_to_remove.append(int(table["folder"][0].removeprefix("table")))
            echo(f"\r{archive.name}/{table['folder'][0]}/{table['name'][0]}/empty")
        elif empty_columns:
            columns_to_remove.append((int(table["folder"][0].removeprefix("table")), empty_columns))
            for column in [c for c in columns if c["columnID"][0] in empty_columns]:
                echo(f"\r{archive.name}/{table['folder'][0]}/{table['name'][0]}/"
                     f"{column['columnID'][0]}/{column['name'][0]}/empty")
        else:
            print("\r" + (" " * len(line)) + "\r", end="", flush=True)

    if (tables_to_remove or columns_to_remove) and commit:
        try:
            table_index_update(tables_index_path, columns_to_remove, tables_to_remove, tables_index_path)

            if tables_to_remove:
                for table in sorted(tables, key=lambda t: int(t["folder"][0].removeprefix("table"))):
                    index = int(table["folder"][0].removeprefix("table"))
                    table_folder: Path = archive.joinpath("tables", table["folder"][0])

                    if index in tables_to_remove:
                        echo(f"{archive.name}/{table['folder'][0]}/{table['name'][0]}/removed")
                        rmdir(archive.joinpath("tables", table["folder"][0]))
                        continue
                    elif index <= min(tables_to_remove, default=-1):
                        continue
                    elif not table_folder.is_dir():
                        echo(f"{archive.name}/{table['folder'][0]}/{table['name'][0]}/folder not found")
                        continue

                    _columns_to_remove: set[str] = next((cs for t, cs in columns_to_remove if t == index), set())
                    index_diff: int = reduce(lambda p, c: (p + 1) if c < index else p, tables_to_remove, 0)
                    new_index: int = index - index_diff

                    if not index_diff and not _columns_to_remove:
                        continue

                    echo(f"{archive.name}/{table['folder'][0]}/{table['name'][0]}/moved to table{new_index}")

                    xml_path: Path = table_folder.joinpath(table["folder"][0]).with_suffix(".xml")
                    xml_path_tmp = table_xml_update(xml_path, new_index, list(_columns_to_remove),
                                                    xml_path.with_name("." + xml_path.name))
                    xml_path.unlink(missing_ok=True)
                    xml_path_tmp.rename(xml_path.with_name(f"table{new_index}.xml"))

                    xsd_path: Path = table_folder.joinpath(table["folder"][0]).with_suffix(".xsd")
                    table_xsd_update(xsd_path, new_index, list(_columns_to_remove), xsd_path)
                    xsd_path.rename(xsd_path.with_name(f"table{new_index}.xsd"))

                    if new_index != index:
                        xml_path.parent.rename(xml_path.parent.with_name(f"table{new_index}"))

            for index, column_ids in columns_to_remove:
                if tables_to_remove and index > min(tables_to_remove, default=-1):
                    continue
                table: dict = next((t for t in tables if t["folder"][0] == f"table{index}"))
                table_folder: Path = archive.joinpath("tables", table["folder"][0])

                if not table_folder.is_dir():
                    echo(f"{archive.name}/{table['folder'][0]}/{table['name'][0]}/folder not found")
                    continue

                xml_path: Path = table_folder.joinpath(table["folder"][0]).with_suffix(".xml")
                table_xml_update(xml_path, index, list(column_ids), xml_path.with_name("." + xml_path.name))
                xml_path.unlink(missing_ok=True)
                xml_path.with_name("." + xml_path.name).rename(xml_path)

                xsd_path: Path = table_folder.joinpath(table["folder"][0]).with_suffix(".xsd")
                table_xsd_update(xsd_path, index, list(column_ids), xsd_path)

            print(f"\r{archive.name}/{len(tables_to_remove)} tables "
                  f"and {len([c for _, cs in columns_to_remove for c in cs])} columns removed")
        except (Exception, BaseException) as err:
            print()
            echo("ERROR: The operation was interrupted before all changes could be written.",
                 f"Archive {archive.name} is likely corrupted.")
            print()
            raise err


def cli():
    """
    Take a list of databases or archive folders and check each table
    for empty columns (all values either null or ''). Completely empty
    tables will also be removed.

    Empty columns are removed only if the `--commit` option is used and are otherwise ignored.
    """

    parser = ArgumentParser("clean-empty-columns", description=cli.__doc__)
    parser.add_argument("type", choices=["archive", "sqlite"],
                        help="whether the files are archives or SQLite databases")
    parser.add_argument("files", nargs="+", type=Path, help="the databases/archives to clean")
    parser.add_argument("--commit", action="store_true", required=False, help="commit changes to database")
    parser.add_argument("--log-file", type=Path, required=True, help="write change events to log file")

    args = parser.parse_args()

    if args.type == "sqlite":
        for file in args.files:
            clean_sqlite(file, args.commit, args.log_file)
    elif args.type == "archive":
        for archive in args.files:
            clean_xml(archive, args.commit, args.log_file)


if __name__ == '__main__':
    cli()
