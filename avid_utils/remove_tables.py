from functools import reduce
from pathlib import Path
from typing import Optional
from typing import Sequence

from xmltodict import parse as parse_xml

from .clean_empty_columns import table_index_update
from .clean_empty_columns import table_xml_update
from .clean_empty_columns import table_xsd_update
from .utils import print_log
from .utils import rmdir


# noinspection DuplicatedCode
def main(archive: Path, table_names: Sequence[str], log_file: Optional[Path]):
    echo = print_log(log_file)
    table_names = list(map(str.lower, table_names))

    tables_index_path: Path = archive.joinpath("Indices", "tableIndex.xml")
    tables_index: dict = parse_xml(tables_index_path.read_text())
    tables: list[dict] = tables_index["siardDiark"]["tables"]["table"]

    table_ids: list[int] = [int(t.removeprefix("table")) for t in map(str.lower, table_names)]

    if not table_ids:
        table_ids = [int(t["folder"].lower().removeprefix("table")) for t in tables if t["rows"] == "0"]

    tables_to_remove: list[int] = [int(t["folder"].lower().removeprefix("table")) for t in tables]
    tables_to_remove = [ti for ti in tables_to_remove if ti in table_ids]

    if not tables_to_remove:
        echo(f"{archive.name}/no tables to remove")
        return

    try:
        for table in sorted(tables, key=lambda t: int(t["folder"].removeprefix("table"))):
            index = int(table["folder"].removeprefix("table"))
            table_folder: Path = archive.joinpath("tables", table["folder"])

            if index in tables_to_remove:
                echo(f"{archive.name}/{table['folder']}/{table['name']}/removed")
                rmdir(archive.joinpath("tables", table["folder"]))
                continue
            elif index <= min(tables_to_remove, default=-1):
                continue
            elif not table_folder.is_dir():
                echo(f"{archive.name}/{table['folder']}/{table['name']}/folder not found")
                continue

            index_diff: int = reduce(lambda p, c: (p + 1) if c < index else p, tables_to_remove, 0)
            new_index: int = index - index_diff

            if not index_diff:
                echo(f"{archive.name}/{table['folder']}/{table['name']}/not modified")
                continue

            print(f"{archive.name}/{table['folder']}/{table['name']}/moving to table{new_index}", end="", flush=True)

            xml_path: Path = table_folder.joinpath(table["folder"]).with_suffix(".xml")
            xml_path_tmp = table_xml_update(xml_path, new_index, [], xml_path.with_name("." + xml_path.name))
            xml_path.unlink(missing_ok=True)
            xml_path_tmp.rename(xml_path.with_name(f"table{new_index}.xml"))

            xsd_path: Path = table_folder.joinpath(table["folder"]).with_suffix(".xsd")
            table_xsd_update(xsd_path, new_index, [], xsd_path)
            xsd_path.rename(xsd_path.with_name(f"table{new_index}.xsd"))

            if new_index != index:
                xml_path.parent.rename(xml_path.parent.with_name(f"table{new_index}"))

            echo(f"\r{archive.name}/{table['folder']}/{table['name']}/moved to table{new_index}   ")

        table_index_update(tables_index_path, [], tables_to_remove, tables_index_path)
    except (Exception, BaseException) as err:
        print()
        echo("ERROR: The operation was interrupted before all changes could be written.",
             f"Archive {archive.name} is likely corrupted.")
        print()
        raise err
