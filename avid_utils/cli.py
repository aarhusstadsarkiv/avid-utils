from pathlib import Path

from click import BadParameter
from click import Choice
from click import Context
from click import IntRange
from click import Path as ClickPath
from click import argument
from click import group
from click import option
from click import pass_context

from .add_primary_keys import main as primary_keys
from .clean_empty_columns import clean_sqlite as remove_empty_columns_sqlite
from .clean_empty_columns import clean_xml as remove_empty_columns_xml
from .clean_xml import main as remove_whitespace
from .remove_control_characters import main as remove_control_characters
from .remove_duplicate_rows import main as remove_duplicate_rows
from .remove_tables import main as remove_tables
from .search_encoded_db import main as search_encoded_db
from .utils import get_parameter

path_exists = ClickPath(exists=True, writable=True)
path_folder = ClickPath(exists=True, file_okay=False, writable=True, resolve_path=True, path_type=Path)
path_file = ClickPath(exists=True, dir_okay=False, writable=True, resolve_path=True, path_type=Path)
path_file_not_exists = ClickPath(exists=False, dir_okay=False, writable=True, resolve_path=True, path_type=Path)
log_option = option("--log-file", required=True, type=path_file_not_exists, help="Path of the log file.")
commit_option = option("--commit", is_flag=True, default=False, help="Save changes.")


@group("avid-utils")
def app():
    pass


@app.command("add-primary-key")
@argument("archive", type=path_folder)
@log_option
@pass_context
def app_primary_keys(_ctx: Context, archive: Path, log_file: Path):
    """
    Add missing primary keys to an ARCHIVE.
    """

    primary_keys(archive, log_file)


@app.command("remove-empty-columns")
@argument("archive_type", type=Choice(("archive", "sqlite")))
@argument("archive", type=path_exists)
@commit_option
@log_option
@pass_context
def app_remove_empty_columns(_ctx: Context, archive_type: str, archive: tuple, commit: bool, log_file: Path):
    """
    Take a list of databases or archive folders and check each table
    for empty columns (all values either null or ''). Completely empty
    tables will also be removed.

    Empty columns are removed only if the '--commit' option is used and are otherwise ignored.
    """

    for path in archive:
        if archive_type == "archive":
            remove_empty_columns_xml(path, commit, log_file)
        elif archive_type == "sqlite":
            remove_empty_columns_sqlite(path, commit, log_file)


@app.command("remove-tables")
@argument("archive", type=path_folder)
@argument("tables", nargs=-1, type=IntRange(1))
@option("--empty-tables", is_flag=True, default=False, help="Remove all empty tables.")
@log_option
@pass_context
def app_remove_tables(ctx: Context, archive: Path, tables: tuple[str, ...], empty_tables: bool, log_file: Path):
    """
    Remove tables from a given ARCHIVE.
    """

    if not tables and not empty_tables:
        tables_param = get_parameter(ctx, "tables")
        empty_tables_param = get_parameter(ctx, "empty_tables")
        raise BadParameter(f"Can only be skipped "
                           f"if {empty_tables_param.get_error_hint(ctx)} is used.", ctx, tables_param)
    elif tables and empty_tables:
        tables_param = get_parameter(ctx, "tables")
        empty_tables_param = get_parameter(ctx, "empty_tables")
        raise BadParameter(f"Cannot be used "
                           f"if {tables_param.get_error_hint(ctx)} is used.", ctx, empty_tables_param)

    remove_tables(archive, tables, log_file)


@app.command("remove-duplicate-rows")
@argument("database", type=path_file)
@commit_option
@log_option
@pass_context
def app_remove_duplicate_rows(_ctx: Context, database: Path, commit: bool, log_file: Path):
    """
    Remove duplicate rows from a SQLite DATABASE.
    """

    return remove_duplicate_rows(database, commit, log_file)


@app.command("remove-control-characters")
@argument("file", nargs=-1, type=path_file)
@option("--keep", is_flag=True, default=False, help="Keep original files.")
@commit_option
@log_option
@pass_context
def app_remove_control_characters(_ctx: Context, file: tuple[Path, ...], keep: bool, commit: bool, log_file: Path):
    """
    Remove control characters from text FILEs.

    Removed characters are: 00, 01, 02, 03, 04, 05, 06, 0b, 0e, 0f, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 1a, 1c, 1d, 1e, 1f.
    """

    for path in file:
        remove_control_characters(path, commit, keep, log_file)


@app.command("clean-xml")
@argument("file", nargs=-1, type=path_file)
@option("--keep", is_flag=True, default=False, help="Keep original files.")
@log_option
@pass_context
def app_remove_whitespace(_ctx: Context, file: tuple[Path], keep: bool, log_file: Path):
    """
    Minimize XML files and trim content between tags.
    """
    for path in file:
        remove_whitespace(path, keep, log_file)


app.add_command(search_encoded_db)
