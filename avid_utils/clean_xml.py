from pathlib import Path
from re import Pattern
from re import compile as re_compile
from typing import Optional
from typing import TextIO
from xml.sax import ContentHandler
from xml.sax import parse as sax_parse
from xml.sax.saxutils import escape
from xml.sax.saxutils import quoteattr
from xml.sax.xmlreader import AttributesImpl

from .utils import clear_line
from .utils import print_log

_whitespace: str = "".join(map(chr, range(0, 33)))
_col_name: Pattern = re_compile(r"^c\d+$")
_escape_entities: dict[str, str] = {'"': "&quot;"}


class ContentHandlerTrim(ContentHandler):
    def __init__(self, file_handle: TextIO):
        super().__init__()
        self.handle: TextIO = file_handle
        self.current_tag: Optional[str] = None
        self.current_tag_is_col: bool = False
        self.current_content: str = ""

    def startDocument(self):
        self.handle.write('<?xml version="1.0" encoding="UTF-8" ?>\n')

    def startElement(self, name: str, attrs: AttributesImpl):
        self.current_tag = name
        self.current_tag_is_col = _col_name.match(name) is not None
        if not self.current_tag_is_col:
            attrs_string: str = " ".join(f'{n}={quoteattr(v)}' for n, v in attrs.items())
            self.handle.write(f"<{name} {attrs_string}".strip() + ">")

    def characters(self, content: str):
        if self.current_tag:
            self.current_content += content

    def endElement(self, name: str):
        if self.current_tag_is_col:
            self.current_content = self.current_content.strip(_whitespace)
            if self.current_content:
                self.handle.write(f'<{name}>{escape(self.current_content, _escape_entities)}</{name}>')
            else:
                self.handle.write(f'<{name} xsi:nil="true"/>')
        else:
            self.handle.write(f'{escape(self.current_content, _escape_entities)}</{name}>')

        self.current_tag = None
        self.current_tag_is_col = False
        self.current_content = ""


# noinspection HttpUrlsUsage
def main(file: Path, keep: bool, log_file: Path):
    echo = print_log(log_file)
    out_file: Path = file.with_name("." + file.name)

    print(f"{file.name}/cleaning... ", end="", flush=True)

    try:
        with file.open("r", encoding="utf-8") as fi:
            with out_file.open("w", encoding="utf-8") as fo:
                sax_parse(fi, ContentHandlerTrim(fo))
    except (Exception, BaseException):
        out_file.unlink(missing_ok=True)
        raise

    new_size, old_size = out_file.stat().st_size, file.stat().st_size

    clear_line()

    if new_size != old_size:
        if keep:
            file_keep = file.replace(file.with_stem(file.stem + ".old"))
            echo(f"{file.name}/preserved {file_keep.name}")
        out_file.replace(file)
        echo(f"{file.name}/saved {new_size}B")
        echo(f"{file.name}/removed {old_size - new_size}B")
    else:
        echo(f"{file.name}/no changes")
