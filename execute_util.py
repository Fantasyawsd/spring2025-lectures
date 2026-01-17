"""
Functions such as (e.g., note, image, link) populate the list of renderings,
which will be shown in place of the line of code in the interface.
"""

import os
import inspect
import re
import subprocess
from file_util import cached, relativize
from dataclasses import dataclass
from arxiv_util import is_arxiv_link, arxiv_reference
from reference import Reference

_BILINGUAL_ENABLED = os.getenv("CS336_BILINGUAL", "1").lower() not in {"0", "false", "no"}


def _contains_chinese(text: str) -> bool:
    return re.search(r"[\u4e00-\u9fa5]", text) is not None


def _contains_latin(text: str) -> bool:
    return re.search(r"[A-Za-z]", text) is not None


def _translate_to_zh(message: str) -> str:
    if os.getenv("OPENAI_API_KEY") or os.getenv("TOGETHER_API_KEY"):
        from model_util import query_gpt4o

        prompt = (
            "Translate the following markdown text to Simplified Chinese. "
            "Preserve markdown formatting, links, and code exactly. "
            "Return only the translated text.\n"
            "<text>\n"
            f"{message}\n"
            "</text>"
        )
        return query_gpt4o(prompt=prompt).strip()
    return "（中文翻译待补）"


def _bilingualize(message: str) -> str:
    if not _BILINGUAL_ENABLED:
        return message
    if not message.strip():
        return message
    if _contains_chinese(message) or not _contains_latin(message):
        return message
    translation = _translate_to_zh(message)
    if not translation:
        return message
    return f"{message}\n\n{translation}"

@dataclass(frozen=True)
class CodeLocation:
    """Refers to a specific line of code."""
    path: str
    line_number: int


@dataclass(frozen=True)
class Rendering:
    """
    Specifies what to display instead of a line of code.  Types:
    - text: plain text (verbatim)
    - markdown: to be rendered as markdown
    - image: an image (data = url)
    - link: an link to internal code or external URL
    """
    type: str
    data: str | None = None
    style: dict | None = None
    external_link: Reference | None = None
    internal_link: CodeLocation | None = None

############################################################

def text(message: str, style: dict | None = None, verbatim: bool = False):
    """Make a note (bullet point) with `message`."""
    style = style or {}
    if verbatim:
        messages = message.split("\n")
        style = {
            "fontFamily": "monospace",
            "whiteSpace": "pre",
            **style
        }
    else:
        messages = [_bilingualize(message)]

    for message in messages:
        _current_renderings.append(Rendering(type="markdown", data=message, style=style))


def image(url: str, style: dict | None = None, width: int | str | None = None):
    """Show the image at `url`."""
    style = style or {}
    if width is not None:
        style["width"] = width

    if is_url(url):
        path = cached(url, "image")
    else:
        path = url
        if not os.path.exists(path):
            raise ValueError(f"Image not found: {path}")

    _current_renderings.append(Rendering(type="image", data=path, style=style))


def is_url(url: str) -> bool:
    """Check if `url` looks like a URL."""
    return url.startswith("http")


def link(arg: type | Reference | str | None = None, style: dict | None = None, **kwargs):
    """
    Shows a link.  There are four possible usages:
    1. link(title="...", url="...") [Creates a new reference]
    2. link(arg: Reference) [Shows an existing reference]
    3. link(arg: type) [Shows a link to the code]
    4. link(arg: str) [Creates a new reference with the given URL]
    """
    style = style or {}

    if arg is None:
        reference = Reference(**kwargs)
        _current_renderings.append(Rendering(type="link", style=style, external_link=reference))
    elif isinstance(arg, Reference):
        _current_renderings.append(Rendering(type="link", style=style, external_link=arg))
    elif isinstance(arg, type) or callable(arg):
        path = inspect.getfile(arg)
        _, line_number = inspect.getsourcelines(arg)
        anchor = CodeLocation(relativize(path), line_number)
        _current_renderings.append(Rendering(type="link", data=arg.__name__, style=style, internal_link=anchor))
    elif isinstance(arg, str):
        if is_arxiv_link(arg):
            reference = arxiv_reference(arg)
            _current_renderings.append(Rendering(type="link", style=style, external_link=reference))
        else:
            reference = Reference(url=arg)
            _current_renderings.append(Rendering(type="link", style=style, external_link=reference))
    else:
        raise ValueError(f"Invalid argument: {arg}")

    style = {"color": "gray"}


############################################################

# Accumulate the renderings during execution (gets flushed).
_current_renderings: list[Rendering] = []

def pop_renderings() -> list[Rendering]:
    """Return the renderings and clear the list."""
    renderings = _current_renderings.copy()
    _current_renderings.clear()
    return renderings


def system_text(command: list[str]):
    output = subprocess.check_output(command).decode('utf-8')
    output = remove_ansi_escape_sequences(output)
    text(output, verbatim=True)


def remove_ansi_escape_sequences(input_text: str) -> str:
    ansi_escape_pattern = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_escape_pattern.sub('', input_text)
