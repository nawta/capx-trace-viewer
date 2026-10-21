"""Model request and response payloads (docs/trace-format.md, section 6).

Runs on the writer thread, except ``sanitize_url`` and
``redact``/``put_redacted``, which hooks also call on the calling thread for
recorder errors and hook failure reasons (``redact`` reads at most
``REDACT_WINDOW`` characters, in linear time). ``normalize_request()`` parses the exact body that
``requests.Session.send`` sent (``str`` for ``data=json.dumps(...)``, ``bytes``
for ``json=``), moves every ``data:`` URL into a content-addressed asset and
returns placeholders; nothing under a secret key name is moved or listed (its
value is masked whole later). ``reconstruct()`` re-inserts the data URLs and
serialises with ``json.dumps(obj)`` (default separators, ``ensure_ascii=True``)
so the writer can check the sha256 against the sent body *before* masking.
``mask_secrets()`` then replaces values of the exact, case-insensitive key
names of section 6.3, whatever their type. The raw body is never stored.

``redact()`` is applied to every stored exception message, cause, traceback,
error-body excerpt, hook failure reason and runner error text. Its scope is four
rules, listed above ``redact()``: URLs lose their userinfo and query, the
credentials after ``Bearer``, ``Basic``, ``Token`` and ``Digest`` and keys of
known formats (``sk-``, ``AIza``, ``ghp_``, ...) are replaced, and the values of
secret key names are masked; everything else is stored as it is.
``put_redacted()`` stores the result with ``<field>_redacted: true`` when a rule
replaced something in the stored text. The exception that is re-raised is never
changed.

Images are decoded (for ``pixel_sha256``) only when they are PNG files with a
valid header and no APNG chunk, and only below 50 megapixels, so that Pillow
never reaches a code path that emits a Python warning.

Related: ``capx_trace_recorder/hooks/http.py`` (H1, captures the bodies),
``capx_trace_recorder/segments.py`` (``input_summary``), ``capx_trace_recorder/store.py``.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import os
import re
import sys
from typing import Any, NamedTuple
from urllib.parse import urlsplit, urlunsplit

from capx_trace_recorder.config import MASK_VALUE, SECRET_KEYS, _parse_proxy_ports

_DATA_URL = re.compile(r"^data:([^;,]*)((?:;[^;,]*)*?);base64,", re.IGNORECASE)
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
MAX_DECODE_PIXELS = 50_000_000
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})

def known_proxy_ports() -> frozenset[int]:
    """Configured loopback ports that may run an upstream-style proxy."""
    from capx_trace_recorder.store import get_store

    store = get_store()
    return store.config.proxy_ports if store is not None else _parse_proxy_ports(os.environ, [])

_EXT = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
    "video/mp4": "mp4",
    "video/webm": "webm",
    "application/json": "json",
    "text/plain": "txt",
}


def ext_for_mime(mime: str) -> str:
    return _EXT.get((mime or "").lower(), "bin")


def join_path(base: str, key: Any) -> str:
    if isinstance(key, int):
        return f"{base}[{key}]"
    key_s = str(key)
    if _IDENT.match(key_s):
        return f"{base}.{key_s}" if base else key_s
    return f"{base}[{json.dumps(key_s)}]"


# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------


def sanitize_url(url: Any) -> dict[str, Any]:
    """URL without userinfo, query and fragment, plus host, port and path."""
    try:
        parts = urlsplit(str(url))
        host = parts.hostname or ""
        try:
            port = parts.port
        except ValueError:
            port = None
        if port is None:
            port = {"http": 80, "https": 443}.get(parts.scheme)
        netloc = host if ":" not in host else f"[{host}]"
        if parts.port is not None:
            netloc = f"{netloc}:{parts.port}"
        clean = urlunsplit((parts.scheme, netloc, parts.path, "", ""))
        return {
            "url": clean,
            "host": host,
            "port": port,
            "path": parts.path,
            "known_proxy": host in LOOPBACK_HOSTS and port in known_proxy_ports(),
        }
    except TimeoutError:  # the trial alarm (section 2): called on the trial thread
        raise
    except Exception:  # noqa: BLE001
        return {"url": None, "host": None, "port": None, "path": None, "known_proxy": False}


def strip_query(url: str) -> str:
    try:
        parts = urlsplit(url)
        netloc = parts.hostname or ""
        if parts.port is not None:
            netloc = f"{netloc}:{parts.port}"
        return urlunsplit((parts.scheme, netloc, parts.path, "", ""))
    except Exception:  # noqa: BLE001 - timeout-ok: writer thread
        return url.split("?", 1)[0]


# ---------------------------------------------------------------------------
# Body bytes
# ---------------------------------------------------------------------------


def body_to_bytes(body: Any) -> bytes | None:
    """The bytes put on the wire for a ``PreparedRequest.body``.

    ``http.client`` encodes a ``str`` body as ISO-8859-1; ``json.dumps`` output
    with ``ensure_ascii=True`` is ASCII, so this equals UTF-8 in practice.
    """
    if body is None:
        return None
    if isinstance(body, bytes):
        return body
    if isinstance(body, str):
        try:
            return body.encode("iso-8859-1")
        except UnicodeEncodeError:
            return body.encode("utf-8")
    return None


# ---------------------------------------------------------------------------
# Request normalisation
# ---------------------------------------------------------------------------


class Media:
    """One ``data:`` URL moved to an asset."""

    __slots__ = ("path", "order", "mime", "data", "prefix", "canonical", "kind", "placeholder")

    def __init__(self, path: str, order: int, mime: str, data: bytes, prefix: str,
                 canonical: bool) -> None:
        self.path = path
        self.order = order
        self.mime = mime
        self.data = data
        self.prefix = prefix
        self.canonical = canonical
        mime_l = (mime or "").lower()
        self.kind = "video" if mime_l.startswith("video/") else (
            "image" if mime_l.startswith("image/") else "other")
        self.placeholder: dict[str, Any] = {}


class ExternalUrl:
    __slots__ = ("path", "order", "url")

    def __init__(self, path: str, order: int, url: str) -> None:
        self.path = path
        self.order = order
        self.url = url


def _parse_data_url(value: str) -> tuple[str, str, bytes, bool] | None:
    match = _DATA_URL.match(value)
    if not match:
        return None
    mime = match.group(1) or "application/octet-stream"
    prefix = value[: match.end()]
    b64 = value[match.end():]
    try:
        data = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        return None
    canonical = base64.b64encode(data).decode("ascii") == b64
    return mime, prefix, data, canonical


def is_secret_key(key: Any) -> bool:
    return isinstance(key, str) and key.lower() in SECRET_KEYS


def normalize_request(obj: Any) -> tuple[Any, list[Media], list[ExternalUrl]]:
    """Copy of ``obj`` with ``data:`` URLs replaced by placeholder dicts.

    Values under a secret key name (and all their descendants) are left as they
    are: no asset is written from them and no external URL is listed, and
    ``mask_secrets`` later replaces the whole value.
    """
    media: list[Media] = []
    external: list[ExternalUrl] = []
    counter = [0]

    def walk(value: Any, path: str, parent_key: Any, secret: bool = False) -> Any:
        if isinstance(value, dict):
            return {k: walk(v, join_path(path, k), k, secret or is_secret_key(k))
                    for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v, join_path(path, i), i, secret) for i, v in enumerate(value)]
        if isinstance(value, str) and not secret:
            if value[:5].lower() == "data:":
                parsed = _parse_data_url(value)
                if parsed is not None:
                    mime, prefix, data, canonical = parsed
                    item = Media(path, counter[0], mime, data, prefix, canonical)
                    counter[0] += 1
                    media.append(item)
                    return item.placeholder
            elif value[:7].lower() == "http://" or value[:8].lower() == "https://":
                if parent_key in ("url", "image_url", "video_url"):
                    external.append(ExternalUrl(path, counter[0], value))
                    counter[0] += 1
        return value

    return walk(obj, "", None), media, external


def fill_placeholder(item: Media, asset_path: str | None, sha256: str, written: bool,
                     reason: str | None = None) -> None:
    ph = item.placeholder
    ph.clear()
    ph["$asset"] = asset_path
    ph["sha256"] = sha256
    ph["mime"] = item.mime
    ph["bytes"] = len(item.data)
    if item.prefix != f"data:{item.mime};base64,":
        ph["prefix"] = item.prefix
    if not item.canonical:
        ph["noncanonical_base64"] = True
    if not written:
        ph["written"] = False
        ph["reason"] = reason


def reconstruct(normalized: Any, media: list[Media]) -> str:
    """Re-insert the data URLs and serialise the way ``client.py`` does."""
    by_id = {id(m.placeholder): m for m in media}

    def walk(value: Any) -> Any:
        if isinstance(value, dict):
            m = by_id.get(id(value))
            if m is not None:
                return m.prefix + base64.b64encode(m.data).decode("ascii")
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        return value

    return json.dumps(walk(normalized))


def mask_secrets(obj: Any, external: list[ExternalUrl] | None = None) -> tuple[Any, list[str]]:
    """Mask the whole value of every exact secret key name (whatever its type)
    and strip queries of external image URLs."""
    masked: list[str] = []
    ext_paths = {e.path for e in (external or [])}

    def walk(value: Any, path: str) -> Any:
        if isinstance(value, dict):
            out = {}
            for k, v in value.items():
                sub = join_path(path, k)
                if is_secret_key(k):
                    out[k] = MASK_VALUE
                    masked.append(sub)
                else:
                    out[k] = walk(v, sub)
            return out
        if isinstance(value, list):
            return [walk(v, join_path(path, i)) for i, v in enumerate(value)]
        if isinstance(value, str) and path in ext_paths:
            clean = strip_query(value)
            if clean != value:
                masked.append(path + " (query removed)")
            return clean
        return value

    return walk(obj, ""), masked


def _is_placeholder(value: Any) -> bool:
    return isinstance(value, dict) and "$asset" in value


def has_nonfinite(obj: Any) -> bool:
    try:
        json.dumps(obj, allow_nan=False)
        return False
    except ValueError:
        return True


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def image_dimensions(data: bytes) -> tuple[int | None, int | None, str | None]:
    """Width, height and format from the file header (no decoding)."""
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24 and data[12:16] == b"IHDR":
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big"), "png"
    if data[:3] == b"\xff\xd8\xff":
        i = 2
        n = len(data)
        while i + 9 < n:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            seg_len = int.from_bytes(data[i + 2:i + 4], "big")
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                h = int.from_bytes(data[i + 5:i + 7], "big")
                w = int.from_bytes(data[i + 7:i + 9], "big")
                return w, h, "jpeg"
            if seg_len < 2:
                break
            i += 2 + seg_len
        return None, None, "jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
        return int.from_bytes(data[6:8], "little"), int.from_bytes(data[8:10], "little"), "gif"
    return None, None, None


def _png_is_plain(data: bytes) -> bool:
    """A PNG with a valid chunk walk and no APNG control chunk."""
    pos = 8
    n = len(data)
    seen_ihdr = False
    while pos + 8 <= n:
        length = int.from_bytes(data[pos:pos + 4], "big")
        ctype = data[pos + 4:pos + 8]
        if pos == 8:
            seen_ihdr = ctype == b"IHDR"
        if ctype in (b"acTL", b"fcTL", b"fdAT"):
            return False
        if ctype == b"IEND":
            return seen_ihdr
        pos += 12 + length
    return False


def pixel_sha256(data: bytes, width: int | None, height: int | None, fmt: str | None) -> str | None:
    """sha256 of the decoded raw pixels (PNG only; see module docstring)."""
    if fmt != "png" or not width or not height or width * height > MAX_DECODE_PIXELS:
        return None
    if not _png_is_plain(data):
        return None
    np = sys.modules.get("numpy")
    try:
        from PIL import Image
    except Exception:  # noqa: BLE001 - timeout-ok: writer thread
        return None
    if np is None:
        import numpy as np  # noqa: PLC0415 - numpy is always installed with capx
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.format != "PNG":
                return None
            arr = np.asarray(img)
        return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
    except Exception:  # noqa: BLE001 - timeout-ok: writer thread
        return None


def raw_pixel_sha256(arr: Any) -> str | None:
    """sha256 of a uint8 pixel array exactly as the harness produced it (writer thread)."""
    np = sys.modules.get("numpy")
    if np is None:
        return None
    try:
        return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
    except Exception:  # noqa: BLE001 - timeout-ok: writer thread
        return None


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


def parse_response_body(data: bytes | None) -> tuple[Any, dict[str, Any]]:
    """Parsed JSON (or ``None``) and the fields of ``model_response``."""
    fields: dict[str, Any] = {
        "content": None,
        "reasoning": None,
        "usage": None,
        "finish_reason": None,
        "response_model": None,
        "response_id": None,
    }
    if not data:
        return None, fields
    try:
        obj = json.loads(data)
    except Exception:  # noqa: BLE001 - timeout-ok: writer thread
        return None, fields
    if not isinstance(obj, dict):
        return obj, fields
    choices = obj.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        first = choices[0]
        message = first.get("message") if isinstance(first.get("message"), dict) else {}
        content = message.get("content")
        fields["content"] = content if isinstance(content, str) else (
            None if content is None else json.dumps(content))
        for key in ("reasoning", "reasoning_content"):
            value = message.get(key)
            if isinstance(value, str) and value:
                fields["reasoning"] = {"text": value, "field": f"choices[0].message.{key}"}
                break
        finish = first.get("finish_reason")
        fields["finish_reason"] = finish if isinstance(finish, str) else None
    elif isinstance(obj.get("output_text"), str):
        fields["content"] = obj["output_text"]
    usage = obj.get("usage")
    if isinstance(usage, dict):
        fields["usage"] = usage
    if isinstance(obj.get("model"), str):
        fields["response_model"] = obj["model"]
    if isinstance(obj.get("id"), (str, int)):
        fields["response_id"] = obj["id"]
    return obj, fields


# ---------------------------------------------------------------------------
# Redaction of stored texts (docs/trace-format.md, sections 2 and 6.3)
# ---------------------------------------------------------------------------
#
# ``redact()`` removes secrets from a text that the recorder stores: an exception
# message, cause or traceback, an HTTP error body, a hook failure reason, the
# error text of a runner record. It replaces what the four rules below describe
# and nothing else. Names, paths (trial ids included), digests, code and prose
# that no rule covers are stored as they are. The rules run in the order (c),
# (d), (a), (b), each as one left-to-right pass:
# - (c) first, so a key it replaces cannot hide a following key name from (d)
#   ("hf_...--password=x") in one pass and show it in the next;
# - (d) before (a), so a quoted query value (?password="x") is masked before (a)
#   removes the query up to its quote and leaves the value without its key;
# - (a) before (b), so a credential (Bearer /x?q=...) cannot take the path of a
#   URL and leave its query.
# Running ``redact()`` on its own result changes nothing.
#
# Escaped line breaks. An error text can hold a JSON string or a Python repr,
# where a line break or tab is written as "\n", "\r" or "\t". Such an escape
# right before a URL path, a scheme name, a key or a key name counts as
# whitespace in rules (a) to (d), and so do escapes around the separator of
# rule (d). A token at the start of a line inside an escaped string is then found
# as in plain text ("...:\nsk-..." or '"...\npassword=x"'). So a value that
# itself starts with a backslash and n, r or t keeps those two characters.
#
# Strings. A string starts at a quote character (" or ') and ends at a later
# quote of the same kind. The backslashes before a quote are read in two ways:
# (1) the closing quote has exactly as many backslashes before it as the opening
# quote; (2) the backslashes are escape levels, as in a JSON string inside a JSON
# string: a quote with n backslashes before it is at the level given by the
# number of trailing 1 bits of n (0 for ", 1 for \", 2 for \\\"), and the
# first quote whose level is at most the opening quote's closes the string. When
# the two readings end a string, or a bracketed value that holds strings, at
# different places, the end is not known and the value runs to the end of the
# text. A triple-quoted string ends at three closing quotes in a row. Inside the
# {...} fields of an f-string or t-string (prefix f or t) every quote starts a
# nested string (Python 3.12 allows the f-string's own quote there). A string
# that does not end runs to the end of the text.
#
# (a) URLs. An absolute URL starts with "scheme://". A relative URL is a path
#     that starts with "/" ("//authority/..." included) at the start of the text
#     or after whitespace, a quote character (' " `) or one of ( [ { < = : ,.
#     Scheme, host, port, path and fragment are kept. The userinfo (the authority
#     up to its last "@") becomes "[userinfo removed]", and a non-empty query
#     becomes "[query removed]". The authority ends at "/", "?", "#",
#     whitespace, a quote character, "<" or ">". The path ends at "?", "#",
#     whitespace, a quote character, "<" or ">". The query ends at "#",
#     whitespace, a quote character, "<", ">" or the end of the text, so ")",
#     "]", "," and "." belong to it. Backslashes that escape the quote ending a
#     query stay. Rule (d) masks the secret-named parameters of a fragment
#     ("#access_token=...").
# (b) Authorization credentials. "Bearer" (any case), "Basic", "Token" or
#     "Digest", not preceded by a letter, digit, "_" or "-", then spaces or tabs
#     and a credential. The credential becomes "***redacted***". A credential is
#     a comma-separated list of name=value parameters (the value a token or a
#     double-quoted string) or, except after "Digest", a token68
#     ([A-Za-z0-9._~+/-] characters, then any "="), as in RFC 9110 section 11. A
#     key that rule (c) replaced counts as token68 characters.
# (c) Known key formats, not preceded by a letter or digit: "sk-" (OpenAI,
#     "sk-proj-" included) and "AIza" (Google) followed by 20 or more of
#     [A-Za-z0-9_-]; "ghp_", "gho_", "ghs_" (GitHub) and "hf_" (Hugging Face)
#     followed by 30 or more letters and digits; "github_pat_" followed by 30 or
#     more of [A-Za-z0-9_]; "xoxa-", "xoxb-", "xoxp-" and "xoxr-" (Slack)
#     followed by 10 or more of [A-Za-z0-9-]; "AKIA" or "ASIA" followed by
#     exactly 16 upper-case letters and digits (AWS). The whole key becomes
#     "***redacted***".
# (d) Values of secret-named keys. A key is one of the names in ``SECRET_KEYS``
#     (section 6.3), compared exactly and case-insensitively, as a whole name:
#     no letter, digit, "_" or "-" right before or after it, apart from the
#     leading "-" of a command-line flag ("--password"). The key may be quoted
#     (" or ', with the same backslashes before both quotes), and a quoted key
#     may be the index of a subscript (config["password"] = ...). It is followed
#     by ":" or "=" (":=" and "=>" too; "::" and "==" are not separators), with
#     any whitespace, line breaks included, on both sides. Its value is replaced
#     by "***masked***":
#     - a string, with a prefix of up to two of r, b, u, f, t in either case:
#       its content, f-strings included;
#     - a value that starts with "[", "{" or "(": the whole value up to the
#       matching bracket, across lines; strings inside are skipped, and a closing
#       bracket of another kind is ignored;
#     - "Bearer", "Basic", "Token" or "Digest" (any case) and a credential as in
#       (b): the credential;
#     - after a lone ":", a Python annotation and then " =" (api_key: str | None
#       = None): the annotation is kept, and the value after the "=" is read by
#       these rules. An annotation is one or more dotted names joined by "|",
#       each with at most one [...] that holds no bracket, quote, "=" or line
#       break;
#     - any other value (a bare token): up to whitespace or one of , ; & ) ] }.
#       After a URL or form parameter (a key right after "?", "&" or "#") the
#       bare token runs up to whitespace, a quote character, "<", ">", "&" or
#       "#" instead, and backslashes that escape the quote ending it stay.
#     A string or bracket that does not end is masked to the end of the text. A
#     key of a known format that rule (c) replaced in the value is masked with it,
#     so the value of a secret key is always "***masked***".
#     Kept as they are: empty strings and brackets; None, null, nil, undefined,
#     true and false (any case); "***masked***"; and reference expressions. A
#     reference expression is a dotted name followed by "(" or "[" (get_key(),
#     os.environ["API_KEY"]), or a dotted name with at least one "." that is the
#     whole bare token (args.api_key). A value is not a reference expression
#     after a URL or form parameter, when it starts with "eyJ" (a JWT), or when
#     its bare token ends at "&".
#
# ``redacted`` is true only when a rule replaced something in the stored part of
# the text; cutting the text is not redaction. The first ``REDACT_WINDOW``
# characters are redacted as a whole, and the result is cut to the stored length
# afterwards, so a secret that crosses the cut is replaced whole. When the text
# is longer than the window, the last whitespace-separated token of the window
# is dropped after redaction (the window may have cut a secret in two), and with
# no limit the stored text ends with a note and ``cut`` is true. Each pass reads
# every character a bounded number of times, so the work is linear in the length
# of the text.

REDACTED = "***redacted***"
QUERY_REMOVED = "?[query removed]"
USERINFO_REMOVED = "[userinfo removed]@"
REDACT_WINDOW = 64 * 1024
REDACTION_FAILED = "[recorder: text not stored (redaction failed)]"
_QUERY_MARK = QUERY_REMOVED[1:]
_USERINFO_MARK = USERINFO_REMOVED[:-1]
_QUOTES = "'\"`"
# right after an escaped line break or tab ("Escaped line breaks" above): an
# alternative to the lookbehinds that make a token start a whole word (lower-case
# n, r, t also in the patterns compiled with IGNORECASE)
_AFTER_ESCAPE = r"(?<=\\(?-i:[nrt]))"


class Redacted(NamedTuple):
    """A redacted text: ``redacted`` when a rule replaced something in the stored
    text, ``cut`` when only the first ``REDACT_WINDOW`` characters were stored."""

    text: str
    redacted: bool
    cut: bool


def _is_mask(text: str, i: int) -> bool:
    """A mask that a rule (or an earlier redaction) put at ``i``."""
    return text.startswith(MASK_VALUE, i) or text.startswith(REDACTED, i)


def _backslashes(text: str, j: int, lo: int) -> int:
    """The number of backslashes right before ``j``, reading back no further than ``lo``."""
    i = j
    while i > lo and text[i - 1] == "\\":
        i -= 1
    return j - i


# strings ----------------------------------------------------------------------

_PREFIX_CHARS = frozenset("rRbBuUfFtT")
_FORMAT_CHARS = frozenset("fFtT")
_NAME_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")
_FORMAT_STOP = {"'": re.compile(r"[{}']"), '"': re.compile(r'[{}"]')}
_FIELD_STOP = re.compile(r"[{}'\"]")


def _trailing_ones(n: int) -> int:
    """Escape level of a quote with ``n`` backslashes before it: the number of
    trailing 1 bits of ``n`` (0 for ``"``, 1 for ``\\"``, 2 for ``\\\\\\"``; an
    even count is escaped backslashes before a quote of level 0)."""
    return (n ^ (n + 1)).bit_length() - 1


def _escape_len(k: int, levels: bool) -> int:
    """The backslashes that belong to a quote with ``k`` backslashes before it: all
    of them when counts are compared, ``2**level - 1`` when they are levels."""
    return (1 << _trailing_ones(k)) - 1 if levels else k


def _is_format_string(text: str, i: int, lo: int) -> bool:
    """Whether the string prefix that ends at ``i`` (up to two of r, b, u, f, t,
    not part of a longer name, not reaching back before ``lo``) holds f or t."""
    p = i
    while p > lo and i - p < 2 and text[p - 1] in _PREFIX_CHARS:
        p -= 1
    if p > lo and text[p - 1] in _NAME_CHARS:
        return False
    return any(c in _FORMAT_CHARS for c in text[p:i])


def _triple_at(text: str, j: int, quote: str, e: int) -> bool:
    """Whether the quote at ``j`` is followed by two more quotes of its kind with
    ``e`` backslashes before each: three quotes in a row at one escape level."""
    escape = "\\" * e
    i = j + 1
    for _ in range(2):
        if not text.startswith(escape, i) or not text.startswith(quote, i + e):
            return False
        i += e + 1
    return True


def _string_end(text: str, q: int, k: int, fmt: bool, levels: bool) -> tuple[int, bool, int, int]:
    """``(end, closed, body, content_end)`` of the string whose opening quote is at
    ``q``, with ``k`` backslashes before it (``fmt``: an f- or t-string), in one of
    the two readings of backslashes (``levels``; see the comment block above).

    ``body`` is the index where the content starts, ``end`` the index after the
    closing quote and ``content_end`` the index of the backslashes that belong to
    it. The strings nested in the {...} fields of f-strings are tracked on an
    explicit stack (no recursion). A string that does not end runs to the end of
    the text (``closed`` is False).
    """
    n = len(text)
    quote = text[q]
    e = _escape_len(k, levels)
    triple = _triple_at(text, q, quote, e)
    pos = body = q + (3 + 2 * e if triple else 1)
    # one frame per string being read: [quote, k, triple, fmt, depth of {...} fields]
    stack: list[list[Any]] = [[quote, k, triple, fmt, 0]]
    while True:
        frame = stack[-1]
        quote, k, triple, fmt, depth = frame
        if fmt:
            m = (_FIELD_STOP if depth else _FORMAT_STOP[quote]).search(text, pos)
            j = m.start() if m is not None else -1
        else:
            j = text.find(quote, pos)
        if j < 0:
            return n, False, body, n
        c = text[j]
        if c == "{":
            if depth == 0 and text.startswith("{", j + 1):
                pos = j + 2  # "{{" is a literal brace
            else:
                frame[4] = depth + 1
                pos = j + 1
            continue
        if c == "}":
            if depth == 0:
                pos = j + (2 if text.startswith("}", j + 1) else 1)
            else:
                frame[4] = depth - 1
                pos = j + 1
            continue
        kj = _backslashes(text, j, pos)
        if depth == 0:
            closes = c == quote and (_trailing_ones(kj) <= _trailing_ones(k) if levels else kj == k)
            e = _escape_len(kj, levels)
            if closes and (not triple or _triple_at(text, j, quote, e)):
                end = j + (3 + 2 * e if triple else 1)
                stack.pop()
                if not stack:
                    return end, True, body, j - e
                pos = end
            else:
                pos = j + 1
            continue
        # inside a {...} field every quote starts a nested string, with its own backslashes:
        # JSON escapes " and not ', so one level of encoding gives them different counts
        e = _escape_len(kj, levels)
        nested_triple = _triple_at(text, j, c, e)
        stack.append([c, kj, nested_triple, _is_format_string(text, j - kj, pos), 0])
        pos = j + (3 + 2 * e if nested_triple else 1)


def _string_span(text: str, q: int, k: int, fmt: bool) -> tuple[int, bool, int, int]:
    """``_string_end`` when both readings of backslashes end the string at the same
    place; otherwise the string runs to the end of the text. Where the readings
    differ, the end cannot be known, and a later key could hide in between."""
    exact = _string_end(text, q, k, fmt, False)
    by_level = _string_end(text, q, k, fmt, True) if exact[1] else exact
    if exact[:3] == by_level[:3] and exact[1]:
        return exact
    n = len(text)
    return n, False, min(exact[2], by_level[2]), n


_BRACKET_STOP = re.compile(r"[\[\]{}()'\"]")
_CLOSING = {"[": "]", "{": "}", "(": ")"}


def _bracket_end(text: str, start: int, levels: bool) -> tuple[int, bool]:
    """``(end, closed)`` of the bracket at ``start`` in one reading of backslashes:
    the index after its matching bracket. Strings are skipped, and a closing bracket
    of another kind is ignored. An unclosed bracket, or a string in it that does not
    end, runs to the end of the text."""
    expected: list[str] = []
    pos = start
    while True:
        m = _BRACKET_STOP.search(text, pos)
        if m is None:
            return len(text), False
        j = m.start()
        c = text[j]
        if c == '"' or c == "'":
            k = _backslashes(text, j, pos)
            end, closed, _, _ = _string_end(text, j, k, _is_format_string(text, j - k, pos), levels)
            if not closed:
                return len(text), False
            pos = end
            continue
        if c in _CLOSING:
            expected.append(_CLOSING[c])
        elif expected and c == expected[-1]:
            expected.pop()
            if not expected:
                return j + 1, True
        pos = j + 1


def _bracket_span(text: str, start: int) -> tuple[int, bool]:
    """``_bracket_end`` when both readings of backslashes close the bracket at the
    same place; otherwise the value runs to the end of the text. A string misread
    in one reading shifts which quotes open strings, so a bracket inside a string
    can close the value early, or late in the middle of the next key's value."""
    exact = _bracket_end(text, start, False)
    if exact[1] and _bracket_end(text, start, True) == exact:
        return exact
    return len(text), False


# rule (a) -------------------------------------------------------------------

# a scheme is tried once per run of scheme characters; a relative URL starts at a "/"
# that begins the text or follows whitespace (or its escape), a quote or one of ( [ { < = : ,
_URL_START = re.compile(
    r"(?<![A-Za-z0-9+.\-])[A-Za-z][A-Za-z0-9+.\-]{0,31}://"
    r"|(?:(?<![^\s'\"`(\[{<=:,])|" + _AFTER_ESCAPE + r")/"
)
# the authority; a userinfo that an earlier redaction removed stays one unit
_AUTHORITY = re.compile(r"(?:\[userinfo removed\]|[^\s/?#'\"`<>])*")
_PATH = re.compile(r"[^\s?#'\"`<>]*")
_QUERY = re.compile(r"[^\s#'\"`<>]*")


def _redact_urls(text: str) -> tuple[str, bool]:
    """Rule (a): userinfo and query of every URL."""
    out: list[str] = []
    copied = pos = 0
    n = len(text)
    while True:
        m = _URL_START.search(text, pos)
        if m is None:
            break
        start = m.start()
        relative = text[start] == "/"
        if not relative:
            authority = m.end()
        else:
            authority = start + 2 if text.startswith("//", start) else -1
        path_start = _AUTHORITY.match(text, authority).end() if authority >= 0 else start
        end = _PATH.match(text, path_start).end()
        has_query = end < n and text[end] == "?"
        if relative and authority < 0 and not has_query:
            pos = max(end, start + 1)  # a later "/" of this path starts the same path
            continue
        if authority >= 0:
            at = text.rfind("@", authority, path_start)
            earlier = at == authority + len(_USERINFO_MARK) and text.startswith(_USERINFO_MARK, authority)
            if at > authority and not earlier:
                out.append(text[copied:authority])
                out.append(_USERINFO_MARK)
                copied = at
        if has_query:
            q = end + 1
            end = _QUERY.match(text, q).end()
            keep = end
            if end < n and text[end] in _QUOTES:
                while keep > q and text[keep - 1] == "\\":  # the escape of the closing quote
                    keep -= 1
            if keep > q and not text.startswith(_QUERY_MARK, q):
                out.append(text[copied:q])
                out.append(_QUERY_MARK)
                copied = keep
        pos = max(end, start + 1)
    if not out:
        return text, False
    out.append(text[copied:])
    return "".join(out), True


# rule (b) -------------------------------------------------------------------

_TOKEN_CHARS = r"[!#$%&'*+.^_`|~0-9A-Za-z\-]"  # RFC 9110 tchar
_AUTH_PARAM = re.compile(_TOKEN_CHARS + r"+[ \t]*=[ \t]*")
_AUTH_TOKEN = re.compile(_TOKEN_CHARS + r"+")
_LIST_SEP = re.compile(r"[ \t]*,[ \t]*")
# a key that rule (c) replaced stays part of the credential after it
_TOKEN68 = re.compile(r"(?:\*\*\*redacted\*\*\*|[A-Za-z0-9\-._~+/])+=*")
_SCHEME = re.compile(
    r"(?:(?<![A-Za-z0-9_\-])|" + _AFTER_ESCAPE + r")((?i:bearer)|Basic|Token|Digest)[ \t]+"
)
_SCHEME_ANY_CASE = re.compile(r"(bearer|basic|token|digest)[ \t]+", re.IGNORECASE)


def _quoted_at(text: str, i: int) -> int | None:
    """End of a double-quoted string (after any backslashes) that starts at ``i``, or None."""
    k = i
    n = len(text)
    while k < n and text[k] == "\\":
        k += 1
    if k < n and text[k] == '"':
        return _string_span(text, k, k - i, False)[0]
    return None


def _credential_end(text: str, i: int, params_only: bool) -> int | None:
    """End of the credential that starts at ``i`` (rule (b)), or None."""
    end = None
    k = i
    while True:
        param = _AUTH_PARAM.match(text, k)
        if param is None:
            break
        value = _quoted_at(text, param.end())
        if value is None:
            token = _AUTH_TOKEN.match(text, param.end())
            if token is None:
                break
            value = token.end()
        end = value
        sep = _LIST_SEP.match(text, value)
        if sep is None:
            break
        k = sep.end()
    if end is not None or params_only:
        return end
    token68 = _TOKEN68.match(text, i)
    return token68.end() if token68 else None


def _redact_credentials(text: str) -> tuple[str, bool]:
    """Rule (b)."""
    out: list[str] = []
    copied = pos = 0
    while True:
        m = _SCHEME.search(text, pos)
        if m is None:
            break
        pos = i = m.end()
        if _is_mask(text, i):
            continue
        end = _credential_end(text, i, m.group(1) == "Digest")
        if end is None:
            continue
        out.append(text[copied:i])
        out.append(REDACTED)
        copied = pos = end
    if not out:
        return text, False
    out.append(text[copied:])
    return "".join(out), True


# rule (c) -------------------------------------------------------------------

_KEY_FORMAT = re.compile(
    r"(?:(?<![A-Za-z0-9])|" + _AFTER_ESCAPE + r")(?:"
    r"sk-[A-Za-z0-9_\-]{20,}"  # OpenAI (sk-proj- included)
    r"|AIza[A-Za-z0-9_\-]{20,}"  # Google
    r"|gh[pos]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"  # GitHub
    r"|hf_[A-Za-z0-9]{30,}"  # Hugging Face
    r"|xox[abpr]-[A-Za-z0-9\-]{10,}"  # Slack
    r"|(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Za-z0-9])"  # AWS access key id
    r"|gsk_[A-Za-z0-9_\-]{20,}|xai-[A-Za-z0-9_\-]{20,}|r8_[A-Za-z0-9_\-]{20,}"
    r"|-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|\Z)"
    r")"
)


def _redact_key_formats(text: str) -> tuple[str, bool]:
    """Rule (c)."""
    out, count = _KEY_FORMAT.subn(REDACTED, text)
    return out, count > 0


# rule (d) -------------------------------------------------------------------

_SECRET_KEY = re.compile(
    r"(?:(?<![A-Za-z0-9_\-])|" + _AFTER_ESCAPE + r")-*("
    + "|".join(sorted((re.escape(k) for k in SECRET_KEYS), key=len, reverse=True))
    + r")(?![A-Za-z0-9_\-])",
    re.IGNORECASE,
)
# whitespace around a separator, escaped line breaks and tabs included (with the
# backslashes of every level of a JSON string inside a JSON string)
_SPACE = re.compile(r"(?:\s|\\+[nrt])*")
_LITERAL_START = re.compile(r"([rRuUbBfFtT]{0,2})(\\*)([\"'])")
# a bare token; the marks of rule (a) hold a space and stay whole
_BARE = re.compile(r"(?:\[query removed\]|\[userinfo removed\]|[^\s,;&)\]}])+")
_FORM_BARE = re.compile(r"[^\s'\"`<>&#]+")
_DOTTED = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_KEPT_WORDS = frozenset({"none", "null", "nil", "undefined", "true", "false"})
_BARE_END = frozenset(",;&)]}")
_FORM_END = frozenset("'\"`<>&#")
_ANNOTATION_ITEM = r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*(?:\[[^\[\]\n'\"=]*\])?"
_ANNOTATION = re.compile(_ANNOTATION_ITEM + r"(?:[ \t]*\|[ \t]*" + _ANNOTATION_ITEM + r")*[ \t]+=(?![=>])")


def _kept_name(text: str, v: int, form: bool) -> bool:
    """A kept word or a reference expression (rule (d)) at ``v``. Only the dotted
    name at ``v`` and the character after it are read, so a kept value costs the
    length of that name."""
    m = _DOTTED.match(text, v)
    if m is None:
        return False
    name, e = m.group(0), m.end()
    after = text[e] if e < len(text) else ""
    ends = not after or after.isspace() or after in (_FORM_END if form else _BARE_END)
    if ends and name.lower() in _KEPT_WORDS:
        return True
    if form or name.startswith("eyJ"):
        return False  # a URL or form parameter, or a JWT: data, not code
    if after == "(" or after == "[":
        return True  # a call or a subscript
    return ends and after != "&" and "." in name  # a dotted name that is the whole value


def _value_span(text: str, v: int, form: bool, annotation: bool) -> tuple[int, int, str] | None:
    """``(start, end, replacement)`` for the value of a secret key that starts at
    ``v``, or None when it is kept (rule (d)). ``form``: the key is a URL or form
    parameter; ``annotation``: the separator was a lone ":"."""
    n = len(text)
    if v >= n or text.startswith(MASK_VALUE, v):
        return None
    literal = _LITERAL_START.match(text, v)
    if literal is not None:
        k = len(literal.group(2))
        q = literal.start(3)
        fmt = any(c in _FORMAT_CHARS for c in literal.group(1))
        _end, _closed, body, content_end = _string_span(text, q, k, fmt)
        content = text[body:content_end]
        if not content or content == MASK_VALUE:
            return None
        return body, content_end, MASK_VALUE
    if text[v] in _CLOSING:
        end, closed = _bracket_span(text, v)
        inner = text[v + 1:end - 1] if closed else text[v + 1:end]
        return (v, end, MASK_VALUE) if inner.strip() else None
    scheme = _SCHEME_ANY_CASE.match(text, v)
    if scheme is not None:
        i = scheme.end()
        if text.startswith(MASK_VALUE, i):
            return None
        end = _credential_end(text, i, scheme.group(1).lower() == "digest")
        if end is not None:
            return i, end, MASK_VALUE
    if annotation:
        typed = _ANNOTATION.match(text, v)
        if typed is not None:
            return _value_span(text, _SPACE.match(text, typed.end()).end(), form, False)
    if _kept_name(text, v, form):
        return None
    if form:
        m = _FORM_BARE.match(text, v)
        if m is None:
            return None
        end = m.end()
        if end < n and text[end] in _QUOTES:
            while end > v and text[end - 1] == "\\":  # the escape of the closing quote
                end -= 1
        return (v, end, MASK_VALUE) if end > v else None
    bare = _BARE.match(text, v)
    return (v, bare.end(), MASK_VALUE) if bare is not None else None


def _mask_secret_values(text: str) -> tuple[str, bool]:
    """Rule (d), one pass. After a kept value the search goes on inside it, so a key
    in a kept reference expression (``get(token="x")``) is still found."""
    out: list[str] = []
    copied = pos = 0
    n = len(text)
    while True:
        hit = _SECRET_KEY.search(text, pos)
        if hit is None:
            break
        pos = hit.end()
        key_start, key_end = hit.start(), hit.end()
        quoted = False
        if key_start > copied and text[key_start - 1] in "\"'":
            quote = text[key_start - 1]
            k = _backslashes(text, key_start - 1, copied)
            c = key_end
            while c < n and text[c] == "\\":
                c += 1
            if c - key_end == k and text.startswith(quote, c):
                quoted = True
                opening = key_start - 1 - k
                key_start, key_end = opening, c + 1
                if opening > copied and text[opening - 1] == "[" and text.startswith("]", key_end):
                    key_end += 1  # the index of a subscript: config["password"] = ...
        j = _SPACE.match(text, key_end).end()
        if j >= n:
            continue
        lone_colon = False
        if text[j] == "=":
            if text.startswith("==", j):
                continue
            j += 2 if text.startswith("=>", j) else 1
        elif text[j] == ":":
            if text.startswith("::", j):
                continue
            lone_colon = not text.startswith(":=", j)
            j += 1 if lone_colon else 2
        else:
            continue
        v = _SPACE.match(text, j).end()
        form = not quoted and key_start > 0 and text[key_start - 1] in "?&#"
        found = _value_span(text, v, form, lone_colon)
        if found is None:
            continue
        start, end, replacement = found
        out.append(text[copied:start])
        out.append(replacement)
        copied = pos = end
    if not out:
        return text, False
    out.append(text[copied:])
    return "".join(out), True


def _apply_rules(text: str) -> tuple[str, bool]:
    """The rules above, in the order (c), (d), (a), (b); ``(text, replaced)``."""
    text, keys = _redact_key_formats(text)
    text, values = _mask_secret_values(text)
    text, urls = _redact_urls(text)
    text, credentials = _redact_credentials(text)
    return text, keys or values or urls or credentials


def _drop_last_token(text: str) -> str:
    """``text`` without its last whitespace-separated token."""
    cut = max(text.rfind(c) for c in " \t\n\r\f\v")
    return text[:cut + 1]


def redact(text: Any, limit: int | None = 2000) -> Redacted:
    """``text`` with secrets removed (the rules above), then cut to ``limit`` characters.

    Raises only the trial alarm (``TimeoutError``) and non-``Exception``
    exceptions: it also runs on the calling thread (``hooks/common``,
    ``bootstrap``). If a rule fails, a fixed note is stored instead of the text,
    so nothing unredacted is stored.
    """
    try:
        if not isinstance(text, str):
            text = str(text)
        longer = len(text) > REDACT_WINDOW
        window = text[:REDACT_WINDOW] if longer else text
        out, replaced = _apply_rules(window)
        if longer:
            out = _drop_last_token(out)
        stored = out if limit is None else out[:limit]
        # a replacement that starts after the cut leaves the stored text a plain prefix
        flag = replaced and stored != window[:len(stored)]
        if limit is not None:
            return Redacted(stored, flag, False)
        if longer:
            stored += f"\n[recorder: only the first {REDACT_WINDOW} of {len(text)} characters were read]"
        return Redacted(stored, flag, longer)
    except TimeoutError:  # the trial alarm (section 2)
        raise
    except Exception:  # noqa: BLE001
        return Redacted(REDACTION_FAILED, True, False)


def redact_text(text: Any, limit: int | None = 2000) -> str:
    """``redact(text, limit).text``."""
    return redact(text, limit).text


def put_redacted(target: dict[str, Any], field: str, text: Any, limit: int | None = 2000) -> Redacted:
    """Store ``text`` redacted as ``target[field]``, with ``<field>_redacted: true``
    when a rule replaced something in the stored text and ``<field>_cut: true``
    when only the first ``REDACT_WINDOW`` characters were read."""
    result = redact(text, limit)
    target[field] = result.text
    if result.redacted:
        target[field + "_redacted"] = True
    if result.cut:
        target[field + "_cut"] = True
    return result


ERROR_TEXT_KEYS = frozenset({"error", "setup_error"})
RECORD_REDACTED_KEY = "trace_redacted_fields"
RECORD_CUT_KEY = "trace_cut_fields"


def _redact_record(value: Any, key: Any, path: str, redacted: list[str], cut: list[str]) -> Any:
    if isinstance(value, dict):
        return {k: _redact_record(v, k, join_path(path, k), redacted, cut) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_record(v, key, join_path(path, i), redacted, cut) for i, v in enumerate(value)]
    if isinstance(value, str) and key in ERROR_TEXT_KEYS:
        result = redact(value, None)
        if result.redacted:
            redacted.append(path)
        if result.cut:
            cut.append(path)
        return result.text
    return value


def redact_record(value: Any) -> Any:
    """A runner record (H16) with its error texts redacted by ``redact``.

    The experiment runner stores tracebacks of failed attempts
    (``attempts[].error``, ``error``, ``setup_error``); a request exception in
    them can carry a URL with a key. Every other field is copied as it is. The
    paths of the texts that changed are listed in ``trace_redacted_fields``, and
    of texts longer than ``REDACT_WINDOW`` in ``trace_cut_fields``.
    """
    redacted: list[str] = []
    cut: list[str] = []
    out = _redact_record(value, None, "", redacted, cut)
    if isinstance(out, dict):
        if redacted:
            out[RECORD_REDACTED_KEY] = redacted
        if cut:
            out[RECORD_CUT_KEY] = cut
    return out


__all__ = [
    "redact",
    "put_redacted",
    "Redacted",
    "REDACT_WINDOW",
    "redact_record",
    "sanitize_url",
    "strip_query",
    "body_to_bytes",
    "normalize_request",
    "fill_placeholder",
    "reconstruct",
    "mask_secrets",
    "is_secret_key",
    "has_nonfinite",
    "image_dimensions",
    "pixel_sha256",
    "raw_pixel_sha256",
    "parse_response_body",
    "redact_text",
    "ext_for_mime",
    "join_path",
    "Media",
    "ExternalUrl",
]
