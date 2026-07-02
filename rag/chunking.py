"""文档切分：递归字符切分。

按 ["\n\n", "\n", "。", " ", ""] 分隔符层级递归切分，尽量保持语义完整，
再按 chunk_size/overlap 合并。返回 (text, meta) 列表。
"""
from __future__ import annotations

from typing import Any

_SEPARATORS = ["\n\n", "\n", "。", "!", "?", " ", ""]


def _split_text(text: str, size: int) -> list[str]:
    """递归切分：找到能在 size 内产生多段的最优分隔符。"""
    if len(text) <= size:
        return [text] if text.strip() else []
    for sep in _SEPARATORS:
        if not sep:
            continue
        if sep in text:
            parts = text.split(sep)
            chunks: list[str] = []
            buf = ""
            for p in parts:
                cand = (buf + sep + p) if buf else p
                if len(cand) > size and buf:
                    chunks.append(buf)
                    buf = p
                else:
                    buf = cand
            if buf:
                chunks.append(buf)
            # 对仍然过长的段继续递归
            final: list[str] = []
            for c in chunks:
                if len(c) > size:
                    final.extend(_split_text(c, size))
                elif c.strip():
                    final.append(c)
            return final
    # 无分隔符命中：硬切
    return [text[i:i + size] for i in range(0, len(text), size) if text[i:i + size].strip()]


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """递归字符切分 + overlap 重叠。"""
    base = _split_text(text, chunk_size)
    if len(base) <= 1 or overlap <= 0:
        return base
    out: list[str] = []
    for i, c in enumerate(base):
        out.append(c)
        if i < len(base) - 1:
            tail = c[-overlap:]
            out[-1] = out[-1]  # 当前段保留
            base[i + 1] = tail + base[i + 1]  # 给下一段加 overlap 头
    return out


def chunk_document(text: str, source: str, chunk_size: int = 512,
                   overlap: int = 64) -> list[dict[str, Any]]:
    """切分单个文档，附带 metadata。"""
    chunks = chunk_text(text, chunk_size, overlap)
    return [
        {"id": f"{source}#{i}", "text": t, "source": source, "index": i}
        for i, t in enumerate(chunks)
    ]


def chunk_documents(docs: list[dict], chunk_size: int = 512,
                    overlap: int = 64) -> list[dict[str, Any]]:
    """docs: [{"text":..., "source":...}, ...]"""
    out: list[dict[str, Any]] = []
    for d in docs:
        out.extend(chunk_document(d["text"], d.get("source", "doc"), chunk_size, overlap))
    return out
