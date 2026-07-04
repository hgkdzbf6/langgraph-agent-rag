"""中文友好的结构化切分。

策略（优先级从高到低）：
1. 按标题（# / ## / ###）切分成语义段——天然保持主题完整；
2. 段落仍过长时，按分隔符层级递归切分，分隔符层级针对中文优化：
   [段落 \\n\\n, 换行 \\n, 句号 。/！/？, 分号 ；, 逗号 ，, 空格, 空串]；
3. 代码块保护：``` 包裹的整段不切割，作为一个 chunk；
4. 切片间加 overlap 保留上下文连续性。

每个 chunk 带 metadata：source、title、heading_path（所属标题链）、index。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .obsidian_loader import NoteDoc

# 中文优化的分隔符层级（从粗到细）
_CN_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
# 标题行
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.M)
# 代码块围栏
_CODE_FENCE = re.compile(r"^```.*?$", re.M | re.S)


@dataclass
class CnChunk:
    text: str
    meta: dict[str, Any]


def _split_by_headings(text: str) -> list[tuple[list[str], str]]:
    """按标题切分，返回 [(heading_path, section_text), ...]。

    heading_path 是从根到当前标题的层级链，如 ["ATE流程", "数据导出"]。
    """
    # 找出所有标题位置
    marks = list(_HEADING.finditer(text))
    if not marks:
        return [[[], text]]

    sections: list[tuple[list[str], str]] = []
    # 标题前的内容（若有）
    if marks[0].start() > 0:
        pre = text[:marks[0].start()].strip()
        if pre:
            sections.append(([], pre))

    # 栈维护标题层级
    stack: list[tuple[int, str]] = []
    for i, m in enumerate(marks):
        level = len(m.group(1))
        title = m.group(2).strip()
        # 弹出更深层级
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        heading_path = [t for _, t in stack]

        start = m.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((heading_path, body))
    return sections


def _protect_code_blocks(text: str) -> tuple[str, dict[str, str]]:
    """把代码块替换为占位符，避免被切碎。返回 (masked_text, placeholders)。"""
    placeholders: dict[str, str] = {}

    def _replace(m: re.Match) -> str:
        key = f"__CODE_BLOCK_{len(placeholders)}__"
        placeholders[key] = m.group(0)
        return key

    # 匹配整段代码块（```...```）
    masked = re.sub(r"```.*?```", _replace, text, flags=re.S)
    return masked, placeholders


def _restore(text: str, placeholders: dict[str, str]) -> str:
    for key, code in placeholders.items():
        text = text.replace(key, code)
    return text


def _recursive_split(text: str, size: int, seps: list[str]) -> list[str]:
    """按分隔符层级递归切分，目标每段 <= size 字符。"""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    for sep in seps:
        if not sep:
            break
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
            final: list[str] = []
            for c in chunks:
                if len(c) > size:
                    final.extend(_recursive_split(c, size, seps))
                elif c.strip():
                    final.append(c)
            return final
    # 无分隔符：硬切
    return [text[i:i + size] for i in range(0, len(text), size) if text[i:i + size].strip()]


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    """给每个 chunk 前面拼上前一个 chunk 的尾部 overlap 字符。"""
    if len(chunks) <= 1 or overlap <= 0:
        return chunks
    out = [chunks[0]]
    for i in range(1, len(chunks)):
        tail = chunks[i - 1][-overlap:]
        out.append(tail + chunks[i])
    return out


def chunk_note(note: NoteDoc, chunk_size: int = 512,
               overlap: int = 64) -> list[CnChunk]:
    """切分单篇笔记。

    流程：保护代码块 → 按标题分段 → 每段递归切分（中文分隔符）→ overlap → 还原代码块。
    """
    text = note.text
    masked, placeholders = _protect_code_blocks(text)
    sections = _split_by_headings(masked)

    chunks: list[CnChunk] = []
    for heading_path, section in sections:
        if not section.strip():
            continue
        # 代码块占位符单独成块（保持完整）
        if section.strip().startswith("__CODE_BLOCK_"):
            restored = _restore(section, placeholders)
            chunks.append(CnChunk(
                text=restored,
                meta=_meta(note, heading_path, len(chunks)),
            ))
            continue
        pieces = _recursive_split(section, chunk_size, _CN_SEPARATORS)
        pieces = _add_overlap(pieces, overlap)
        for p in pieces:
            if not p.strip():
                continue
            restored = _restore(p, placeholders)
            chunks.append(CnChunk(
                text=restored,
                meta=_meta(note, heading_path, len(chunks)),
            ))
    # 整篇无标题/无切分兜底
    if not chunks:
        chunks.append(CnChunk(text=_restore(masked, placeholders),
                              meta=_meta(note, [], 0)))
    return chunks


def _meta(note: NoteDoc, heading_path: list[str], index: int) -> dict[str, Any]:
    return {
        "source": note.source,
        "title": note.title,
        "heading_path": " > ".join(heading_path) if heading_path else note.title,
        "index": index,
        "tags": note.tags,
    }


def chunk_notes(notes: list[NoteDoc], chunk_size: int = 512,
                overlap: int = 64) -> list[dict[str, Any]]:
    """切分多篇笔记，返回 pipeline 可消费的 [{text, source, ...}] 列表。"""
    out: list[dict[str, Any]] = []
    for note in notes:
        for c in chunk_note(note, chunk_size, overlap):
            d = {"text": c.text}
            d.update(c.meta)
            out.append(d)
    return out
