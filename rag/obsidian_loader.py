"""Obsidian 笔记专用加载与清洗。

针对 worknote 这类 Obsidian 仓库做格式规范化，提升后续切分与检索质量：
- 去除 frontmatter（YAML 元数据块）
- 去除图片/附件嵌入  ![[xxx.png]]  ![](path)
- 解析 wiki 双链 [[note]] -> note（保留锚文本，丢弃纯图片链接）
- 折叠多余空白
- 保护代码块：切分时不打断 ``` 包裹的代码
- 抽取元数据：title、tags、双链目标（用于后续图增强检索）

清洗后输出结构化文档，供 chunking 使用。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 匹配 frontmatter（文件开头的 --- ... ---）
_FRONTMATTER = re.compile(r"\A\s*---\s*\n.*?\n---\s*\n", re.S)
# 图片嵌入：![[...]] 或 ![](...)
_IMG_EMBED = re.compile(r"!\[\[[^\]]*\]\]")
_IMG_MD = re.compile(r"!\[[^\]]*\]\([^)]*\)")
# wiki 双链 [[target]] 或 [[target|alias]] 或 [[target#heading]]
_WIKI_LINK = re.compile(r"\[\[([^\]]+)\]\]")
# 各种图片/附件后缀
_ATTACH_SUFFIX = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp",
                  ".pdf", ".canvas", ".excalidraw", ".mp4", ".mov"}
# 连续 3+ 空行折叠
_BLANK = re.compile(r"\n{3,}")
# 行尾空白
_TRAIL = re.compile(r"[ \t]+\n")
# 标题行
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.M)


def _normalize_wiki_link(match: re.Match) -> str:
    """[[target|alias]] / [[target#head]] / [[target]] -> 显示文本。

    纯附件链接（图片等）直接丢弃，返回空串。
    """
    inner = match.group(1)
    # 带 alias：[[target|alias]] -> alias
    if "|" in inner:
        target, alias = inner.split("|", 1)
        target, alias = target.strip(), alias.strip()
        if Path(target).suffix.lower() in _ATTACH_SUFFIX:
            return ""
        return alias
    # 带 heading：[[target#head]] -> head（或 target）
    if "#" in inner:
        target, head = inner.split("#", 1)
        target, head = target.strip(), head.strip()
        if Path(target).suffix.lower() in _ATTACH_SUFFIX:
            return ""
        return head or target
    # 普通链接
    target = inner.strip()
    if Path(target).suffix.lower() in _ATTACH_SUFFIX:
        return ""
    return target  # 保留笔记名作为可检索文本


def clean_obsidian(text: str) -> tuple[str, dict[str, Any]]:
    """清洗 Obsidian 语法，返回 (clean_text, meta)。

    meta 包含：tags、wiki_targets（笔记间双链目标，用于图增强）。
    """
    meta: dict[str, Any] = {"tags": [], "wiki_targets": []}

    # 1. 抽取 frontmatter 中的 tags（先于删除）
    fm = _FRONTMATTER.match(text)
    if fm:
        for line in fm.group(0).splitlines():
            line = line.strip()
            if line.lower().startswith("tags:"):
                # tags: [a, b] 或 tags: a, b 或 tags:\n - a
                vals = line.split(":", 1)[1].strip()
                # 去掉行内数组方括号
                vals = vals.strip("[]")
                meta["tags"].extend(
                    t.strip().lstrip("#") for t in re.split(r"[,\s]+", vals)
                    if t.strip().lstrip("#")
                )
            elif line.startswith("- ") or line.startswith("#"):
                # YAML 列表式 tag：- #tag
                vals = line.lstrip("- ").lstrip("#")
                meta["tags"].extend(
                    t.strip().lstrip("#") for t in re.split(r"[,\s]+", vals)
                    if t.strip().lstrip("#")
                )

    # 2. 删 frontmatter
    text = _FRONTMATTER.sub("", text, count=1)

    # 3. 抽取并归一化 wiki 双链（先记录，再替换）
    for m in _WIKI_LINK.finditer(text):
        inner = m.group(1)
        # 带 alias/heading 时取 target 部分
        tgt = re.split(r"[|#]", inner)[0].strip()
        if tgt and Path(tgt).suffix.lower() not in _ATTACH_SUFFIX:
            meta["wiki_targets"].append(tgt)

    # 4. 删图片嵌入
    text = _IMG_EMBED.sub("", text)
    text = _IMG_MD.sub("", text)
    # 5. 归一化 wiki 双链为纯文本
    text = _WIKI_LINK.sub(_normalize_wiki_link, text)
    # 6. 折叠空白
    text = _TRAIL.sub("\n", text)
    text = _BLANK.sub("\n\n", text)
    return text.strip(), meta


def extract_headings(text: str) -> list[tuple[int, str]]:
    """抽取标题层级，用于结构化切分。返回 [(level, title), ...]。"""
    return [(len(m.group(1)), m.group(2).strip())
            for m in _HEADING.finditer(text)]


@dataclass
class NoteDoc:
    """一篇清洗后的笔记。"""
    source: str                # 相对路径，如 2026/0105_数据导出步骤.md
    title: str                 # 笔记标题（取文件名去后缀）
    text: str                  # 清洗后正文
    tags: list[str] = field(default_factory=list)
    wiki_targets: list[str] = field(default_factory=list)
    headings: list[tuple[int, str]] = field(default_factory=list)


def load_obsidian_notes(root: str | Path,
                        skip_dirs: set[str] | None = None) -> list[NoteDoc]:
    """递归加载 Obsidian 仓库下的 .md 笔记并清洗。

    自动跳过 .git / .obsidian / _pic 等非笔记目录。
    """
    root = Path(root)
    skip = skip_dirs or {".git", ".obsidian", "_pic", "2025_pic", "2026_pic",
                         "Excalidraw", "node_modules"}
    docs: list[NoteDoc] = []
    for f in sorted(root.rglob("*.md")):
        if any(part in skip for part in f.parts):
            continue
        try:
            raw = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if not raw.strip():
            continue
        text, meta = clean_obsidian(raw)
        if not text.strip():
            continue
        rel = str(f.relative_to(root))
        title = f.stem  # 文件名去后缀作为标题
        docs.append(NoteDoc(
            source=rel, title=title, text=text,
            tags=meta["tags"], wiki_targets=meta["wiki_targets"],
            headings=extract_headings(text),
        ))
    return docs
