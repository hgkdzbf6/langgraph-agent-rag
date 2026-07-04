"""Obsidian 清洗 + 中文切分测试（免 Key 免网络）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.obsidian_loader import clean_obsidian
from rag.cn_chunking import chunk_note, chunk_notes
from rag.obsidian_loader import NoteDoc


SAMPLE = """---
tags: [协议, ATE]
---

# 新版协议设计

报文头部 2byte，报文校验 1byte。

![[Pasted image 20260105144843.png]]

## parse定义

解析和处理分开。处理字符和处理业务逻辑应该分开。

参考 [[0905讨论]] 这篇笔记。

```python
def parse(data):
    return data[:2]
```
"""


def test_clean_removes_frontmatter_and_images():
    text, meta = clean_obsidian(SAMPLE)
    assert "---" not in text[:10]          # frontmatter 已删
    assert "![[Pasted" not in text          # 图片嵌入已删
    assert "tags" not in text               # frontmatter 内容已删
    assert "报文头部" in text                # 正文保留


def test_clean_preserves_wiki_link_text():
    text, _ = clean_obsidian(SAMPLE)
    assert "0905讨论" in text                # 双链保留为纯文本
    assert "[[" not in text                  # 语法标记已去


def test_clean_extracts_tags_and_targets():
    _, meta = clean_obsidian(SAMPLE)
    assert "协议" in meta["tags"]
    assert "0905讨论" in meta["wiki_targets"]


def test_clean_drops_image_wiki_links():
    text, meta = clean_obsidian("[[xxx.png]] 和 [[笔记A]]")
    assert "xxx.png" not in text
    assert "笔记A" in text
    assert "笔记A" in meta["wiki_targets"]
    assert all(".png" not in t for t in meta["wiki_targets"])


def test_chunk_by_headings():
    note = NoteDoc(source="t.md", title="t", text=SAMPLE.split("---\n", 2)[-1].strip())
    chunks = chunk_note(note, chunk_size=512, overlap=0)
    assert len(chunks) >= 2
    heading_paths = [c.meta["heading_path"] for c in chunks]
    assert any("新版协议设计" in h for h in heading_paths)
    assert any("parse定义" in h for h in heading_paths)


def test_chunk_protects_code_block():
    note = NoteDoc(source="t.md", title="t", text=SAMPLE.split("---\n", 2)[-1].strip())
    chunks = chunk_note(note, chunk_size=512, overlap=0)
    # 代码块应完整保留在某个 chunk 里
    code_kept = any("def parse" in c.text and "return data" in c.text for c in chunks)
    assert code_kept


def test_chunk_notes_metadata():
    note = NoteDoc(source="2026/x.md", title="x笔记", text="# 标题\n内容")
    out = chunk_notes([note])
    assert len(out) >= 1
    assert out[0]["source"] == "2026/x.md"
    assert out[0]["title"] == "x笔记"
    assert "heading_path" in out[0]
