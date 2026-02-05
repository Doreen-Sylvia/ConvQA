# coding: utf-8
"""
Segmenter（适配器版）：

你原来的 segmenter.py 已经实现了：
- DialogueSegmenter.segment_questions(questions) -> List[Segment]
- MemoryTripleStore(memory_tsv).load()
- KeyEntityExtractor.extract_for_segments(...)

但 main_infer.py 期望存在：
- class Segmenter
- segger.segment(conv) 或 segger.segment(turns)

因此这里新增 Segmenter 适配器类（不破坏原有实现），保证 main_infer 直接可用，
并且尽可能把 key_entities/key_relations 附加到 segment dict 上，从而让 Router 的 entity_hit 生效。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import Counter, defaultdict


@dataclass(frozen=True)
class Segment:
    """
    topic 分段结果（完全由 question['topic'] 推导）。
    """
    seg_id: int
    topic: str
    start_turn: int
    end_turn: int
    turn_ids: List[int]


class DialogueSegmenter:
    """
    对 merged dialogue 的 questions 做按 topic 的历史分段。
    规则：topic 变化 == boundary；topic/turn 缺失则跳过该 turn。
    """

    @staticmethod
    def segment_questions(questions: List[Dict]) -> List[Segment]:
        segments: List[Segment] = []

        cur_topic: Optional[str] = None
        cur_turn_ids: List[int] = []
        cur_start: Optional[int] = None
        cur_end: Optional[int] = None

        def flush() -> None:
            nonlocal cur_topic, cur_turn_ids, cur_start, cur_end
            if cur_topic is None or cur_start is None or cur_end is None or not cur_turn_ids:
                cur_topic, cur_turn_ids, cur_start, cur_end = None, [], None, None
                return
            segments.append(
                Segment(
                    seg_id=len(segments),
                    topic=cur_topic,
                    start_turn=cur_start,
                    end_turn=cur_end,
                    turn_ids=list(cur_turn_ids),
                )
            )
            cur_topic, cur_turn_ids, cur_start, cur_end = None, [], None, None

        for q in questions or []:
            topic = (q.get("topic") or "").strip()
            if not topic:
                continue

            turn_raw = q.get("turn")
            if turn_raw is None:
                continue
            try:
                turn_id = int(turn_raw)
            except (TypeError, ValueError):
                continue

            if cur_topic is None:
                cur_topic = topic
                cur_start = turn_id
                cur_end = turn_id
                cur_turn_ids = [turn_id]
                continue

            if topic != cur_topic:
                flush()
                cur_topic = topic
                cur_start = turn_id
                cur_end = turn_id
                cur_turn_ids = [turn_id]
            else:
                cur_end = turn_id
                cur_turn_ids.append(turn_id)

        flush()
        return segments

    @staticmethod
    def segment_turns(turns: List[Dict[str, Any]]) -> List[Segment]:
        """
        兼容：当只传入 turns（没有 conv_id）时，也能基于 turn['topic'] 分段。
        turns 的 key 可能是 questions 也可能是 turns，字段名一样：topic/turn。
        """
        return DialogueSegmenter.segment_questions(turns)


class MemoryTripleStore:
    """
    读取并索引 `data/preprocessed/dialogue_kg/memory_triples.tsv`。

    依赖字段：conv_id, turn_id, topic, head, relation, tail
    """

    REQUIRED_FIELDS = ("conv_id", "turn_id", "topic", "head", "relation", "tail")

    def __init__(self, memory_tsv_path: str | Path):
        self.memory_tsv_path = Path(memory_tsv_path)
        self._by_conv_topic: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)

    def load(self) -> None:
        if not self.memory_tsv_path.exists():
            raise FileNotFoundError(f"missing file: {self.memory_tsv_path}")

        with self.memory_tsv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if reader.fieldnames is None:
                raise ValueError("memory triples tsv has no header")

            missing = [k for k in self.REQUIRED_FIELDS if k not in set(reader.fieldnames)]
            if missing:
                raise ValueError(f"memory triples tsv missing fields: {missing}")

            for row in reader:
                conv_id = (row.get("conv_id") or "").strip()
                topic = (row.get("topic") or "").strip()
                if not conv_id or not topic:
                    continue

                self._by_conv_topic[(conv_id, topic)].append(
                    {
                        "conv_id": conv_id,
                        "turn_id": (row.get("turn_id") or "").strip(),
                        "topic": topic,
                        "head": (row.get("head") or "").strip(),
                        "relation": (row.get("relation") or "").strip(),
                        "tail": (row.get("tail") or "").strip(),
                    }
                )

    def get_triples_by_conv_topic(self, conv_id: str, topic: str) -> List[Dict[str, str]]:
        return list(self._by_conv_topic.get((str(conv_id), str(topic)), []))

    def get_triples_by_conv_seg(self, conv_id: str, segment: Segment) -> List[Dict[str, str]]:
        return self.get_triples_by_conv_topic(conv_id=str(conv_id), topic=segment.topic)


class KeyEntityExtractor:
    """
    从某段相关 triples 里提取 key_entities / key_relations。
    """

    @staticmethod
    def extract_key_entities(
        triples: Iterable[Dict[str, str]],
        top_k: int = 10,
        *,
        drop_bool: bool = True,
        drop_typed_literals_prefixes: Optional[Tuple[str, ...]] = ("YEAR::", "COUNT::", "BOOL::"),
    ) -> List[str]:
        cnt = Counter()
        for t in triples:
            h = (t.get("head") or "").strip()
            tail = (t.get("tail") or "").strip()

            if h:
                if not (drop_bool and h.startswith("BOOL::")):
                    cnt[h] += 1

            if tail:
                if drop_bool and tail.startswith("BOOL::"):
                    continue
                if drop_typed_literals_prefixes and any(tail.startswith(p) for p in drop_typed_literals_prefixes):
                    continue
                cnt[tail] += 1

        return [x for x, _ in cnt.most_common(max(0, int(top_k)))]

    @staticmethod
    def extract_key_relations(
        triples: Iterable[Dict[str, str]],
        top_k: int = 5,
    ) -> List[str]:
        cnt = Counter()
        for t in triples:
            rel = (t.get("relation") or "").strip()
            if rel:
                cnt[rel] += 1
        return [x for x, _ in cnt.most_common(max(0, int(top_k)))]

    @classmethod
    def extract_for_segments(
        cls,
        *,
        store: MemoryTripleStore,
        conv_id: str,
        segments: List[Segment],
        top_k_entities: int = 10,
        top_k_relations: int = 5,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for seg in segments:
            triples = store.get_triples_by_conv_seg(conv_id, seg)
            out.append(
                {
                    "seg_id": seg.seg_id,
                    "topic": seg.topic,
                    "start_turn": seg.start_turn,
                    "end_turn": seg.end_turn,
                    "key_entities": cls.extract_key_entities(triples, top_k=top_k_entities),
                    "key_relations": cls.extract_key_relations(triples, top_k=top_k_relations),
                }
            )
        return out


def build_segments_and_keys_for_dialogue(
    dialogue: Dict[str, Any],
    *,
    store: MemoryTripleStore,
    top_k_entities: int = 10,
    top_k_relations: int = 5,
) -> Dict[str, Any]:
    conv_id = (dialogue.get("conv_id") or "").strip()
    questions = dialogue.get("questions") or []
    segments = DialogueSegmenter.segment_questions(questions)

    seg_keys = KeyEntityExtractor.extract_for_segments(
        store=store,
        conv_id=conv_id,
        segments=segments,
        top_k_entities=top_k_entities,
        top_k_relations=top_k_relations,
    )

    return {"conv_id": conv_id, "segments": seg_keys}


class Segmenter:
    """
    main_infer.py 适配器：提供 segment() 方法。

    用法（main_infer 里）：
      segger = Segmenter()
      segs = segger.segment(turns) 或 segger.segment(conv)

    初始化参数：
    - memory_tsv_path: 若提供，则会加载 MemoryTripleStore 并附加 key_entities/key_relations；
      若不提供，则只做 topic 分段（key_entities 为空，但 topic/recency routing 仍可用）
    """

    def __init__(
        self,
        memory_tsv_path: Optional[str | Path] = None,
        *,
        top_k_entities: int = 10,
        top_k_relations: int = 5,
        auto_load_store: bool = True,
    ) -> None:
        self.top_k_entities = int(top_k_entities)
        self.top_k_relations = int(top_k_relations)
        self.store: Optional[MemoryTripleStore] = None

        if memory_tsv_path is not None:
            self.store = MemoryTripleStore(memory_tsv_path)
            if auto_load_store:
                self.store.load()

    @staticmethod
    def _get_conv_id(conv: Dict[str, Any]) -> str:
        for k in ("conv_id", "conversation_id", "id", "convId"):
            v = (conv.get(k) or "").strip() if isinstance(conv.get(k), str) else str(conv.get(k) or "").strip()
            if v:
                return v
        return ""

    def segment(self, obj: Any) -> List[Dict[str, Any]]:
        """
        obj 可以是：
        - conv dict（包含 conv_id + questions/turns）
        - turns list（每个元素含 topic/turn）
        """
        # case 1: conv dict
        if isinstance(obj, dict):
            conv_id = self._get_conv_id(obj)
            turns = obj.get("questions") or obj.get("turns") or []
            if not isinstance(turns, list):
                turns = []

            segs = DialogueSegmenter.segment_questions([x for x in turns if isinstance(x, dict)])

            # 如果没有 store 或没有 conv_id，就返回不带 key_entities 的最小段信息
            if self.store is None or not conv_id:
                return [
                    {
                        "seg_id": s.seg_id,
                        "topic": s.topic,
                        "start_turn": s.start_turn,
                        "end_turn": s.end_turn,
                        "key_entities": [],
                        "key_relations": [],
                    }
                    for s in segs
                ]

            # 有 store + conv_id：附加 key entities/relations
            return KeyEntityExtractor.extract_for_segments(
                store=self.store,
                conv_id=conv_id,
                segments=segs,
                top_k_entities=self.top_k_entities,
                top_k_relations=self.top_k_relations,
            )

        # case 2: turns list
        if isinstance(obj, list):
            turns = [x for x in obj if isinstance(x, dict)]
            segs = DialogueSegmenter.segment_turns(turns)
            # 没 conv_id 的情况下无法查 store，返回最小段信息
            return [
                {
                    "seg_id": s.seg_id,
                    "topic": s.topic,
                    "start_turn": s.start_turn,
                    "end_turn": s.end_turn,
                    "key_entities": [],
                    "key_relations": [],
                }
                for s in segs
            ]

        return []


__all__ = [
    "Segment",
    "DialogueSegmenter",
    "MemoryTripleStore",
    "KeyEntityExtractor",
    "build_segments_and_keys_for_dialogue",
    "Segmenter",
]