# coding: utf-8
"""
Step 4: Gating (C 的第二步)

修复点：
- 兼容 memory_store 返回 MemoryTriple(dataclass/object) 而不是 dict：
  使用 _triple_field() 读取 head/tail/relation 等字段，避免 AttributeError
- 补充 import: is_dataclass
"""

from __future__ import annotations

import re
from dataclasses import dataclass, is_dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence


# -------------------------
# Shared relation inference
# -------------------------
def infer_relation(original_question: str) -> str:
    """
    与 extractor 侧尽量保持一致的 relation 规则（最小覆盖版）。
    注意：relation 名必须与 KG 中一致。
    """
    q = (original_question or "").strip().lower()

    # first/final/num_books/when + award/prize
    if "first" in q and "book" in q:
        return "first_book"
    if any(k in q for k in ["final", "ended", "concluded", "last book"]):
        return "final_book"
    if any(k in q for k in ["how many", "amount", "number of books", "number of book", "number of"]):
        return "num_books"

    when_only = (q == "when") or ("when" in q)
    if when_only:
        if any(k in q for k in ["award", "prize"]):
            return "award_year"
        return "publication_year"

    # common attribute-like
    if any(k in q for k in ["author", "writer", "wrote"]):
        return "author"
    if any(k in q for k in ["nationality", "country"]):
        return "nationality"
    if "year" in q:
        return "publication_year"
    if "title" in q:
        return "book_title"
    if "award" in q:
        return "award"
    if "publisher" in q:
        return "publisher"
    if "genre" in q:
        return "genre"

    return "related_to"


# -------------------------
# Output schema
# -------------------------
@dataclass(frozen=True)
class GatingResult:
    use_history: bool
    history_turn_ids: List[int]
    head_candidates: List[str]
    relation_candidate: str
    gate_reason: str = ""


# -------------------------
# Gating core
# -------------------------
class Gating:
    """
    简单但可执行的 gating:
    - use_history: 指代词/短问/when/who/where 等默认 True；否则若 topic/entity 明确可 False
    - history_turn_ids: 取 assigned_seg 覆盖范围内最近 n 轮（按 turn 过滤）
    - head_candidates:
        1) 当前问题中显式出现的实体（字符串匹配到已知实体表）
        2) assigned_segment.topic
        3) assigned segment 最近 n 轮的 answer_text（直接用 history_turns 取）
        4) assigned_segment.key_entities top-m
        5) 可选：从 memory_store 按 (conv_id, topic) 取 triples，把 tail/head 补充进来
    """

    _PRONOUN_PAT = re.compile(
        r"\b(when|where|who|whom|whose|which|it|he|she|they|them|this|that|these|those)\b",
        flags=re.IGNORECASE,
    )

    _ZH_TRIGGERS = (
        "他",
        "她",
        "它",
        "他们",
        "她们",
        "它们",
        "这",
        "那",
        "这个",
        "那个",
        "这些",
        "那些",
        "什么时候",
        "何时",
        "哪里",
        "在哪",
        "在哪儿",
        "谁",
    )

    _SHORT_Q_WORDS = {
        "when",
        "where",
        "who",
        "what",
        "which",
        "why",
        "how",
        "it",
        "he",
        "she",
        "they",
        "this",
        "that",
        "these",
        "those",
        "again",
        "then",
    }

    def __init__(
        self,
        *,
        recent_n_turns: int = 3,
        key_entities_top_m: int = 10,
        max_head_candidates: int = 30,
        min_entity_len: int = 2,
    ):
        self.recent_n_turns = max(0, int(recent_n_turns))
        self.key_entities_top_m = max(0, int(key_entities_top_m))
        self.max_head_candidates = max(1, int(max_head_candidates))
        self.min_entity_len = max(1, int(min_entity_len))

    @staticmethod
    def _norm(s: Any) -> str:
        if s is None:
            return ""
        return str(s).strip()

    @staticmethod
    def _casefold(s: str) -> str:
        return s.casefold()

    @staticmethod
    def _safe_int(x: Any) -> Optional[int]:
        try:
            return int(x)
        except (TypeError, ValueError):
            return None

    def _is_pronoun_or_short_question(self, q_t: str) -> bool:
        q = self._norm(q_t)
        if not q:
            return True

        q_cf = self._casefold(q)

        for z in self._ZH_TRIGGERS:
            if z in q:
                return True

        if self._PRONOUN_PAT.search(q):
            return True

        tokens = re.findall(r"[A-Za-z]+", q_cf)
        if 0 < len(tokens) <= 3 and all(t in self._SHORT_Q_WORDS for t in tokens):
            return True

        if len(q) <= 6:
            return True

        return False

    def _select_recent_turns_in_segment(
        self,
        assigned_seg: Optional[Dict[str, Any]],
        history_turns: Sequence[Dict[str, Any]],
    ) -> List[int]:
        if not assigned_seg or not history_turns:
            return []

        start_turn = self._safe_int(assigned_seg.get("start_turn"))
        end_turn = self._safe_int(assigned_seg.get("end_turn"))
        if start_turn is None or end_turn is None:
            return []

        in_seg: List[int] = []
        for t in history_turns:
            turn_id = self._safe_int(t.get("turn"))
            if turn_id is None:
                continue
            if start_turn <= turn_id <= end_turn:
                in_seg.append(turn_id)

        in_seg = sorted(set(in_seg))
        if self.recent_n_turns <= 0:
            return []
        return in_seg[-self.recent_n_turns :]

    def _collect_recent_answers_as_heads(
        self,
        history_turns: Sequence[Dict[str, Any]],
        selected_turn_ids: Sequence[int],
    ) -> List[str]:
        if not history_turns or not selected_turn_ids:
            return []

        sel = set(int(x) for x in selected_turn_ids)
        out: List[str] = []
        for t in sorted(history_turns, key=lambda x: (self._safe_int(x.get("turn")) or 10**18)):
            turn_id = self._safe_int(t.get("turn"))
            if turn_id is None or turn_id not in sel:
                continue
            ans = self._norm(t.get("answer_text"))
            if ans and ans not in out:
                out.append(ans)
        return out

    def _match_known_entities_in_question(self, q_t: str, known_entities: Iterable[str]) -> List[str]:
        q = self._norm(q_t)
        q_cf = self._casefold(q)

        ents = []
        for e in known_entities:
            s = self._norm(e)
            if len(s) < self.min_entity_len:
                continue
            ents.append(s)

        ents.sort(key=lambda x: len(x), reverse=True)

        matched: List[str] = []
        seen_cf = set()
        for e in ents:
            e_cf = self._casefold(e)
            if not e_cf or e_cf in seen_cf:
                continue
            if e_cf in q_cf:
                seen_cf.add(e_cf)
                matched.append(e)
        return matched

    @staticmethod
    def _dedup_keep_order(items: Iterable[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for x in items:
            if x is None:
                continue
            s = str(x).strip()
            if not s:
                continue
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    def _triple_field(self, t: Any, key: str, default: Any = "") -> Any:
        """
        兼容 MemoryTriple(对象/数据类) 和 dict 两种表示：
        - dict: t.get(key)
        - object/dataclass: getattr(t, key)
        """
        if t is None:
            return default
        if isinstance(t, dict):
            return t.get(key, default)
        if is_dataclass(t) or hasattr(t, key):
            return getattr(t, key, default)
        return default

    def gate(
        self,
        *,
        q_t: str,
        assigned_seg: Optional[Dict[str, Any]],
        history_turns: Sequence[Dict[str, Any]],
        memory_store: Optional[Any] = None,
        conv_id: Optional[str] = None,
        known_entities: Optional[Iterable[str]] = None,
    ) -> GatingResult:
        """
        memory_store: 兼容：
          - MemoryTripleStore（segmenter.py）：get_triples_by_conv_topic(conv_id, topic) -> list[dict]
          - MemoryKG（memory_kg.py）：get_triples_by_conv_topic(conv_id, topic) -> list[MemoryTriple dataclass]
        """
        q = self._norm(q_t)
        relation_candidate = infer_relation(q)

        history_turn_ids = self._select_recent_turns_in_segment(assigned_seg, history_turns)

        needs_history = self._is_pronoun_or_short_question(q)
        if assigned_seg is not None and relation_candidate in {
            "publication_year",
            "award_year",
            "author",
            "nationality",
            "first_book",
            "final_book",
            "num_books",
        }:
            needs_history = True

        head_candidates: List[str] = []

        # (1) 当前问题显式出现的实体
        if known_entities is not None:
            head_candidates.extend(self._match_known_entities_in_question(q, known_entities))

        # (2) assigned_segment.topic
        topic = self._norm(assigned_seg.get("topic")) if assigned_seg else ""
        if topic:
            head_candidates.append(topic)

        # (3) segment 最近 n 轮答案
        head_candidates.extend(self._collect_recent_answers_as_heads(history_turns, history_turn_ids))

        # (4) segment.key_entities top-m
        if assigned_seg:
            ke = assigned_seg.get("key_entities") or []
            if isinstance(ke, list) and ke:
                head_candidates.extend([self._norm(x) for x in ke[: self.key_entities_top_m]])

        # (5) 从 memory_store 拉 (conv_id, topic) 的 triples，补充 head/tail
        if memory_store is not None and conv_id and topic:
            triples = []
            try:
                # MemoryTripleStore / MemoryKG 都应该实现这个方法
                triples = memory_store.get_triples_by_conv_topic(str(conv_id), str(topic))
            except Exception:
                triples = []

            for t in triples or []:
                h = self._norm(self._triple_field(t, "head", ""))
                tail = self._norm(self._triple_field(t, "tail", ""))
                if h:
                    head_candidates.append(h)
                if tail:
                    head_candidates.append(tail)

        head_candidates = self._dedup_keep_order(head_candidates)[: self.max_head_candidates]

        if not head_candidates and topic:
            head_candidates = [topic]

        use_history = bool(needs_history and len(history_turn_ids) > 0)

        gate_reason = []
        if assigned_seg is None:
            gate_reason.append("no_assigned_seg")
        if needs_history:
            gate_reason.append("needs_history")
        if history_turn_ids:
            gate_reason.append(f"history_turns={len(history_turn_ids)}")
        gate_reason.append(f"rel={relation_candidate}")
        gate_reason.append(f"heads={len(head_candidates)}")

        return GatingResult(
            use_history=use_history,
            history_turn_ids=list(history_turn_ids),
            head_candidates=list(head_candidates),
            relation_candidate=relation_candidate,
            gate_reason=";".join(gate_reason),
        )


__all__ = [
    "GatingResult",
    "Gating",
    "infer_relation",
]