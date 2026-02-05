# coding: utf-8
"""
Step 3: Routing（修正版）

核心修复：
- 当 segments 非空时，任何路径都尽量返回 assigned_topic（至少 recent segment）
  避免 pred_topic 为空导致 routing_accuracy 恒为 0
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class RouteResult:
    assigned_seg_id: Optional[int]
    assigned_topic: Optional[str]
    route_reason: str


class Router:
    _PRONOUN_PAT = re.compile(
        r"\b(when|where|who|whom|whose|which|it|he|she|they|them|this|that|these|those)\b",
        flags=re.IGNORECASE,
    )

    _ZH_TRIGGERS = (
        "他", "她", "它", "他们", "她们", "它们", "这", "那", "这个", "那个", "这些", "那些",
        "什么时候", "何时", "哪里", "在哪", "在哪儿", "谁",
    )

    _SHORT_Q_WORDS = {
        "when", "where", "who", "what", "which", "why", "how",
        "it", "he", "she", "they", "this", "that", "these", "those",
        "again", "then",
    }

    def __init__(
        self,
        *,
        allow_new: bool = False,
        min_entity_len: int = 2,
        prefer_recent_on_tie: bool = True,
    ):
        self.allow_new = bool(allow_new)
        self.min_entity_len = int(min_entity_len)
        self.prefer_recent_on_tie = bool(prefer_recent_on_tie)

    @staticmethod
    def _norm_text(s: Optional[str]) -> str:
        if s is None:
            return ""
        return str(s).strip()

    @staticmethod
    def _casefold(s: str) -> str:
        return s.casefold()

    @staticmethod
    def _segment_end_turn(seg: Dict[str, Any]) -> int:
        end_turn = seg.get("end_turn")
        start_turn = seg.get("start_turn")
        seg_id = seg.get("seg_id")
        for v in (end_turn, start_turn, seg_id):
            try:
                return int(v)
            except (TypeError, ValueError):
                continue
        return -1

    def _get_recent_segment(self, segments: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not segments:
            return None
        return max(segments, key=self._segment_end_turn)

    def _is_pronoun_or_short_question(self, q_t: str) -> bool:
        q = self._norm_text(q_t)
        if not q:
            return True

        q_cf = self._casefold(q)

        for z in self._ZH_TRIGGERS:
            if z in q:
                return True

        if self._PRONOUN_PAT.search(q):
            return True

        tokens = re.findall(r"[A-Za-z]+", q_cf)
        if 0 < len(tokens) <= 3:
            if all(t in self._SHORT_Q_WORDS for t in tokens):
                return True

        if len(q) <= 6:
            return True

        return False

    def _topic_hit(self, q_cf: str, seg: Dict[str, Any]) -> bool:
        topic = self._norm_text(seg.get("topic"))
        if not topic:
            return False
        return self._casefold(topic) in q_cf

    def _entity_hit_score(self, q_cf: str, seg: Dict[str, Any]) -> Tuple[int, List[str]]:
        key_entities = seg.get("key_entities") or []
        if not isinstance(key_entities, list):
            return 0, []

        matched: List[str] = []
        seen = set()
        for ent in key_entities:
            e = self._norm_text(ent)
            if len(e) < self.min_entity_len:
                continue
            e_cf = self._casefold(e)
            if not e_cf:
                continue
            if e_cf in q_cf:
                if e_cf in seen:
                    continue
                seen.add(e_cf)
                matched.append(e)
        return len(matched), matched

    def route(self, q_t: str, segments: List[Dict[str, Any]]) -> RouteResult:
        q = self._norm_text(q_t)
        q_cf = self._casefold(q)

        if not segments:
            if self.allow_new:
                return RouteResult(None, None, "no_segments->NEW")
            return RouteResult(None, None, "no_segments(empty)")

        recent = self._get_recent_segment(segments)

        # 1) topic 命中
        topic_hits: List[Dict[str, Any]] = [seg for seg in segments if self._topic_hit(q_cf, seg)]
        if topic_hits:
            chosen = self._break_tie(topic_hits)
            return RouteResult(
                int(chosen.get("seg_id")) if chosen.get("seg_id") is not None else None,
                self._norm_text(chosen.get("topic")) or (self._norm_text(recent.get("topic")) if recent else None),
                "topic_hit",
            )

        # 2) key entity 命中
        best_seg: Optional[Dict[str, Any]] = None
        best_score = 0
        best_matches: List[str] = []

        for seg in segments:
            score, matches = self._entity_hit_score(q_cf, seg)
            if score <= 0:
                continue
            if score > best_score:
                best_seg, best_score, best_matches = seg, score, matches
            elif score == best_score and self.prefer_recent_on_tie and best_seg is not None:
                if self._segment_end_turn(seg) > self._segment_end_turn(best_seg):
                    best_seg, best_score, best_matches = seg, score, matches

        if best_seg is not None:
            reason = "key_entity_hit:" + ",".join(best_matches[:5])
            return RouteResult(
                int(best_seg.get("seg_id")) if best_seg.get("seg_id") is not None else None,
                self._norm_text(best_seg.get("topic")) or (self._norm_text(recent.get("topic")) if recent else None),
                reason,
            )

        # 3) 短问/指代：最近
        if self._is_pronoun_or_short_question(q):
            if recent is None:
                if self.allow_new:
                    return RouteResult(None, None, "pronoun_or_short->NEW")
                return RouteResult(None, None, "pronoun_or_short(no_recent)")
            return RouteResult(
                int(recent.get("seg_id")) if recent.get("seg_id") is not None else None,
                self._norm_text(recent.get("topic")) or None,
                "pronoun_or_short->recent",
            )

        # 4) 兜底：最近（确保 assigned_topic 非空）
        if recent is None:
            if self.allow_new:
                return RouteResult(None, None, "fallback->NEW")
            return RouteResult(None, None, "fallback_no_recent")
        return RouteResult(
            int(recent.get("seg_id")) if recent.get("seg_id") is not None else None,
            self._norm_text(recent.get("topic")) or None,
            "fallback->recent",
        )

    def _break_tie(self, segs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not segs:
            return {}
        if self.prefer_recent_on_tie:
            return max(segs, key=self._segment_end_turn)
        return segs[0]


__all__ = [
    "RouteResult",
    "Router",
]