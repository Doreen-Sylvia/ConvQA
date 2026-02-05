# coding: utf-8

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Set, Any

from src.inference.memory_kg import MemoryKG


@dataclass(frozen=True)
class EvidenceTriple:
    head: str
    relation: str
    tail: str
    turn_id: int
    scope: str  # "topic" or "conv"


class KGExecutor:
    """
    Step 5.2: KG 执行检索

    清晰接口:
      execute(conv_id, topic, head_candidates, relation_candidate, top_k=3,
              current_turn=None, allow_current_turn=False) -> evidence_triples

    逻辑:
    1) 优先在 (conv_id, topic) 子图里查邻接表
    2) 若无结果，回退到 conv 全图扫描
    3) 排序: turn_id 越大越近越优先; head 在 head_candidates 中越靠前越优先
    4) 时间因果约束（模拟在线）：
       - 若 current_turn 为 None：不做过滤
       - allow_current_turn=False：仅使用 turn_id < current_turn 的证据
       - allow_current_turn=True ：允许 turn_id <= current_turn 的证据
    """

    def __init__(self, kg: MemoryKG) -> None:
        self.kg = kg

    @staticmethod
    def _norm(s: Any) -> str:
        if s is None:
            return ""
        return str(s).strip()

    @staticmethod
    def _safe_int(x: Any) -> Optional[int]:
        try:
            return int(x)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _dedup_keep_order(items: Iterable[str]) -> List[str]:
        out: List[str] = []
        seen: Set[str] = set()
        for x in items:
            s = KGExecutor._norm(x)
            if not s:
                continue
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    @staticmethod
    def _rel_match(rel: str, rel_cand: Optional[str]) -> bool:
        if rel_cand is None:
            return True
        return KGExecutor._norm(rel) == KGExecutor._norm(rel_cand)

    @staticmethod
    def _sort_key(ev: EvidenceTriple, head_rank: Dict[str, int]) -> Tuple[int, int]:
        # turn_id 降序 -> 用 -turn_id
        # head 越靠前越优先 -> rank 升序
        return (-int(ev.turn_id), int(head_rank.get(ev.head, 10**9)))

    @staticmethod
    def _time_allowed(
        *,
        evidence_turn: Optional[int],
        current_turn: Optional[int],
        allow_current_turn: bool,
    ) -> bool:
        """
        判断证据是否满足在线时间约束。
        - current_turn=None：不约束
        - allow_current_turn=False：evidence_turn < current_turn
        - allow_current_turn=True ：evidence_turn <= current_turn
        """
        if current_turn is None:
            return True
        if evidence_turn is None:
            return False
        if allow_current_turn:
            return evidence_turn <= current_turn
        return evidence_turn < current_turn

    def execute(
        self,
        conv_id: str,
        topic: str,
        head_candidates: Sequence[str],
        relation_candidate: Optional[str],
        top_k: int = 3,
        *,
        current_turn: Optional[int] = None,
        allow_current_turn: bool = False,
    ) -> List[EvidenceTriple]:
        conv_id_n = self._norm(conv_id)
        topic_n = self._norm(topic)
        rel_cand_n = self._norm(relation_candidate) if relation_candidate is not None else None

        heads = self._dedup_keep_order(head_candidates or [])
        if not conv_id_n or not heads:
            return []

        head_rank = {h: i for i, h in enumerate(heads)}
        k = max(0, int(top_k))

        # 1) topic 子图优先: adj[(conv_id, topic, head)]
        evidence: List[EvidenceTriple] = []
        if topic_n:
            for h in heads:
                for rel, tail, turn_id in self.kg.neighbors(conv_id_n, topic_n, h):
                    if not self._rel_match(rel, rel_cand_n):
                        continue
                    tid = self._safe_int(turn_id)
                    if not self._time_allowed(
                        evidence_turn=tid,
                        current_turn=current_turn,
                        allow_current_turn=allow_current_turn,
                    ):
                        continue
                    evidence.append(
                        EvidenceTriple(
                            head=h,
                            relation=self._norm(rel),
                            tail=self._norm(tail),
                            turn_id=int(tid),
                            scope="topic",
                        )
                    )

        evidence.sort(key=lambda x: self._sort_key(x, head_rank))
        if k > 0 and len(evidence) >= k:
            return evidence[:k]

        # 2) 回退: conv 全图扫描（避免 topic 过严导致空）
        fallback: List[EvidenceTriple] = []
        triples = self.kg.get_triples_by_conv(conv_id_n)
        if triples:
            heads_set = set(heads)
            for t in triples:
                if t.head not in heads_set:
                    continue
                if not self._rel_match(t.relation, rel_cand_n):
                    continue
                tid = self._safe_int(getattr(t, "turn_id", None))
                if not self._time_allowed(
                    evidence_turn=tid,
                    current_turn=current_turn,
                    allow_current_turn=allow_current_turn,
                ):
                    continue
                fallback.append(
                    EvidenceTriple(
                        head=self._norm(t.head),
                        relation=self._norm(t.relation),
                        tail=self._norm(t.tail),
                        turn_id=int(tid),
                        scope="conv",
                    )
                )

        fallback.sort(key=lambda x: self._sort_key(x, head_rank))

        # 合并去重: 同一 (head, relation, tail, turn_id) 只保留一次，优先保留 topic 结果
        seen_keys: Set[Tuple[str, str, str, int]] = set()
        merged: List[EvidenceTriple] = []

        for ev in evidence + fallback:
            key = (ev.head, ev.relation, ev.tail, int(ev.turn_id))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged.append(ev)

        merged.sort(key=lambda x: self._sort_key(x, head_rank))
        return merged[:k] if k > 0 else merged


def execute(
    kg: MemoryKG,
    conv_id: str,
    topic: str,
    head_candidates: Sequence[str],
    relation_candidate: Optional[str],
    top_k: int = 3,
    *,
    current_turn: Optional[int] = None,
    allow_current_turn: bool = False,
) -> List[EvidenceTriple]:
    """
    函数式接口（便于直接调用）:
      execute(kg, conv_id, topic, head_candidates, relation_candidate, top_k=3,
              current_turn=None, allow_current_turn=False)
    """
    return KGExecutor(kg).execute(
        conv_id=conv_id,
        topic=topic,
        head_candidates=head_candidates,
        relation_candidate=relation_candidate,
        top_k=top_k,
        current_turn=current_turn,
        allow_current_turn=allow_current_turn,
    )


__all__ = [
    "EvidenceTriple",
    "KGExecutor",
    "execute",
]