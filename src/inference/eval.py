# coding: utf-8
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_PREFIX_RE = re.compile(r"^(YEAR|COUNT|BOOL)::", flags=re.IGNORECASE)
_MULTI_SPACE_RE = re.compile(r"\s+")


def normalize_answer(s: Any) -> str:
    if s is None:
        return ""
    t = str(s).strip().lower()
    t = _PREFIX_RE.sub("", t)
    t = _MULTI_SPACE_RE.sub(" ", t).strip()
    while t.endswith(".") or t.endswith("。"):
        t = t[:-1].rstrip()
    return t


def _safe_get(d: Any, path: List[str], default: Any = None) -> Any:
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return default
        if k not in cur:
            return default
        cur = cur[k]
    return cur


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {line_no}: {e}") from e
            if isinstance(obj, dict):
                yield obj


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _evidence_list(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    ev = rec.get("pred_evidence", None)
    if ev is None:
        ev = rec.get("evidence", None)
    ev_list = _as_list(ev)
    out: List[Dict[str, Any]] = []
    for item in ev_list:
        if isinstance(item, dict):
            out.append(item)
    return out


def _extract_router_topic(rec: Dict[str, Any]) -> str:
    t = _safe_get(rec, ["router", "assigned_topic"], "")
    if t:
        return str(t)
    # fallback: meta.topic（main_infer 会写入）
    t = _safe_get(rec, ["meta", "topic"], "")
    return str(t) if t else ""


def _extract_relation_candidate(rec: Dict[str, Any]) -> str:
    rc = _safe_get(rec, ["gating", "relation_candidate"], "")
    return str(rc) if rc is not None else ""


def _extract_gold_fields(rec: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """
    gold_answer_text / gold_topic / gold_relation 优先从 jsonl 顶层读取（main_infer 已写入）
    其次兼容 rec["gold"] 子结构
    """
    gold_answer = rec.get("gold_answer_text") or _safe_get(rec, ["gold", "answer_text"], "") or ""
    gold_topic = rec.get("gold_topic") or _safe_get(rec, ["gold", "topic"], "") or ""
    gold_relation = rec.get("gold_relation") or _safe_get(rec, ["gold", "relation"], "") or ""
    gold_tail = rec.get("gold_tail") or gold_answer or ""
    return str(gold_answer), str(gold_topic), str(gold_relation), str(gold_tail)


def _extract_pred_answer_value(rec: Dict[str, Any]) -> str:
    # 优先用 pred_answer_value（main_infer 已写入）
    v = rec.get("pred_answer_value")
    if v is not None and str(v).strip():
        return str(v)
    # fallback: 用 pred_answer_text（不推荐）
    return str(rec.get("pred_answer_text") or rec.get("answer_text") or "")


def _evidence_contains_gold_tail(evidence: List[Dict[str, Any]], gold_tail: str) -> bool:
    g = normalize_answer(gold_tail)
    if not g:
        return False
    for ev in evidence:
        tail = normalize_answer(ev.get("tail", ""))
        if tail and tail == g:
            return True
    return False


@dataclass
class CounterBucket:
    n: int = 0
    em: int = 0
    ev_empty: int = 0
    routing_ok: int = 0
    ev_contains_gold: int = 0
    rel_ok: int = 0

    def add(
        self,
        *,
        em: bool,
        ev_empty: bool,
        routing_ok: Optional[bool],
        ev_contains_gold: Optional[bool],
        rel_ok: Optional[bool],
    ) -> None:
        self.n += 1
        self.em += int(bool(em))
        self.ev_empty += int(bool(ev_empty))
        if routing_ok is not None:
            self.routing_ok += int(bool(routing_ok))
        if ev_contains_gold is not None:
            self.ev_contains_gold += int(bool(ev_contains_gold))
        if rel_ok is not None:
            self.rel_ok += int(bool(rel_ok))

    def to_metrics(self) -> Dict[str, Any]:
        n = max(self.n, 1)
        return {
            "n": self.n,
            "em": self.em / n,
            "evidence_empty_rate": self.ev_empty / n,
            "routing_accuracy": self.routing_ok / n,
            "evidence_contains_gold_rate": self.ev_contains_gold / n,
            "relation_accuracy": self.rel_ok / n,
        }


def evaluate_jsonl(preds_jsonl: Path) -> Dict[str, Any]:
    overall = CounterBucket()
    by_relation: Dict[str, CounterBucket] = defaultdict(CounterBucket)

    n_with_gold_answer = 0
    n_with_gold_topic = 0
    n_with_gold_relation = 0

    for rec in _read_jsonl(preds_jsonl):
        pred_answer_val = _extract_pred_answer_value(rec)
        gold_answer, gold_topic, gold_relation, gold_tail = _extract_gold_fields(rec)

        pred_topic = _extract_router_topic(rec)
        pred_relation = _extract_relation_candidate(rec)

        ev_list = _evidence_list(rec)

        em = normalize_answer(pred_answer_val) == normalize_answer(gold_answer) if gold_answer else False
        ev_empty = len(ev_list) == 0

        routing_ok: Optional[bool] = None
        if gold_topic:
            n_with_gold_topic += 1
            routing_ok = normalize_answer(pred_topic) == normalize_answer(gold_topic)

        rel_ok: Optional[bool] = None
        if gold_relation:
            n_with_gold_relation += 1
            rel_ok = normalize_answer(pred_relation) == normalize_answer(gold_relation)

        ev_contains_gold: Optional[bool] = None
        if gold_tail or gold_answer:
            if gold_answer:
                n_with_gold_answer += 1
            ev_contains_gold = _evidence_contains_gold_tail(ev_list, gold_tail or gold_answer)

        overall.add(
            em=bool(em),
            ev_empty=bool(ev_empty),
            routing_ok=routing_ok,
            ev_contains_gold=ev_contains_gold,
            rel_ok=rel_ok,
        )

        bucket_key = pred_relation.strip() if pred_relation.strip() else "__NONE__"
        by_relation[bucket_key].add(
            em=bool(em),
            ev_empty=bool(ev_empty),
            routing_ok=routing_ok,
            ev_contains_gold=ev_contains_gold,
            rel_ok=rel_ok,
        )

    metrics = {
        "preds_jsonl": str(preds_jsonl),
        "overall": overall.to_metrics(),
        "by_relation_candidate": {k: v.to_metrics() for k, v in sorted(by_relation.items(), key=lambda x: x[0])},
        "coverage": {
            "n_total": overall.n,
            "n_with_gold_answer": n_with_gold_answer,
            "n_with_gold_topic": n_with_gold_topic,
            "n_with_gold_relation": n_with_gold_relation,
        },
    }
    return metrics


def _print_metrics(metrics: Dict[str, Any], *, top_relations: int = 50) -> None:
    overall = metrics.get("overall", {})
    print("=== Overall ===")
    print(f"n: {overall.get('n')}")
    print(f"EM: {overall.get('em'):.4f}")
    print(f"Evidence Empty Rate: {overall.get('evidence_empty_rate'):.4f}")
    print(f"Routing Accuracy: {overall.get('routing_accuracy'):.4f}")
    print(f"Evidence Contains Gold Rate: {overall.get('evidence_contains_gold_rate'):.4f}")
    print(f"Relation Accuracy: {overall.get('relation_accuracy'):.4f}")

    print("\n=== By relation_candidate (show limited) ===")
    br: Dict[str, Any] = metrics.get("by_relation_candidate", {})
    keys = list(br.keys())[: max(int(top_relations), 0)]
    for k in keys:
        m = br[k]
        print(f"[{k}] n={m.get('n')} em={m.get('em'):.4f} empty_ev={m.get('evidence_empty_rate'):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--preds_jsonl",
        type=str,
        required=False,
        default=r"D:\Project\ConvQA\outputs\preds.jsonl",
        help=r"推理输出 JSONL，例如 D:\Project\ConvQA\outputs\preds.jsonl",
    )
    ap.add_argument("--metrics_json", type=str, default="", help="可选：保存 metrics.json 的路径")
    ap.add_argument("--top_relations", type=int, default=50, help="按 relation 分桶时打印前多少个桶")
    args = ap.parse_args()

    preds_path = Path(args.preds_jsonl)
    metrics = evaluate_jsonl(preds_path)
    _print_metrics(metrics, top_relations=int(args.top_relations))

    if str(args.metrics_json).strip():
        out = Path(args.metrics_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()