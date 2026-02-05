# coding: utf-8
# File: src/inference/diagnostics.py

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
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


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"preds_jsonl not found: {path}")
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


def _safe_get(d: Any, path: List[str], default: Any = None) -> Any:
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return default
        if k not in cur:
            return default
        cur = cur[k]
    return cur


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


def _extract_pred_answer_value(rec: Dict[str, Any]) -> str:
    v = rec.get("pred_answer_value")
    if v is not None and str(v).strip():
        return str(v)
    return str(rec.get("pred_answer_text") or rec.get("answer_text") or rec.get("pred") or "")


def _extract_gold_fields(rec: Dict[str, Any]) -> Tuple[str, str, str, str]:
    gold_answer = rec.get("gold_answer_text") or _safe_get(rec, ["gold", "answer_text"], "") or ""
    gold_topic = rec.get("gold_topic") or _safe_get(rec, ["gold", "topic"], "") or ""
    gold_relation = rec.get("gold_relation") or _safe_get(rec, ["gold", "relation"], "") or ""
    gold_tail = rec.get("gold_tail") or gold_answer or ""
    return str(gold_answer), str(gold_topic), str(gold_relation), str(gold_tail)


def _extract_pred_topic(rec: Dict[str, Any]) -> str:
    t = _safe_get(rec, ["router", "assigned_topic"], "")
    if t:
        return str(t)
    t = _safe_get(rec, ["meta", "topic"], "")
    return str(t) if t else ""


def _extract_relation_candidate(rec: Dict[str, Any]) -> str:
    rc = _safe_get(rec, ["gating", "relation_candidate"], "")
    return str(rc) if rc is not None else ""


def _extract_question(rec: Dict[str, Any]) -> str:
    q = rec.get("question_text")
    if q is None:
        q = _safe_get(rec, ["turn", "question_text"], "")
    if not q:
        q = rec.get("question") or _safe_get(rec, ["turn", "question"], "") or ""
    return str(q)


def _extract_conv_turn_id(rec: Dict[str, Any], idx: int) -> Tuple[str, Optional[int]]:
    conv_id = (
        rec.get("conv_id")
        or rec.get("dialogue_id")
        or _safe_get(rec, ["meta", "conv_id"], "")
        or _safe_get(rec, ["meta", "dialogue_id"], "")
        or ""
    )
    turn_id = rec.get("turn_id")
    if turn_id is None:
        turn_id = _safe_get(rec, ["meta", "turn_id"], None)
    try:
        tid = int(turn_id) if turn_id is not None else None
    except Exception:
        tid = None
    if not conv_id:
        conv_id = f"__IDX__{idx}"
    return str(conv_id), tid


def _extract_head_used(rec: Dict[str, Any]) -> str:
    h = rec.get("head_used")
    if not h:
        h = _safe_get(rec, ["gating", "head_used"], "")
    return str(h or "")


def _norm_topic(t: str) -> str:
    return normalize_answer(t).replace(" ", "")


def _topic_mismatch(pred_topic: str, gold_topic: str) -> Optional[bool]:
    if not gold_topic:
        return None
    return _norm_topic(pred_topic) != _norm_topic(gold_topic)


def _evidence_contains_gold_tail(evidence: List[Dict[str, Any]], gold_tail: str) -> bool:
    g = normalize_answer(gold_tail)
    if not g:
        return False
    for ev in evidence:
        if normalize_answer(ev.get("tail", "")) == g:
            return True
    return False


def _evidence_relations(evidence: List[Dict[str, Any]]) -> List[str]:
    rels: List[str] = []
    for ev in evidence:
        r = ev.get("relation", "")
        if r is None:
            r = ""
        rr = normalize_answer(r)
        if rr:
            rels.append(rr)
    return rels


def _topic_name_appears_in_evidence(evidence: List[Dict[str, Any]], topic: str) -> bool:
    t = normalize_answer(topic)
    if not t:
        return False
    for ev in evidence:
        blob = " ".join(
            [
                str(ev.get("head", "") or ""),
                str(ev.get("relation", "") or ""),
                str(ev.get("tail", "") or ""),
            ]
        )
        if t in normalize_answer(blob):
            return True
    return False


def _head_suspicious(head_used: str, gold_topic: str, question_text: str) -> bool:
    h = normalize_answer(head_used)
    if not h:
        return False
    gt = normalize_answer(gold_topic)
    q = normalize_answer(question_text)

    # 强可解释的简单规则：
    # \- head 不包含 gold_topic 的任何线索
    # \- head 也没出现在问题文本附近（表示“从当前问句匹配不到”）
    if gt and gt not in h:
        if q and h not in q:
            return True
    return False


def _oracle_answer_in_topic(rec: Dict[str, Any], gold_topic: str, gold_answer: str) -> Optional[bool]:
    """
    可选 Oracle：在正确 topic 子图里，忽略 relation_candidate，
    看 topic_subgraph 中是否存在 tail==gold_answer 的 triple。

    依赖 preds.jsonl 里存在类似字段：
    \- rec["topic_subgraph"][topic] -> list[triple]
    triple 可以是 dict: {head, relation, tail} 或简化结构
    """
    if not gold_topic or not gold_answer:
        return None

    g_top = normalize_answer(gold_topic)
    g_ans = normalize_answer(gold_answer)
    if not g_top or not g_ans:
        return None

    topic_subgraph = rec.get("topic_subgraph")
    if not isinstance(topic_subgraph, dict):
        return None

    # key 可能不是完全一致：尝试直接命中，否则遍历找 normalize 后相等的 key
    triples_any: Any = None
    if gold_topic in topic_subgraph:
        triples_any = topic_subgraph.get(gold_topic)
    else:
        for k, v in topic_subgraph.items():
            if normalize_answer(k) == g_top:
                triples_any = v
                break

    triples = _as_list(triples_any)
    for t in triples:
        if isinstance(t, dict):
            tail = normalize_answer(t.get("tail", ""))
            if tail == g_ans:
                return True
        elif isinstance(t, (list, tuple)) and len(t) >= 3:
            tail = normalize_answer(t[2])
            if tail == g_ans:
                return True

    return False


@dataclass
class FailureCase:
    conv_id: str
    turn_id: Optional[int]
    question: str
    gold_answer: str
    pred_answer: str
    gold_topic: str
    pred_topic: str
    gold_relation: str
    relation_candidate: str
    head_used: str
    evidence: List[Dict[str, Any]]
    label: str
    oracle_answer_in_topic: Optional[bool]


def _assign_label(
    *,
    em: bool,
    evidence: List[Dict[str, Any]],
    pred_topic: str,
    gold_topic: str,
    relation_candidate: str,
    gold_relation: str,
    head_used: str,
    question_text: str,
    gold_answer: str,
    gold_tail: str,
) -> str:
    """
    失败分类优先级（A\~E）：
    (A) NO_EVIDENCE
    (B) ROUTING_ERROR
    (C) RELATION_MISMATCH
    (D) HEAD_SELECTION_ERROR
    (E) EVIDENCE_HAS_GOLD_BUT_PICKED_WRONG
    """
    if em:
        return "OK"

    if len(evidence) == 0:
        return "A_NO_EVIDENCE"

    # B: routing 错（gold_topic 存在且不一致），且 evidence 弱/指向别的 topic
    tm = _topic_mismatch(pred_topic, gold_topic)
    if tm is True:
        gold_in_ev = _topic_name_appears_in_evidence(evidence, gold_topic)
        pred_in_ev = _topic_name_appears_in_evidence(evidence, pred_topic) if pred_topic else False
        # evidence 里不含 gold_topic，且更像 pred_topic -> 高置信 routing 错
        if (not gold_in_ev) and (pred_in_ev or True):
            return "B_ROUTING_ERROR"

    # E: evidence 已经有 gold tail，但最后输出错
    if _evidence_contains_gold_tail(evidence, gold_tail or gold_answer):
        return "E_EVIDENCE_HAS_GOLD_BUT_PICKED_WRONG"

    # C: relation mismatch（evidence relation 与 relation_candidate 不一致，或 evidence 只有 related_to）
    rels = _evidence_relations(evidence)
    rc = normalize_answer(relation_candidate)
    if rels:
        all_related_to = all(r == "related_to" for r in rels)
        if all_related_to:
            return "C_RELATION_MISMATCH"
        if rc and any(r != rc for r in rels):
            return "C_RELATION_MISMATCH"
        if gold_relation and normalize_answer(gold_relation) and rc and normalize_answer(gold_relation) != rc:
            return "C_RELATION_MISMATCH"

    # D: head 选错（启发式）
    if _head_suspicious(head_used, gold_topic, question_text):
        return "D_HEAD_SELECTION_ERROR"

    # 兜底：无法解释的失败
    return "Z_OTHER"


def diagnose(
    preds_jsonl: Path,
    *,
    max_examples_per_label: int = 20,
    seed: int = 13,
    enable_oracle: bool = False,
    only_failures: bool = True,
) -> Dict[str, Any]:
    rng = random.Random(seed)

    total = 0
    total_fail = 0
    label_counter: Counter[str] = Counter()
    cases_by_label: Dict[str, List[FailureCase]] = defaultdict(list)
    relation_counter_in_fail: Counter[str] = Counter()

    for idx, rec in enumerate(_read_jsonl(preds_jsonl)):
        total += 1

        pred_answer = _extract_pred_answer_value(rec)
        gold_answer, gold_topic, gold_relation, gold_tail = _extract_gold_fields(rec)
        pred_topic = _extract_pred_topic(rec)
        relation_candidate = _extract_relation_candidate(rec)
        question = _extract_question(rec)
        head_used = _extract_head_used(rec)
        conv_id, turn_id = _extract_conv_turn_id(rec, idx)

        evidence = _evidence_list(rec)

        em = False
        if gold_answer:
            em = normalize_answer(pred_answer) == normalize_answer(gold_answer)

        if only_failures and em:
            continue

        if not em:
            total_fail += 1
            relation_counter_in_fail[relation_candidate.strip() or "__NONE__"] += 1

        label = _assign_label(
            em=bool(em),
            evidence=evidence,
            pred_topic=pred_topic,
            gold_topic=gold_topic,
            relation_candidate=relation_candidate,
            gold_relation=gold_relation,
            head_used=head_used,
            question_text=question,
            gold_answer=gold_answer,
            gold_tail=gold_tail,
        )

        oracle_flag: Optional[bool] = None
        if enable_oracle and (not em):
            oracle_flag = _oracle_answer_in_topic(rec, gold_topic=gold_topic, gold_answer=gold_answer)

        label_counter[label] += 1

        if label != "OK":
            # reservoir 风格：保证“看起来随机”，同时也可用于 Top\=前 N
            lst = cases_by_label[label]
            fc = FailureCase(
                conv_id=conv_id,
                turn_id=turn_id,
                question=question,
                gold_answer=gold_answer,
                pred_answer=pred_answer,
                gold_topic=gold_topic,
                pred_topic=pred_topic,
                gold_relation=gold_relation,
                relation_candidate=relation_candidate,
                head_used=head_used,
                evidence=evidence,
                label=label,
                oracle_answer_in_topic=oracle_flag,
            )
            if len(lst) < max_examples_per_label:
                lst.append(fc)
            else:
                j = rng.randint(0, total_fail if total_fail > 0 else 1)
                if j % (max_examples_per_label + 1) == 0:
                    lst[rng.randrange(max_examples_per_label)] = fc

    def _case_to_dict(c: FailureCase) -> Dict[str, Any]:
        return {
            "conv_id": c.conv_id,
            "turn_id": c.turn_id,
            "question": c.question,
            "gold_answer": c.gold_answer,
            "pred_answer": c.pred_answer,
            "gold_topic": c.gold_topic,
            "pred_topic": c.pred_topic,
            "gold_relation": c.gold_relation,
            "relation_candidate": c.relation_candidate,
            "head_used": c.head_used,
            "evidence": c.evidence,
            "oracle_answer_in_topic": c.oracle_answer_in_topic,
        }

    stats = {
        "preds_jsonl": str(preds_jsonl),
        "total": total,
        "total_fail": total_fail,
        "labels": {},
        "top_relation_candidate_in_fail": relation_counter_in_fail.most_common(30),
        "examples": {},
    }

    denom = total_fail if only_failures else max(total, 1)
    for label, cnt in label_counter.most_common():
        stats["labels"][label] = {
            "count": cnt,
            "rate": (cnt / max(denom, 1)),
        }
        if label != "OK":
            stats["examples"][label] = [_case_to_dict(x) for x in cases_by_label.get(label, [])]

    return stats


def _print_report(report: Dict[str, Any]) -> None:
    print("=== Diagnostics Summary ===")
    print(f"preds_jsonl: {report.get('preds_jsonl')}")
    print(f"total: {report.get('total')}")
    print(f"total_fail: {report.get('total_fail')}")

    print("\n=== Failure Labels ===")
    labels: Dict[str, Any] = report.get("labels", {})
    for k, v in labels.items():
        print(f"{k}: count={v.get('count')} rate={v.get('rate'):.4f}")

    print("\n=== Top relation_candidate in failures ===")
    for rc, cnt in report.get("top_relation_candidate_in_fail", []):
        print(f"{rc}: {cnt}")

    print("\n=== Examples (truncated) ===")
    ex: Dict[str, Any] = report.get("examples", {})
    for label, items in ex.items():
        print(f"\n[{label}] show={len(items)}")
        for i, it in enumerate(items[:5], start=1):
            conv_id = it.get("conv_id")
            turn_id = it.get("turn_id")
            q = it.get("question")
            gold = it.get("gold_answer")
            pred = it.get("pred_answer")
            gt = it.get("gold_topic")
            pt = it.get("pred_topic")
            rc = it.get("relation_candidate")
            oracle = it.get("oracle_answer_in_topic")
            print(f"{i}. conv_id={conv_id} turn_id={turn_id} rc={rc}")
            print(f"   q: {q}")
            print(f"   gold: {gold}")
            print(f"   pred: {pred}")
            print(f"   topic: pred={pt} gold={gt}")
            if oracle is not None:
                print(f"   oracle_answer_in_topic: {oracle}")
            ev = it.get("evidence", [])
            if isinstance(ev, list) and ev:
                e0 = ev[0]
                print(f"   ev0: head={e0.get('head','')} rel={e0.get('relation','')} tail={e0.get('tail','')}")
            else:
                print("   ev: []")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--preds_jsonl",
        type=str,
        required=False,
        default=r"D:\Project\ConvQA\outputs\preds.jsonl",
        help=r"推理输出 JSONL，例如 D:\Project\ConvQA\outputs\preds.jsonl",
    )
    ap.add_argument("--out_json", type=str, default="", help="可选：保存 diagnostics 报告到 json")
    ap.add_argument("--max_examples", type=int, default=20, help="每类最多保留多少条样例")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--enable_oracle", action="store_true", help="启用 Oracle 可答性检查（需要 topic_subgraph 字段）")
    ap.add_argument("--include_ok", action="store_true", help="包含 OK 样本（默认只看失败样本）")
    args = ap.parse_args()

    preds_path = Path(args.preds_jsonl)
    report = diagnose(
        preds_path,
        max_examples_per_label=int(args.max_examples),
        seed=int(args.seed),
        enable_oracle=bool(args.enable_oracle),
        only_failures=not bool(args.include_ok),
    )
    _print_report(report)

    if str(args.out_json).strip():
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

