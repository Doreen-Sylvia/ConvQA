# coding: utf-8
"""
推理入口脚本（main）：全链路跑通，并把每轮推理结果落盘为 JSONL（供 eval/diagnostics 读取）

修复点：
1) JSONL 写入 gold 字段：gold_answer_text / gold_topic / gold_relation
2) Segmenter 调用：优先传 turns list（Segmenter 常见签名），避免 no_segments
3) 输出 pred_answer_value：用于 EM（从 top-1 evidence.tail 提取）
4) KG 执行加入时间因果过滤：不允许使用未来 turn 的证据（<= current_turn 可配置）
5) turn id 统一：优先用数据字段 turn（通常从 0 开始），否则 fallback idx
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.inference.router import Router
from src.inference.gating import Gating
from src.inference.memory_kg import MemoryKG
from src.inference.kg_execute import KGExecutor, EvidenceTriple
from src.inference.verbalizer import Verbalizer


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _norm(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip()


def _safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _coerce_dialogues(root: Any) -> List[Dict[str, Any]]:
    if isinstance(root, list):
        return [x for x in root if isinstance(x, dict)]
    if isinstance(root, dict):
        for k in ("dialogues", "conversations", "data"):
            v = root.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        if any(key in root for key in ("conv_id", "questions", "turns")):
            return [root]
    return []


def _get_conv_id(conv: Dict[str, Any], fallback_idx: int) -> str:
    for k in ("conv_id", "conversation_id", "id", "convId"):
        v = _norm(conv.get(k))
        if v:
            return v
    return str(fallback_idx)


def _get_turns(conv: Dict[str, Any]) -> List[Dict[str, Any]]:
    for k in ("questions", "turns"):
        v = conv.get(k)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
    return []


def _get_question_text(turn: Dict[str, Any]) -> str:
    for k in ("question", "question_text", "q", "q_t", "text"):
        s = _norm(turn.get(k))
        if s:
            return s
    return ""


def _get_original_question_text(turn: Dict[str, Any]) -> str:
    # 你的 merged 数据里通常是 original_question
    for k in ("original_question", "question_original", "orig_question"):
        s = _norm(turn.get(k))
        if s:
            return s
    return _get_question_text(turn)


def _turn_id(turn: Dict[str, Any], fallback_idx0: int) -> int:
    """
    turn id 优先取数据字段 turn（你的 merged 样例是从 0 开始）。
    如果缺失，用 fallback_idx0（也是从 0 开始）保证与“在线因果过滤”一致。
    """
    for k in ("turn", "turn_id", "tid"):
        v = _safe_int(turn.get(k))
        if v is not None:
            return v
    return int(fallback_idx0)


def _as_serializable_evidence(ev: EvidenceTriple) -> Dict[str, Any]:
    if is_dataclass(ev):
        return asdict(ev)
    return {
        "head": getattr(ev, "head", ""),
        "relation": getattr(ev, "relation", ""),
        "tail": getattr(ev, "tail", ""),
        "turn_id": getattr(ev, "turn_id", None),
        "scope": getattr(ev, "scope", ""),
    }


def _predict_relation_from_question(q: str) -> str:
    """
    gold_relation 生成：用与你 extractor/预测端一致的轻量规则。
    先保证评测口径一致；后续你可以替换成共享函数或更完整的规则。
    """
    t = q.strip().lower()
    # order matters
    if "how many" in t or "amount" in t or "number of books" in t:
        return "num_books"
    if "first" in t and "book" in t:
        return "first_book"
    if ("final" in t or "ended" in t or "concluded" in t or "last book" in t) and "book" in t:
        return "final_book"
    if "author" in t or "writer" in t or "wrote" in t:
        return "author"
    if "nationality" in t or "country" in t:
        return "nationality"
    if "publisher" in t:
        return "publisher"
    if "genre" in t:
        return "genre"
    if "award" in t or "prize" in t:
        # when? + award -> award_year (如果你 KG 里有此关���)
        if "when" in t or "year" in t:
            return "award_year"
        return "award"
    if "year" in t or "when" in t:
        return "publication_year"
    if "title" in t:
        return "book_title"
    return "related_to"


def _extract_pred_answer_value(
    *,
    pred_answer_text: str,
    evidence: List[EvidenceTriple],
) -> str:
    """
    为 EM 评测提供一个“答案值”字段：
    - 优先用 top-1 evidence.tail（最可靠，且与 gold_answer_text 同空间）
    - 否��尝试从 pred_answer_text 里抽取（兜底）
    """
    if evidence:
        ev0 = evidence[0]
        tail = getattr(ev0, "tail", "")
        if tail:
            return str(tail)

    # fallback: 简单从句子里抽取最后一个“词/数字”，不保证完美
    s = _norm(pred_answer_text)
    if not s:
        return ""
    # 去掉中文句号
    s = s.rstrip("。.")
    # 常见模板 “X 的 Y 是 Z”
    m = re.search(r"是\s*([^\s，,]+)\s*$", s)
    if m:
        return m.group(1)
    # 否则取最后一个 token
    parts = re.split(r"\s+", s)
    return parts[-1] if parts else ""


# 1) 修改：_try_segment() 增加 memory_tsv_path 参数，并传给 Segmenter
def _try_segment(
    turns: List[Dict[str, Any]],
    conv: Dict[str, Any],
    *,
    memory_tsv_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    让 Segmenter 读取 memory_triples.tsv，从而附加 key_entities/key_relations。
    """
    try:
        from src.inference.segmenter import Segmenter  # type: ignore
    except Exception:
        return []

    try:
        segger = Segmenter(memory_tsv_path=memory_tsv_path)  # <-- 新增：传 memory_tsv_path
        segs = None

        try:
            segs = segger.segment(turns)  # type: ignore
        except Exception:
            segs = None

        if segs is None:
            try:
                segs = segger.segment(conv)  # type: ignore
            except Exception:
                segs = None

        if isinstance(segs, list):
            return [s for s in segs if isinstance(s, dict)]
        return []
    except Exception:
        return []


class ResultWriter:
    """
    以 JSONL 方式落盘：每次 write_one 追加写一行 JSON。
    """

    def __init__(self, output_path: Path, *, append: bool = True) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._mode = "a" if append else "w"
        self._fp = self.output_path.open(self._mode, encoding="utf-8", newline="\n")

    def write_one(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        self._fp.write(line + "\n")
        self._fp.flush()

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass

    def __enter__(self) -> "ResultWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def run_inference(
    *,
    merged_json_path: Path,
    memory_tsv_path: Path,
    out_path: Optional[Path],
    output_jsonl_path: Optional[Path],
    top_k: int,
    recent_n_turns: int,
    print_stdout: bool,
    allow_current_turn_evidence: bool,
) -> Dict[str, Any]:
    root = _read_json(merged_json_path)
    dialogues = _coerce_dialogues(root)

    kg = MemoryKG(memory_tsv_path, build_entity_vocab=True)
    kg.load()

    router = Router(allow_new=False)
    gating = Gating(recent_n_turns=recent_n_turns)
    executor = KGExecutor(kg)
    verbalizer = Verbalizer()

    results: Dict[str, Any] = {
        "merged_json": str(merged_json_path),
        "memory_tsv": str(memory_tsv_path),
        "num_dialogues": len(dialogues),
        "conversations": [],
    }

    known_entities = list(kg.iter_entities())

    writer_cm = ResultWriter(output_jsonl_path) if output_jsonl_path is not None else None
    writer = writer_cm.__enter__() if writer_cm is not None else None

    try:
        for i, conv in enumerate(dialogues):
            conv_id = _get_conv_id(conv, i)
            turns_all = _get_turns(conv)

            segments = _try_segment(
                turns_all,
                conv,
                memory_tsv_path=memory_tsv_path,  # <-- 新增：把 memory_tsv_path 传进去
            )

            conv_out: Dict[str, Any] = {
                "conv_id": conv_id,
                "num_turns": len(turns_all),
                "turns": [],
            }

            history_turns: List[Dict[str, Any]] = []

            for idx, turn in enumerate(turns_all):
                current_turn_id = _turn_id(turn, idx)  # 0-based
                q_t = _get_question_text(turn)

                # gold fields（仅用于评测/诊断，不参与推理决策）
                gold_answer_text = _norm(turn.get("answer_text"))
                gold_topic = _norm(turn.get("topic"))
                gold_relation = _predict_relation_from_question(_get_original_question_text(turn))

                rr = router.route(q_t, segments)
                assigned_seg: Optional[Dict[str, Any]] = None
                if rr.assigned_seg_id is not None:
                    for s in segments:
                        if _safe_int(s.get("seg_id")) == int(rr.assigned_seg_id):
                            assigned_seg = s
                            break

                topic = _norm(rr.assigned_topic)
                if not topic and assigned_seg is not None:
                    topic = _norm(assigned_seg.get("topic"))

                gr = gating.gate(
                    q_t=q_t,
                    assigned_seg=assigned_seg,
                    history_turns=history_turns,
                    memory_store=kg,
                    conv_id=conv_id,
                    known_entities=known_entities,
                )

                # KG 执行：加入时间因果过滤
                evidence = executor.execute(
                    conv_id=conv_id,
                    topic=topic,
                    head_candidates=gr.head_candidates,
                    relation_candidate=gr.relation_candidate,
                    top_k=top_k,
                    current_turn=current_turn_id,
                    allow_current_turn=allow_current_turn_evidence,
                )

                vr = verbalizer.verbalize(q_t, evidence)

                pred_answer_value = _extract_pred_answer_value(
                    pred_answer_text=vr.answer_text,
                    evidence=evidence,
                )

                turn_out = {
                    "turn": current_turn_id,
                    "question": q_t,
                    "gold": {
                        "answer_text": gold_answer_text,
                        "topic": gold_topic,
                        "relation": gold_relation,
                    },
                    "router": {
                        "assigned_seg_id": rr.assigned_seg_id,
                        "assigned_topic": rr.assigned_topic,
                        "route_reason": rr.route_reason,
                    },
                    "gating": {
                        "use_history": gr.use_history,
                        "history_turn_ids": gr.history_turn_ids,
                        "head_candidates": gr.head_candidates,
                        "relation_candidate": gr.relation_candidate,
                        "gate_reason": getattr(gr, "gate_reason", ""),
                    },
                    "evidence": [_as_serializable_evidence(ev) for ev in evidence],
                    "answer_text": vr.answer_text,
                    "pred_answer_value": pred_answer_value,
                    "evidence_lines": vr.evidence_lines,
                }
                conv_out["turns"].append(turn_out)

                # JSONL 记录（eval/diagnostics 用）
                if writer is not None:
                    jsonl_record = {
                        "conv_id": conv_id,
                        "turn": current_turn_id,
                        "question": q_t,
                        # pred
                        "pred_answer_text": vr.answer_text,
                        "pred_answer_value": pred_answer_value,
                        "pred_evidence": [_as_serializable_evidence(ev) for ev in evidence],
                        "router": turn_out["router"],
                        "gating": turn_out["gating"],
                        # gold (for eval only)
                        "gold_answer_text": gold_answer_text,
                        "gold_topic": gold_topic,
                        "gold_relation": gold_relation,
                        # meta
                        "meta": {
                            "merged_json": str(merged_json_path),
                            "memory_tsv": str(memory_tsv_path),
                            "topic": topic,
                            "top_k": top_k,
                            "recent_n_turns": recent_n_turns,
                            "allow_current_turn_evidence": allow_current_turn_evidence,
                        },
                    }
                    writer.write_one(jsonl_record)

                if print_stdout:
                    print(f"[conv={conv_id} turn={current_turn_id}] Q: {q_t}")
                    print(f"  -> A: {vr.answer_text}")
                    if vr.evidence_lines:
                        for line in vr.evidence_lines:
                            print(f"     {line}")

                # 模拟在线：历史只包含 < current_turn 的 turns（我们按顺序遍历，所以 append 即可）
                history_turns.append(turn)

            results["conversations"].append(conv_out)

    finally:
        if writer_cm is not None:
            writer_cm.__exit__(None, None, None)

    if out_path is not None:
        _write_json(out_path, results)

    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--merged_json",
        type=str,
        required=False,
        default=r"D:\Project\ConvQA\data\merged_dialogues\comprehensive_merged_dialogues.json",
        help="merged 对话 JSON 路径（可选；不传则用 default）",
    )
    ap.add_argument(
        "--memory_tsv",
        type=str,
        required=False,
        default=r"D:\Project\ConvQA\data\preprocessed\dialogue_kg\memory_triples.tsv",
        help="memory_triples.tsv 路径（可选；不传则用 default）",
    )
    ap.add_argument("--out_json", type=str, default="")
    ap.add_argument(
        "--output_jsonl",
        type=str,
        default=r"D:\Project\ConvQA\outputs\preds.jsonl",
        help=r"逐 turn 写入 JSONL；默认 D:\Project\ConvQA\outputs\preds.jsonl",
    )
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--recent_n_turns", type=int, default=3)
    ap.add_argument("--no_print", action="store_true")
    ap.add_argument(
        "--allow_current_turn_evidence",
        action="store_true",
        help="若指定：允许使用 turn_id == 当前 turn 的证据（离线评测更容易）。不指定则严格只用 < 当前 turn。",
    )

    args = ap.parse_args()

    merged_json_path = Path(args.merged_json)
    memory_tsv_path = Path(args.memory_tsv)
    out_path = Path(args.out_json) if _norm(args.out_json) else None
    output_jsonl_path = Path(args.output_jsonl) if _norm(args.output_jsonl) else None

    if output_jsonl_path is not None:
        output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        # 覆盖写，避免追加到旧文件
        output_jsonl_path.write_text("", encoding="utf-8")

    print(f"[main_infer] merged_json={merged_json_path}")
    print(f"[main_infer] memory_tsv={memory_tsv_path}")
    print(f"[main_infer] output_jsonl={output_jsonl_path}")

    run_inference(
        merged_json_path=merged_json_path,
        memory_tsv_path=memory_tsv_path,
        out_path=out_path,
        output_jsonl_path=output_jsonl_path,
        top_k=int(args.top_k),
        recent_n_turns=int(args.recent_n_turns),
        print_stdout=not bool(args.no_print),
        allow_current_turn_evidence=bool(args.allow_current_turn_evidence),
    )


if __name__ == "__main__":
    main()