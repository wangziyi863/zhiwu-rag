"""Direct four-arm comparison on one frozen candidate set.

The quantitative task is exact ABO product identity.  General visual
similarity has no automatic ground truth here, so this module exports a
blinded review sheet instead of silently treating every different product
ID as irrelevant.
"""
import argparse
import csv
import json
import random
from pathlib import Path

from .posttrain_data import read_json, write_json, verify_manifest
from .posttrain_followup import far_threshold, load_arm, paired_counts
from .ranking import rank_candidates


def _ratio(n, d):
    return n / d if d else None


def summarize_rows(rows):
    closed = [r for r in rows if r["scenario"] == "closed"]
    opened = [r for r in rows if r["scenario"] == "open"]
    recalled = [r for r in closed if r["has_positive"]]
    return {
        "closed_queries": len(closed),
        "no_match_queries": len(opened),
        "recall_at_k": _ratio(len(recalled), len(closed)),
        "conditional_top1": _ratio(sum(bool(r["top_correct"]) for r in recalled), len(recalled)),
        "retrieval_top1": _ratio(sum(bool(r["top_correct"]) for r in closed), len(closed)),
        "accepted_correct_rate": _ratio(
            sum(bool(r["top_correct"]) and bool(r.get("accepted")) for r in closed), len(closed)
        ),
        "false_accept_rate": _ratio(sum(bool(r.get("accepted")) for r in opened), len(opened)),
    }


def validate_shared_cases(cases, rows):
    truth = {c["case_id"]: c for c in cases}
    if len(rows) != len(truth) or {r["case_id"] for r in rows} != set(truth):
        raise ValueError("case IDs differ across arms")
    for row in rows:
        case = truth[row["case_id"]]
        expected = [c["identity"] for c in case["candidates"]]
        supplied = row.get("candidate_identities")
        if supplied is not None and supplied != expected:
            raise ValueError(f"candidate order differs for {row['case_id']}")
        if row.get("candidate_count", len(expected)) != len(expected):
            raise ValueError(f"candidate count differs for {row['case_id']}")
        if row["scenario"] != case["scenario"]:
            raise ValueError(f"scenario differs for {row['case_id']}")


def rank_web_case(case, assessments):
    hits = []
    values = []
    for candidate in case["candidates"]:
        hit = dict(candidate)
        hit.setdefault("similarity", 0.0)
        hits.append(hit)
        values.append(assessments[candidate["identity"]])
    ranked = rank_candidates(hits, values, 1)
    top = ranked[0] if ranked else None
    top_identity = top.get("identity") if top else None
    return {
        "case_id": case["case_id"],
        "query_id": case["query_id"],
        "scenario": case["scenario"],
        "has_positive": case["has_positive"],
        "top_correct": bool(top_identity and top_identity == case["identity"]),
        "score": top["rerank"]["score"] if top else None,
        "top_identity": top_identity,
        "accepted": top is not None,
        "candidate_count": len(hits),
        "candidate_identities": [h["identity"] for h in hits],
    }


def _lookup_candidate(case, identity):
    return next((c for c in case["candidates"] if c["identity"] == identity), None)


def build_blind_review(cases, arms, seed=20261009, limit=60):
    rng = random.Random(seed)
    closed = [c for c in cases if c["scenario"] == "closed"]
    by_arm = {name: {r["case_id"]: r for r in rows} for name, rows in arms.items()}
    chosen = list(closed)
    rng.shuffle(chosen)
    chosen = chosen[:limit]
    review, key = [], []
    for case in chosen:
        proposals = {}
        for arm, rows in by_arm.items():
            identity = rows[case["case_id"]].get("top_identity")
            if identity:
                proposals.setdefault(identity, []).append(arm)
        identities = list(proposals)
        rng.shuffle(identities)
        for n, identity in enumerate(identities, 1):
            candidate = _lookup_candidate(case, identity)
            if not candidate:
                raise ValueError(f"Top identity {identity} is outside frozen candidates")
            code = f"C{n}"
            review.append({
                "query_id": case["query_id"],
                "query_image": case["image_path"],
                "candidate_code": code,
                "candidate_image": candidate["image_path"],
                "similarity_label": "",
                "review_note": "",
            })
            key.append({
                "query_id": case["query_id"],
                "candidate_code": code,
                "candidate_identity": identity,
                "same_product": identity == case["identity"],
                "proposed_by": sorted(proposals[identity]),
            })
    return review, key


def _save_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _web_pair(request, scratch):
    # Calling the production reranker with one candidate preserves its exact
    # prompt, parser and acceptance rule while making each pair resumable.
    from .vlm import rerank
    result = rerank(request, scratch)
    audit = result["audit"][0]
    stats = result.get("stats", {})
    return {
        "assessment": audit["assessment"],
        "raw": audit["raw"],
        "seconds": stats.get("generation_seconds", 0.0),
        "peak_mib": stats.get("peak_allocated_mib"),
    }


def score_web_split(cases, output, model):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scratch = output / "scratch"
    scratch.mkdir(exist_ok=True)
    cache_path = output / "pairs.json"
    cache = read_json(cache_path) if cache_path.exists() else {}
    rows = []
    for number, case in enumerate(cases, 1):
        assessments = {}
        seconds = 0.0
        peaks = []
        for hit in case["candidates"]:
            key = f"{case['query_id']}|{hit['identity']}"
            if key not in cache:
                request = {
                    "task": "rerank",
                    "model": model,
                    "crop": case["image_path"],
                    "question": "Find the same product design.",
                    "hits": [hit],
                    "top_k": 1,
                }
                cache[key] = _web_pair(request, scratch)
                write_json(cache_path, cache)
            pair = cache[key]
            assessments[hit["identity"]] = pair["assessment"]
            seconds += float(pair.get("seconds") or 0.0)
            if pair.get("peak_mib") is not None:
                peaks.append(float(pair["peak_mib"]))
        row = rank_web_case(case, assessments)
        row["pair_seconds_sum"] = seconds
        row["peak_mib"] = max(peaks) if peaks else None
        rows.append(row)
        write_json(output / "rows.partial.json", rows)
        if number % 10 == 0 or number == len(cases):
            print(f"web_original {number}/{len(cases)} cases; {len(cache)} unique pairs", flush=True)
    write_json(output / "rows.json", rows)
    write_json(output / "complete.json", {"cases": len(rows), "pairs": len(cache), "model": model})
    return rows


def _apply_threshold(rows, threshold):
    result = []
    for row in rows:
        item = dict(row)
        item["accepted"] = item.get("score") is not None and item["score"] >= threshold
        result.append(item)
    return result


def _report_table(rows, keys):
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        vals = []
        for key in keys:
            value = row[key]
            vals.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def analyze(args):
    data = Path(args.data)
    root = Path(args.root)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    verify_manifest(data)
    cases = {split: read_json(data / f"{split}.json") for split in ("val", "test")}

    web = {}
    for split in ("val", "test"):
        web[split] = score_web_split(cases[split], output / "web_original" / split, args.model)
        validate_shared_cases(cases[split], web[split])

    arms = {
        "untrained_yes_no": load_arm(root / "baseline", data),
        "continue_sft": load_arm(Path(args.control) / "continue_sft", data),
        "dpo": load_arm(root / "dpo", data),
    }
    scored = {"web_original": web}
    thresholds = {"web_original": far_threshold(web["val"], args.validation_far)}
    for name, arm in arms.items():
        validate_shared_cases(cases["val"], arm["val"])
        validate_shared_cases(cases["test"], arm["test"])
        thresholds[name] = far_threshold(arm["val"], args.validation_far)
        scored[name] = arm

    summaries = []
    test_rows = {}
    for name, arm in scored.items():
        val = _apply_threshold(arm["val"], thresholds[name])
        test = _apply_threshold(arm["test"], thresholds[name])
        test_rows[name] = test
        val_metrics = summarize_rows(val)
        metrics = summarize_rows(test)
        summaries.append({
            "arm": name,
            "k": max(r["candidate_count"] for r in test),
            "threshold": thresholds[name],
            **metrics,
            "validation_false_accept_rate": val_metrics["false_accept_rate"],
            "mean_pair_seconds_sum": sum(float(r.get("pair_seconds_sum") or 0) for r in test) / len(test),
            "peak_mib": max((float(r["peak_mib"]) for r in test if r.get("peak_mib") is not None), default=None),
        })

    comparisons = [
        ("web_original", "untrained_yes_no"),
        ("web_original", "continue_sft"),
        ("web_original", "dpo"),
        ("untrained_yes_no", "continue_sft"),
        ("continue_sft", "dpo"),
    ]
    paired = [
        {"before": left, "after": right, **paired_counts(test_rows[left], test_rows[right])}
        for left, right in comparisons
    ]

    review, key = build_blind_review(cases["test"], test_rows, args.seed, args.review_queries)
    _save_csv(output / "summary.csv", summaries)
    _save_csv(output / "paired_top1.csv", paired)
    _save_csv(output / "similarity_blind_review.csv", review)
    write_json(output / "similarity_blind_key.json", key)
    write_json(output / "analysis.json", {
        "task": "ABO exact same-product ranking; product identity used only for evaluation",
        "similarity_status": "pending human relevance judgments",
        "validation_far_target": args.validation_far,
        "thresholds": thresholds,
        "summary": summaries,
        "paired_top1": paired,
    })
    keys = ["arm", "k", "recall_at_k", "conditional_top1", "retrieval_top1",
            "accepted_correct_rate", "false_accept_rate", "validation_false_accept_rate",
            "mean_pair_seconds_sum", "peak_mib"]
    report = (
        "# 网页原评分与后训练重排直接比较\n\n"
        "四组使用同一冻结查询与同一 K=12 候选。商品 ID 只用于计算同款命中率，不输入模型。"
        "拒绝阈值只在验证集上按目标 FAR 选择。网页原评分保留生产规则，并能读取候选文本资料；"
        "三个 Yes/No 版本只比较两张图片，因此该实验比较的是完整重排版本，不是纯模型能力。\n\n"
        + _report_table(summaries, keys)
        + "\n\n## 同一查询上的 Top-1 得失\n\n"
        + _report_table(paired, ["before", "after", "gained", "lost", "both_correct", "both_wrong", "mcnemar_exact_p"])
        + "\n\n## 相似商品任务\n\n"
        "不同商品 ID 不代表视觉上不相关。`similarity_blind_review.csv` 已隐藏模型来源并去重候选，"
        "需要人工填写 `similarity_label` 后才能计算相似商品相关性；模型来源密钥单独保存在"
        " `similarity_blind_key.json`。\n"
    )
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--validation-far", type=float, default=0.10)
    parser.add_argument("--review-queries", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20261009)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
