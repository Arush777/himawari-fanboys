"""CLI for judge2: score, compare, probe, discover, and selftest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from judge2_lib import (
    ARTIFACT_DIR,
    PROMPT_VERSION,
    compare_pairs,
    compare_results,
    get_judge_frames,
    layer0_check,
    load_judges,
    read_json,
    response_cache_key,
    response_cache_path,
    run_selftest,
    score_caption_items,
    score_results,
    summarize_compare,
    validate_score_output,
    write_json,
)


def cmd_selftest(_: argparse.Namespace) -> int:
    failures = run_selftest()
    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    print("judge2 selftest passed")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    judges = load_judges(args.models)
    estimated = _estimate_score_calls(args.tasks, args.results, judges, args.no_cache)
    if not _budget_ok(args, estimated):
        return 2
    output = score_results(args.tasks, args.results, args.out, judges, no_cache=args.no_cache)
    if not validate_score_output(output, len(judges)):
        print("score output failed structure validation", file=sys.stderr)
        return 1
    print(f"judge2 score {output['summary']['score_normalized']:.4f} -> {args.out}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    judges = load_judges(args.models)
    estimated = _estimate_compare_calls(args.tasks, args.a, args.b, judges, args.no_cache)
    if not _budget_ok(args, estimated):
        return 2
    output = compare_results(args.tasks, args.a, args.b, args.out, judges, no_cache=args.no_cache)
    print(f"judge2 compare A win-rate {output['summary']['a_win_rate']:.4f} -> {args.out}")
    return 0


def _probe_tasks() -> dict:
    return {task["task_id"]: task for task in read_json("sample_input/tasks.json")}


def _probe_items(probes: list[dict], tasks: dict) -> list[dict]:
    items = []
    for probe in probes:
        task = tasks[probe["task_id"]]
        items.append({
            "item_id": f"{probe['task_id']}:{probe['probe_id']}:{probe['style']}",
            "task_id": probe["task_id"],
            "video_url": task["video_url"],
            "style": probe["style"],
            "caption": probe["caption"],
            "duplicate": False,
            "probe_id": probe["probe_id"],
            "expectation": probe["expectation"],
        })
    return items


def _cache_missing(judge, mode: str, video_url: str, style: str, caption: str,
                   caption_b: str = "") -> bool:
    key = response_cache_key(judge, mode, video_url, style, caption, caption_b)
    return not response_cache_path(key).exists()


def _estimate_score_calls(tasks_path: str, results_path: str, judges: list,
                          no_cache: bool) -> int:
    results = read_json(results_path)
    tasks = {task["task_id"]: task for task in read_json(tasks_path)}
    calls = 0
    for row in results:
        task = tasks[row["task_id"]]
        for style in task["styles"]:
            caption = str(row.get("captions", {}).get(style, "")).strip()
            for judge in judges:
                if no_cache or _cache_missing(judge, "absolute", task["video_url"], style, caption):
                    calls += 1
    return calls


def _estimate_compare_calls(tasks_path: str, a_path: str, b_path: str, judges: list,
                            no_cache: bool) -> int:
    a_results = {row["task_id"]: row for row in read_json(a_path)}
    b_results = {row["task_id"]: row for row in read_json(b_path)}
    tasks = {task["task_id"]: task for task in read_json(tasks_path)}
    calls = 0
    for task_id, row_a in a_results.items():
        task = tasks[task_id]
        row_b = b_results[task_id]
        for style in task["styles"]:
            original_a = str(row_a.get("captions", {}).get(style, "")).strip()
            original_b = str(row_b.get("captions", {}).get(style, "")).strip()
            for judge in judges:
                for order, cap_a, cap_b in (
                    ("ab", original_a, original_b),
                    ("ba", original_b, original_a),
                ):
                    mode = f"compare_{order}"
                    if no_cache or _cache_missing(judge, mode, task["video_url"], style, cap_a, cap_b):
                        calls += 1
    return calls


def _probe_pairs(probes: list[dict], tasks: dict) -> list[dict]:
    items = _probe_items(probes, tasks)
    by_key = {(row["task_id"], row["probe_id"]): row for row in items}
    pairs = []
    for task_id, task in sorted(tasks.items()):
        if (task_id, "P1") not in by_key:
            continue
        pairs.append({
            "task_id": task_id,
            "video_url": task["video_url"],
            "style": "formal",
            "caption_a": by_key[(task_id, "P1")]["caption"],
            "caption_b": by_key[(task_id, "P6")]["caption"],
        })
        pairs.append({
            "task_id": task_id,
            "video_url": task["video_url"],
            "style": "sarcastic",
            "caption_a": by_key[(task_id, "P7")]["caption"],
            "caption_b": by_key[(task_id, "P8")]["caption"],
        })
    return pairs


def _estimate_probe_calls(probes: list[dict], tasks: dict, judges: list, no_cache: bool,
                          skip_repeatability: bool, repeatability_task_id: str) -> int:
    calls = 0
    for item in _probe_items(probes, tasks):
        for judge in judges:
            if no_cache or _cache_missing(
                judge, "absolute", item["video_url"], item["style"], item["caption"],
            ):
                calls += 1
    for pair in _probe_pairs(probes, tasks):
        for judge in judges:
            for order, cap_a, cap_b in (
                ("ab", pair["caption_a"], pair["caption_b"]),
                ("ba", pair["caption_b"], pair["caption_a"]),
            ):
                mode = f"compare_{order}"
                if no_cache or _cache_missing(
                    judge, mode, pair["video_url"], pair["style"], cap_a, cap_b,
                ):
                    calls += 1
    if not skip_repeatability:
        subset = [probe for probe in probes if probe["task_id"] == repeatability_task_id]
        calls += len(subset) * len(judges)
    return calls


def _budget_ok(args: argparse.Namespace, estimated_live_calls: int) -> bool:
    limit = args.max_live_calls
    if limit is None:
        limit = int(__import__("os").environ.get("JUDGE2_MAX_LIVE_CALLS", "40"))
    if estimated_live_calls:
        print(f"estimated live judge calls: {estimated_live_calls}", file=sys.stderr)
    if estimated_live_calls > limit and not args.allow_expensive:
        print(
            f"refusing: estimated live judge calls ({estimated_live_calls}) exceed "
            f"limit ({limit}). Re-run with --allow-expensive or raise "
            "JUDGE2_MAX_LIVE_CALLS only after confirming budget.",
            file=sys.stderr,
        )
        return False
    return True


def _probe_pass(row: dict, by_clip_probe: dict[tuple[str, str], dict]) -> tuple[bool, str]:
    probe_id = row["probe_id"]
    ens = row["ensemble"]
    acc = ens["accuracy_0_10"]
    tone = ens["tone_0_10"]
    combined = ens["combined_0_10"]
    hard = row["layer0"]["hard_flags"]
    if probe_id == "P1":
        ok = 7 <= acc < 10
        return ok, "accuracy >= 7 and < 10"
    if probe_id == "P2":
        return acc <= 6, "accuracy <= 6"
    if probe_id == "P3":
        return acc <= 4, "accuracy <= 4"
    if probe_id == "P4":
        return tone <= 4, "tone <= 4"
    if probe_id == "P5":
        return ("meta_leak" in hard and combined <= 4), "meta_leak hard flag and capped score <= 4"
    if probe_id == "P6":
        p1 = by_clip_probe[(row["task_id"], "P1")]["ensemble"]["accuracy_0_10"]
        p3 = by_clip_probe[(row["task_id"], "P3")]["ensemble"]["accuracy_0_10"]
        return p3 < acc < p1, "accuracy strictly between P3 and P1"
    if probe_id == "P7":
        return tone >= 7, "tone >= 7"
    if probe_id == "P8":
        p7 = by_clip_probe[(row["task_id"], "P7")]["ensemble"]["tone_0_10"]
        return (tone <= 6 and tone < p7), "tone <= 6 and < P7"
    return False, "unknown probe"


def _probe_pair_rows(scored: list[dict], tasks: dict, judges, no_cache: bool) -> list[dict]:
    by_key = {(row["task_id"], row["probe_id"]): row for row in scored}
    pairs = []
    for task_id, task in sorted(tasks.items()):
        pairs.append({
            "pair_id": f"{task_id}:P1_vs_P6",
            "task_id": task_id,
            "video_url": task["video_url"],
            "style": "formal",
            "caption_a": by_key[(task_id, "P1")]["caption"],
            "caption_b": by_key[(task_id, "P6")]["caption"],
            "expected": "A",
        })
        pairs.append({
            "pair_id": f"{task_id}:P7_vs_P8",
            "task_id": task_id,
            "video_url": task["video_url"],
            "style": "sarcastic",
            "caption_a": by_key[(task_id, "P7")]["caption"],
            "caption_b": by_key[(task_id, "P8")]["caption"],
            "expected": "A",
        })
    return compare_pairs(pairs, judges, no_cache=no_cache)


def _write_probe_report(path: Path, scored: list[dict], pair_rows: list[dict],
                        repeatability: dict | None) -> bool:
    by_clip_probe = {(row["task_id"], row["probe_id"]): row for row in scored}
    lines = [
        "# judge2 Probe Report",
        "",
        f"Prompt version: `{PROMPT_VERSION}`",
        "",
        "## Iterations",
        "",
        "- Iteration 1 (`judge2-v1`): failed because mild sarcasm probes P8 were scored too generously on tone; one v3 P1 judge call also transiently failed and passed when rerun.",
        "- Iteration 2 (`judge2-v2`): tightened the tone rubric for hedged generic sarcasm phrases such as \"not the most exciting\", \"not that thrilling\", \"I guess\", and \"I suppose\".",
        "- Iteration 3 (`judge2-v3`): tightened the 10-point anchors so strong gold captions usually top out at 9 unless they are genuinely exhaustive and exceptional.",
        "- Addendum fix (`judge2-v4`): replaced v1/P6 with a one-attribute red/yellow swap, added 0-2 and 3-4 accuracy anchors, downgraded `footage` to a soft Layer-0 flag, and changed P1 acceptance to accuracy >= 7 and < 10.",
        "- Addendum iteration 1 (`judge2-v5`): clarified that tone is scored independently from factual accuracy and that pairwise captions with contradicted visible attributes must lose rather than tie.",
        "",
        "## Absolute Probe Outcomes",
        "",
        "| Clip | Probe | Style | Hard flags | Accuracy | Tone | Combined | Required outcome | Verdict |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    all_pass = True
    for row in sorted(scored, key=lambda r: (r["task_id"], r["probe_id"])):
        ok, required = _probe_pass(row, by_clip_probe)
        all_pass = all_pass and ok
        lines.append(
            f"| `{row['task_id']}` | `{row['probe_id']}` | `{row['style']}` | "
            f"`{','.join(row['layer0']['hard_flags']) or '-'}` | "
            f"{row['ensemble']['accuracy_0_10']:.1f} | {row['ensemble']['tone_0_10']:.1f} | "
            f"{row['ensemble']['combined_0_10']:.1f} | {required} | {'PASS' if ok else 'FAIL'} |"
        )
    lines.extend(["", "## Pairwise Exam", ""])
    p1_rows = [row for row in pair_rows if "P1_vs_P6" in row["pair_id"]]
    p7_rows = [row for row in pair_rows if "P7_vs_P8" in row["pair_id"]]
    p1_wins = sum(1 for row in p1_rows if row["winner"] == "A")
    p7_wins = sum(1 for row in p7_rows if row["winner"] == "A")
    p1_losses = sum(1 for row in p1_rows if row["winner"] == "B")
    p7_losses = sum(1 for row in p7_rows if row["winner"] == "B")
    pair_pass = p1_wins >= 2 and p7_wins >= 2 and p1_losses == 0 and p7_losses == 0
    all_pass = all_pass and pair_pass
    lines.append("| Pair | Winner | Expected | Verdict |")
    lines.append("| --- | --- | --- | --- |")
    for row in sorted(pair_rows, key=lambda r: r["pair_id"]):
        ok = row["winner"] == "A"
        lines.append(f"| `{row['pair_id']}` | `{row['winner']}` | `A` | {'PASS' if ok else 'FAIL'} |")
    lines.append("")
    lines.append(f"Pairwise aggregate: {'PASS' if pair_pass else 'FAIL'} "
                 f"(P1 wins {p1_wins}/3, P7 wins {p7_wins}/3, losses {p1_losses + p7_losses}).")
    if repeatability:
        lines.extend(["", "## Repeatability", ""])
        lines.append(f"Verdict: {'PASS' if repeatability['pass'] else 'FAIL'}")
        lines.append(f"Max median shift: {repeatability['max_shift']}")
        if repeatability["changed"]:
            lines.append("Changed outcomes: " + ", ".join(repeatability["changed"]))
        if repeatability["pair_changed"]:
            lines.append("Changed pairwise verdicts: " + ", ".join(repeatability["pair_changed"]))
        all_pass = all_pass and repeatability["pass"]
    lines.extend(["", "## Overall", "", "PASS" if all_pass else "FAIL", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return all_pass


def _probe_outcome_map(scored: list[dict]) -> dict[str, tuple[bool, float]]:
    by_clip_probe = {(row["task_id"], row["probe_id"]): row for row in scored}
    out = {}
    for row in scored:
        ok, _ = _probe_pass(row, by_clip_probe)
        out[row["item_id"]] = (ok, row["ensemble"]["combined_0_10"])
    return out


def _repeatability(probes: list[dict], tasks: dict, judges,
                   first: list[dict] | None = None) -> dict:
    if first is None:
        first = score_caption_items(_probe_items(probes, tasks), judges, no_cache=False)
    second = score_caption_items(_probe_items(probes, tasks), judges, no_cache=True)
    a = _probe_outcome_map(first)
    b = _probe_outcome_map(second)
    max_shift = max(abs(a[key][1] - b[key][1]) for key in a)
    changed = [key for key in a if a[key][0] != b[key][0]]
    return {
        "pass": max_shift <= 1 and not changed,
        "max_shift": max_shift,
        "changed": changed,
        "pair_changed": [],
    }


def cmd_probe(args: argparse.Namespace) -> int:
    judges = load_judges(args.models)
    probes = read_json(args.probes)
    tasks = _probe_tasks()
    estimated = _estimate_probe_calls(
        probes, tasks, judges, args.no_cache, args.skip_repeatability,
        args.repeatability_task_id,
    )
    if not _budget_ok(args, estimated):
        return 2
    scored = score_caption_items(_probe_items(probes, tasks), judges, no_cache=args.no_cache)
    pair_rows = _probe_pair_rows(scored, tasks, judges, no_cache=args.no_cache)
    report_path = Path(args.out)
    first_pass = _write_probe_report(report_path, scored, pair_rows, repeatability=None)
    repeatability = None
    if first_pass and not args.skip_repeatability:
        subset_probes = [probe for probe in probes if probe["task_id"] == args.repeatability_task_id]
        subset_tasks = {args.repeatability_task_id: tasks[args.repeatability_task_id]}
        first_subset = [
            row for row in scored
            if row["task_id"] == args.repeatability_task_id
        ]
        repeatability = _repeatability(
            subset_probes, subset_tasks, judges,
            first=first_subset,
        )
    final_pass = _write_probe_report(report_path, scored, pair_rows, repeatability=repeatability)
    print(f"judge2 probe {'PASS' if final_pass else 'FAIL'} -> {report_path}")
    return 0 if final_pass else 1


def cmd_discover(_: argparse.Namespace) -> int:
    # Discovery is intentionally scripted in Phase J0 to avoid exposing keys in logs.
    print("Discovery artifacts are in /Users/arush/AMD-hackathon/codex-run-artifacts/judge2/")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="run Layer-0 selftests")
    parser.add_argument("--models", default="judge2_models.json")
    sub = parser.add_subparsers(dest="command")

    score = sub.add_parser("score")
    score.add_argument("--tasks", required=True)
    score.add_argument("--results", required=True)
    score.add_argument("--out", required=True)
    score.add_argument("--no-cache", action="store_true")
    score.add_argument("--max-live-calls", type=int)
    score.add_argument("--allow-expensive", action="store_true")
    score.set_defaults(func=cmd_score)

    compare = sub.add_parser("compare")
    compare.add_argument("--tasks", required=True)
    compare.add_argument("--a", required=True)
    compare.add_argument("--b", required=True)
    compare.add_argument("--out", required=True)
    compare.add_argument("--no-cache", action="store_true")
    compare.add_argument("--max-live-calls", type=int)
    compare.add_argument("--allow-expensive", action="store_true")
    compare.set_defaults(func=cmd_compare)

    probe = sub.add_parser("probe")
    probe.add_argument("--probes", default="probes/probes.json")
    probe.add_argument("--out", default=str(ARTIFACT_DIR / "probe_report.md"))
    probe.add_argument("--no-cache", action="store_true")
    probe.add_argument("--skip-repeatability", action="store_true")
    probe.add_argument("--repeatability-task-id", default="v1")
    probe.add_argument("--max-live-calls", type=int)
    probe.add_argument("--allow-expensive", action="store_true")
    probe.set_defaults(func=cmd_probe)

    discover = sub.add_parser("discover")
    discover.set_defaults(func=cmd_discover)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.selftest:
        return cmd_selftest(args)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
