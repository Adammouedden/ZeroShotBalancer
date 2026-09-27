"""Readable summaries and conservative transition brackets, including joint grids."""
import csv
import html
import itertools
import json
from pathlib import Path
from model_adapter import write_json


def grouped(rows):
    result = {}
    for row in rows:
        result.setdefault(row["case_id"], []).append(row)
    return result


def outcome(rows, seeds):
    if sorted(r["seed"] for r in rows) != sorted(seeds):
        return None
    return (sum(bool(r["balance"]) for r in rows), sum(r["numerical"] == "valid_observed" for r in rows),
            sum(bool(r["review_flags"]) for r in rows))


def transitions(rows, manifest, config):
    groups = grouped(rows)
    cases = [c for c in manifest["cases"] if outcome(groups.get(c["id"], []), config["seeds"]) is not None]
    directions = config["directions"] if config["property"] == "center_of_mass" else [config["property"]]
    brackets = []
    for direction in directions:
        others = [d for d in directions if d != direction]
        slices = {}
        for case in cases:
            key = tuple((d, round(case["values"][d], 13)) for d in others)
            slices.setdefault(key, []).append(case)
        for fixed, line in slices.items():
            ordered = sorted(line, key=lambda c: c["values"][direction])
            for a, b in zip(ordered, ordered[1:]):
                oa = outcome(groups[a["id"]], config["seeds"])
                ob = outcome(groups[b["id"]], config["seeds"])
                if oa != ob:
                    brackets.append({"direction": direction, "fixed": dict(fixed), "low": a["values"][direction],
                        "high": b["values"][direction], "low_outcome": oa, "high_outcome": ob,
                        "interpretation": "observed endpoint transition, not a proven unique threshold"})
    return brackets


def refinement_candidates(rows, manifest, config):
    existing = {tuple((k, round(v, 13)) for k, v in sorted(c["values"].items())) for c in manifest["cases"]}
    candidates = []
    for b in transitions(rows, manifest, config):
        tolerance = config.get("refine_to_mm", .5) if config["property"] == "center_of_mass" else config.get("refine_to", .01)
        if isinstance(tolerance, dict):
            tolerance = tolerance.get(b["direction"], .5)
        if config["property"] == "center_of_mass":
            tolerance /= 1000
        if b["high"] - b["low"] <= tolerance:
            continue
        values = {**b["fixed"], b["direction"]: (b["high"] + b["low"]) / 2}
        key = tuple((k, round(v, 13)) for k, v in sorted(values.items()))
        if key not in existing:
            existing.add(key)
            candidates.append((b["high"] - b["low"], values))
    return [v for _, v in sorted(candidates, key=lambda item: -item[0])]


def grid_svg(path, manifest, groups, seeds):
    cases = [c for c in manifest["cases"] if c["kind"] == "mutation"]
    directions = manifest["config"]["directions"]
    if len(directions) != 2:
        return
    xname, yname = directions
    xs = sorted({c["values"][xname] for c in cases})
    ys = sorted({c["values"][yname] for c in cases}, reverse=True)
    cell, left, top = 42, 110, 70
    width, height = left + cell * len(xs) + 60, top + cell * len(ys) + 110
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>', '<g font-family="sans-serif" font-size="12">',
           '<text x="15" y="24">Frozen PPO balance successes / seeds (not physical validation)</text>',
           f'<text x="{left}" y="{height-50}">{html.escape(xname)} offset (mm)</text>',
           f'<text x="10" y="48">{html.escape(yname)} (mm)</text>']
    for i, value in enumerate(xs):
        svg.append(f'<text x="{left+i*cell+3}" y="{top+len(ys)*cell+20}">{value*1000:.1f}</text>')
    for i, value in enumerate(ys):
        svg.append(f'<text x="35" y="{top+i*cell+25}">{value*1000:.1f}</text>')
    for case in cases:
        x, y = left + xs.index(case["values"][xname])*cell, top + ys.index(case["values"][yname])*cell
        score = outcome(groups.get(case["id"], []), seeds)
        if "xml" not in case:
            color, label = "#ddd", "X"
        elif score is None:
            color, label = "#fff", "?"
        else:
            color = "#84cfb0" if score[0] == len(seeds) else ("#f3c782" if score[0] else "#edaaaa")
            label = f"{score[0]}/{len(seeds)}"
            if score[1] != len(seeds) or score[2]:
                color = "#c4b3e8"
        svg.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="#555"/><text x="{x+10}" y="{y+25}">{label}</text>')
    svg.append(f'<text x="15" y="{height-22}">X: excluded by physical screening; ?: untested; purple: numerical/review flags</text></g></svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def summarize(directory, rows, manifest, config, statuses, state):
    directory = Path(directory)
    groups = grouped(rows)
    # Rebuilt from flushed per-test logs, so interrupted runs retain their evidence.
    with (directory / "results.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    fields = ["case_id", "kind", "values", "seed", "physical", "numerical", "balance", "simulated_seconds", "first_fall_s", "resets", "review_flags", "max_pitch_deg", "max_contact_force_n", "worker_status"]
    with (directory / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in manifest["cases"]:
            common = {"case_id": case["id"], "kind": case["kind"], "values": json.dumps(case["values"]),
                      "physical": case["physical"]["status"], "worker_status": statuses.get(case["id"], {}).get("status", "pending" if "xml" in case else "excluded")}
            present = {r["seed"]: r for r in groups.get(case["id"], [])}
            for seed in config["seeds"] if "xml" in case else [None]:
                row = present.get(seed)
                if row is None:
                    writer.writerow({**common, "seed": seed, "numerical": "not_run"})
                else:
                    writer.writerow({**common, **{k: row[k] for k in ("seed", "numerical", "balance", "simulated_seconds", "first_fall_s", "resets")},
                                     "review_flags": ";".join(row["review_flags"]), "max_pitch_deg": row["peaks"]["pitch_deg"],
                                     "max_contact_force_n": row["peaks"]["contact_force_n"]})
    brackets = transitions(rows, manifest, config)
    write_json(directory / "transitions.json", brackets)
    expected = sum("xml" in c for c in manifest["cases"]) * len(config["seeds"])
    errors = {k: v for k, v in statuses.items() if v["status"] != "completed"}
    if state == "completed" and (errors or len(rows) != expected):
        state = "completed_with_errors"
    write_json(directory / "status.json", {"state": state, "completed_trials": len(rows), "expected_trials": expected, "workers": statuses})
    valid = sum(r["numerical"] == "valid_observed" for r in rows)
    balanced = sum(r["balance"] for r in rows)
    reviews = sum(bool(r["review_flags"]) for r in rows)
    excluded = sum("xml" not in c for c in manifest["cases"])
    lines = ["# Self-balancer evaluation", "", f"Status: **{state}**. Property: **{config['property']}**.", "",
        f"Completed **{len(rows)}/{expected} trials**: {valid} numerically valid observations, {balanced} uninterrupted balance successes, {reviews} trials with review flags.",
        f"Excluded **{excluded} candidate configurations** before simulation. Their reasons are in manifest.json; exclusions are not simulator failures.", "",
        "## Interpretation", "",
        "Numerical validity, physical feasibility and controller performance are independent. Nonzero fixed-inertia CoM cases that pass necessary moment checks remain **unverified**, even when they balance. No nonzero physically established CoM interval is claimed.",
        "Passing samples and transition brackets do not establish an exact threshold or certify untested values. One-axis results hold the other coordinates at baseline; joint results describe sampled combinations only.", "",
        "## Baseline and protocol", "",
        "The explicit-inertia conversion is checked against the original legacy environment with the same frozen PPO, seeds and duration (reference_check.json). The corrected neutral reset is qualified separately (baseline_results.json).",
        f"Each trial uses {config['seconds_per_seed']:g} cumulative simulated seconds and seeds {config['seeds']}. Falls trigger seeded resets; balance success requires no fall for the full duration.",
        "Neutral mode uses zero speed/yaw targets, no pitch bias, correct quaternion ordering, and cleared derivative history. Original observation/action semantics and 200 Hz control are retained. The default reset samples +/-0.02 rad pitch, random yaw, and places the geometry 2 mm above the floor.",
        "Termination uses the inherited 50-degree pitch rule plus true body tilt above 50 degrees. This is survival, not precision position holding. Diagnostic thresholds are review heuristics, not calibrated hardware limits. Diagnostics inspect control-step endpoints; intermediate states can be missed.", "",
        "## Calculated bounds", "", "See bounds.json for exact inequalities, normalization lengths and assumptions. Geometry-derived maxima are not automatically achievable CoM positions.", ""]
    for b in manifest["bounds"]["bounds"]:
        if "geometric" in b:
            necessary = b["necessary_moment"]
            text = "none" if necessary is None else f"{necessary[0]*1000:.2f} to {necessary[1]*1000:.2f} mm"
            lines.append(f"- **{b['direction']}**: geometric offsets {b['geometric'][0]*1000:.2f} to {b['geometric'][1]*1000:.2f} mm; necessary moment interval {text}. Percent denominator: {b['normalization_length_m']*1000:.2f} mm.")
        else:
            lines.append(f"- **{b['direction']}**: search {b['search']} {b['units']}; {b['source']}.")
    if config["property"] == "center_of_mass" and len(config["directions"]) == 2:
        grid_svg(directory / "balance_grid.svg", manifest, groups, config["seeds"])
        lines += ["", "## Joint grid", "", "![Sampled balance outcomes](balance_grid.svg)", ""]
    lines += ["", "## Sample results", "", "Offsets below are millimeters for CoM; scalar properties use the units above. Exact values and all per-seed diagnostics are in summary.csv.", "",
              "| Case | Values | Physical status | Numerical valid / trials | Balance / trials | Review flags |",
              "| --- | --- | --- | --- | --- | --- |"]
    for case in manifest["cases"]:
        actual = groups.get(case["id"], [])
        scale = 1000 if config["property"] == "center_of_mass" else 1
        values = ", ".join(f"{k}={v*scale:.3g}" for k, v in case["values"].items())
        lines.append(f"| {case['id']} | {values} | {case['physical']['status']} | {sum(r['numerical']=='valid_observed' for r in actual)}/{len(actual)} | {sum(r['balance'] for r in actual)}/{len(actual)} | {sum(bool(r['review_flags']) for r in actual)} |")
    lines += ["", "## Refinement and evidence", "",
              f"Observed endpoint transitions: **{len(brackets)}** (transitions.json). Refinement budget: {config['max_refinement_cases']} additional cases. Any unresolved bracket or untested region remains unknown.",
              "Artifacts: config.json, bounds.json, manifest.json, baseline.xml, converted_reference.xml, policy.zip, provenance.json, source snapshots, reference_check.json, baseline_results.json, per-worker logs/traces, results.jsonl and summary.csv.",
              "Each trace records policy actions, controls, state, contacts and targets. Review flags are kept separate from numerical anomalies and falls. No training or hardware testing was performed."]
    if state == "baseline_failed":
        lines.insert(4, "**The neutral baseline did not qualify. Property rollouts were stopped; analytical bounds and generated cases remain available.**")
    retried = sum("retry" in value for value in statuses.values())
    if retried:
        lines += ["", f"**Infrastructure retries:** {retried} interrupted cases were rerun using their original saved source and dependency versions. Original attempts remain on disk; selected_attempt.json in each case folder identifies the included attempt. Status history is retained in status.json."]
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(directory / "bounds.json", manifest["bounds"])
