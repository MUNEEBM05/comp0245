"""Builds results/report.html from results/all_results.json and the tier CSVs."""
import csv
import glob
import html
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
SCENES = ["clear", "sparse", "cluttered", "gap", "blocked"]
with open(os.path.join(R, "all_results.json")) as fh:
    RES = json.load(fh)
BY = {r["label"]: r for r in RES}


def pct(a, n):
    return round(a / n * 100)


def bars(labels):
    out = []
    for k in labels:
        r = BY[k]
        before = r["group"] == "Before"
        val = f"{r['reached']}" if before else f"{r['reached']} / {r['held']}"
        tip = f"{k}: {r['reached']} reached" + ("" if before else f", {r['held']} held 5 s") + f", of {r['n']} drives"
        held = "" if before else f'<i class="b held" style="width:{r["held"] / r["n"] * 100:.1f}%"></i>'
        out.append(f'<div class="row" title="{html.escape(tip)}"><span class="lab">{html.escape(k)}</span><span class="track"><i class="b reach" style="width:{r["reached"] / r["n"] * 100:.1f}%"></i>{held}</span><span class="val">{val}</span></div>')
    return "".join(out)


def all_table():
    rows, group = [], None
    for r in RES:
        if r["group"] != group:
            group = r["group"]
            rows.append(f'<tr class="grp"><th colspan="11" scope="colgroup">{group}</th></tr>')
        before = group == "Before"
        cells = "".join(f"<td>{r['scenes'][s]['reached']}</td>" if before else f"<td>{r['scenes'][s]['reached']}/{r['scenes'][s]['held']}</td>" for s in SCENES)
        tot = f"{r['reached']}" if before else f"{r['reached']} / {r['held']}"
        p = f"{pct(r['reached'], r['n'])}" if before else f"{pct(r['reached'], r['n'])} / {pct(r['held'], r['n'])}"
        rows.append(f'<tr><th scope="row">{html.escape(r["label"])}</th><td class="s">{tot}</td><td class="s">{p}</td>{cells}<td>{r["timeouts"]}</td><td>{r["arm_crashes"]}</td><td>{r["obstacle_hits"]}</td></tr>')
    return "".join(rows)


def heat():
    keep = ["Planner + expert arm", "A seed 0 + expert, brake", "A seed 1 + expert, brake", "Planner + R", "A seed 0 + R, brake", "A seed 1 + R, brake", "Camera + planner + expert", "Camera + planner + R",
            "Camera + planner + vS seed 0", "Camera + A seed 0 + R, brake", "Camera + A seed 1 + R, brake", "Search + planner + expert"]
    out = []
    for k in keep:
        tds = ""
        for s in SCENES:
            v = BY[k]["scenes"][s]["held"]
            tds += f'<td class="h" style="--a:{v / 30:.2f}" title="{html.escape(k)}, {s}: {v} of 30 held"><span>{v}</span></td>'
        out.append(f'<tr><th scope="row">{html.escape(k)}</th>{tds}</tr>')
    return "".join(out)


def tiers():
    setups = [("Original expert", "expert_hold5", "ev"), ("R alone", "vR_ik_hold5", "ev"), ("Planner + expert arm", "planner_expert_hold5", "ev"), ("Planner + R", "planner_vR_ik_hold5", "ev"),
              ("A seed 0 + expert, brake", "brake_driver_A_expert_hold5", "ev"), ("A seed 1 + expert, brake", "brake_driver_A_s1_expert_hold5", "ev"), ("Camera + planner + expert", "cam_planner_expert_bench", "sn")]
    rows = []
    for label, name, kind in setups:
        cells, N, A, H = "", 0, 0, 0
        ok = True
        for t in (1, 2, 3):
            try:
                if kind == "ev":
                    with open(os.path.join(R, "tiers", f"{name}_tier{t}.csv")) as fh:
                        x = list(csv.DictReader(fh))
                else:
                    with open(os.path.join(R, "tiers", name + ".csv")) as fh:
                        x = [y for y in csv.DictReader(fh) if y["scenario"] == f"tier{t}"]
            except OSError:
                ok = False
                break
            a, h = sum(1 for y in x if y.get("first_reach_s") not in ("", None)), sum(1 for y in x if y["outcome"] == "reached")
            N, A, H = N + len(x), A + a, H + h
            cells += f"<td>{a}/{h}</td>"
        if ok and N == 150:
            rows.append(f'<tr><th scope="row">{html.escape(label)}</th><td class="s">{A} / {H}</td><td class="s">{pct(A, N)} / {pct(H, N)}</td>{cells}</tr>')
    return "".join(rows), len(rows), len(setups)


T_ROWS, T_DONE, T_ALL = tiers()
if T_DONE == T_ALL:
    T_NOTE = "All setups on the team's benchmark, no obstacles, 50 scenes per tier, 90 s limit (camera 110 s). Tier 1 is like training, tier 2 has higher or farther targets, tier 3 has unusual starts."
else:
    T_NOTE = f"<strong>Still running:</strong> {T_DONE} of {T_ALL} setups have finished. This page will be updated when the rest are in."
ROUTES = '<figure class="fig"><img src="figures/routes.png" alt="Top-down routes in the five scene types for the planner, learned driver A and camera mode"></figure><p class="cap">One validation scene of each type, for illustration. The camera route is the true path, which the robot itself does not know.</p>' if os.path.exists(os.path.join(HERE, "figures", "routes.png")) else ""

KEY = ["Original expert", "Planner + expert arm", "A seed 0 + expert, brake", "A seed 1 + expert, brake", "Planner + R", "A seed 0 + R, brake", "A seed 1 + R, brake", "Camera + planner + expert",
       "Camera + A seed 0 + expert", "Camera + A seed 1 + expert", "Camera + planner + R", "Camera + A seed 0 + R, brake", "Camera + A seed 1 + R, brake", "Search + planner + expert", "Odometry only + planner + expert"]

with open(os.path.join(HERE, "report_template.html")) as fh:
    page = fh.read()
for k, v in {"{{BARS}}": bars(KEY), "{{ALL}}": all_table(), "{{HEAT}}": heat(), "{{TIERS}}": T_ROWS, "{{TIER_NOTE}}": T_NOTE, "{{ROUTES}}": ROUTES}.items():
    page = page.replace(k, v)
with open(os.path.join(R, "report.html"), "w") as fh:
    fh.write(page)
print("tiers", T_DONE, "of", T_ALL)
