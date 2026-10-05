"""Poster figures from the committed results in results_v3/ (no data access needed).

    python analysis/poster_figures.py            # -> results_v3/figures/fig*.png + .pdf

fig1_backchannels   the headline: listener backchannels, humans vs every LLM condition
fig2_act_mix        what the talk is made of (coarse dialogue acts), humans vs architectures
fig3_distance       distance from human act structure (JSD) per condition, with reference lines
fig4_turn_length    words per turn per condition vs humans
fig5_alignment      ALIGN: earlier -> later convergence, and length-controlled gap vs humans
"""
from __future__ import annotations

import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results_v3"
DA = RES / "dialogue_acts"
OUT = RES / "figures"

ARCHS = ["C1", "C2", "C3", "C4"]
ARCH_NAME = {"C1": "C1\nall at once", "C2": "C2\nturn by turn", "C3": "C3\n2 agents\n(same model)",
             "C4": "C4\n2 agents\n(2 models)"}
ARCH_COLOR = {"C1": "#4C72B0", "C2": "#55A868", "C3": "#DD8452", "C4": "#C44E52"}
PROMPTS = ["P0", "P1", "P2"]
PROMPT_ALPHA = {"P0": 0.45, "P1": 0.75, "P2": 1.0}
HUMAN = "#222222"

plt.rcParams.update({"font.size": 15, "axes.titlesize": 18, "axes.labelsize": 16,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.dpi": 100})


def rows(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def by(path: pathlib.Path, key: str = "condition") -> dict[str, dict]:
    return {r[key]: r for r in rows(path)}


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def grouped(ax, values: dict[str, float], errors: dict[str, float] | None = None, fmt: str = "{:.0f}"):
    """Bars grouped by architecture, one bar per prompt level; returns x centers."""
    width, centers = 0.26, []
    for i, a in enumerate(ARCHS):
        centers.append(i + 1)
        for j, p in enumerate(PROMPTS):
            c = f"{a}-{p}"
            x = i + 1 + (j - 1) * width
            ax.bar(x, values[c], width * 0.92, color=ARCH_COLOR[a], alpha=PROMPT_ALPHA[p],
                   yerr=None if errors is None else errors[c], capsize=3, ecolor="#555")
            ax.text(x, values[c] + (0 if errors is None else errors[c]), fmt.format(values[c]),
                    ha="center", va="bottom", fontsize=10.5)
    return centers


def prompt_legend(ax, loc="upper left"):
    from matplotlib.patches import Patch
    handles = [Patch(color="#777", alpha=PROMPT_ALPHA[p], label=lab) for p, lab in
               zip(PROMPTS, ["P0 paper prompt", "P1 spoken + persona", "P2 + real example"])]
    ax.legend(handles=handles, loc=loc, fontsize=12, title="bar shade = prompt", title_fontsize=12)


def fig1_backchannels() -> None:
    dist = by(DA / "da_distribution_coarse_by_condition.csv")
    rule = by(DA / "da_rule_crosscheck.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1.6, 1]})

    vals = {c: 100 * float(dist[c]["Backchannel"]) for c in dist if c[0] == "C"}
    centers = grouped(a1, vals, fmt="{:.1f}")
    hum = 100 * float(dist["SB-tagger"]["Backchannel"])
    a1.bar(0, hum, 0.6, color=HUMAN)
    a1.text(0, hum, f"{hum:.0f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    a1.set_xticks([0] + centers, ["Humans\n(Switchboard)"] + [ARCH_NAME[a] for a in ARCHS], fontsize=12.5)
    a1.set_ylabel("% of talk units tagged backchannel")
    a1.set_title("(a) Backchannels, as tagged")
    prompt_legend(a1, loc="upper right")

    llm = [100 * float(rule[c]["rule_backchannel_rate"]) for c in rule if c != "SB"]
    hb = 100 * float(rule["SB"]["rule_backchannel_rate"])
    a2.bar([0, 1], [hb, max(llm)], 0.55, color=[HUMAN, "#999"])
    a2.text(0, hb, f"{hb:.0f}%", ha="center", va="bottom", fontsize=15, fontweight="bold")
    a2.text(1, max(llm), f"{min(llm):.1f}–{max(llm):.1f}%", ha="center", va="bottom", fontsize=15,
            fontweight="bold")
    a2.set_xticks([0, 1], ["Humans", "All 12 LLM\nconditions"])
    a2.set_ylabel("% of units that are ONLY\n'uh-huh / yeah / right'")
    a2.set_title("(b) Stand-alone listener turns")
    a2.set_ylim(0, hb * 1.25)
    fig.suptitle("LLM conversations are missing the listener", fontsize=21, fontweight="bold", y=1.02)
    save(fig, "fig1_backchannels")


def fig2_act_mix() -> None:
    dist = by(DA / "da_distribution_coarse_by_condition.csv")
    groups = [("Statement", ["Statement"]), ("Opinion", ["Opinion"]), ("Backchannel", ["Backchannel"]),
              ("Question", ["YesNoQuestion", "WhQuestion/OpenQuestion"]), ("Agreement", ["Agreement"]),
              ("Other", ["Answer", "Directive", "Repair/Hedge", "Abandoned/Other"])]
    colors = ["#8DA0CB", "#E78AC3", "#222222", "#FC8D62", "#66C2A5", "#CCCCCC"]

    def share(conds: list[str]) -> list[float]:
        return [100 * np.mean([sum(float(dist[c][k]) for k in keys) for c in conds]) for _, keys in groups]

    bars = [("Humans", share(["SB-tagger"]))] + [
        (ARCH_NAME[a].replace("\n", " "), share([f"{a}-{p}" for p in PROMPTS])) for a in ARCHS]
    fig, ax = plt.subplots(figsize=(14, 5.2))
    for i, (name, vals) in enumerate(bars[::-1]):
        left = 0
        for (g, _), v, col in zip(groups, vals, colors):
            ax.barh(i, v, left=left, color=col, edgecolor="white")
            if v >= 4:
                ax.text(left + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=12,
                        color="white" if col in ("#222222",) else "black")
            left += v
    ax.set_yticks(range(len(bars)), [b[0] for b in bars[::-1]])
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of talk units (LLM rows: mean over prompts P0–P2)")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in colors], [g for g, _ in groups],
              ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=13)
    ax.spines["left"].set_visible(False)
    fig.suptitle("Fewer listener reactions, more opinions and questions", fontsize=20,
                 fontweight="bold", y=1.08)
    save(fig, "fig2_act_mix")


def fig3_distance() -> None:
    jsd = {r["condition"]: r for r in rows(DA / "da_jsd_vs_sb.csv")
           if r["label_set"] == "coarse" and r["view"] == "tagger_human_vs_tagger_llm_primary"}
    calib = next(float(r["jsd_dist"]) for r in rows(DA / "da_jsd_vs_sb.csv")
                 if r["label_set"] == "coarse" and r["view"] == "gold_vs_tagger_human_calibration")
    floor = max(float(r["p95_jsd"]) for r in rows(DA / "da_noise_floor.csv") if r["label_set"] == "coarse")
    fig, ax = plt.subplots(figsize=(12, 5.6))
    vals = {c: float(jsd[c]["jsd_dist"]) for c in jsd}
    centers = grouped(ax, vals, fmt="{:.2f}")
    l1 = ax.axhline(calib, color=HUMAN, ls="--", lw=1.5)
    l2 = ax.axhline(floor, color=HUMAN, ls=":", lw=1.5)
    ax.set_xticks(centers, [ARCH_NAME[a] for a in ARCHS])
    ax.set_xlim(0.5, 4.5)
    ax.set_ylabel("distance from humans\n(Jensen–Shannon, dialogue-act mix)")
    ax.set_title("Separate agents do not bring the structure closer to humans", fontweight="bold")
    prompt_legend(ax)
    ax.add_artist(ax.get_legend())
    ax.legend([l1, l2], [f"tagger's own error on human talk ({calib:.3f})",
                         f"sampling noise: random human samples of the same size ({floor:.3f})"],
              loc="upper right", fontsize=12)
    ax.set_ylim(0, max(vals.values()) * 1.25)
    save(fig, "fig3_distance")


def fig4_turn_length() -> None:
    m = by(RES / "metrics_summary.csv")
    fig, ax = plt.subplots(figsize=(12, 5.6))
    vals = {c: float(m[c]["mean_words_per_turn_mean"]) for c in m if c[0] == "C"}
    err = {c: 1.96 * float(m[c]["mean_words_per_turn_sd"]) / np.sqrt(float(m[c]["n"])) for c in vals}
    centers = grouped(ax, vals, err)
    hum = float(m["SB"]["mean_words_per_turn_mean"])
    hl = ax.axhline(hum, color=HUMAN, ls="--", lw=2)
    ax.set_xticks(centers, [ARCH_NAME[a] for a in ARCHS])
    ax.set_xlim(0.5, 4.5)
    ax.set_ylabel("mean words per turn (±95% CI)")
    ax.set_title("Separate agents talk in long, message-like turns", fontweight="bold")
    prompt_legend(ax)
    ax.add_artist(ax.get_legend())
    ax.legend([hl], [f"humans: {hum:.1f} words per turn"], loc="upper center", fontsize=13)
    save(fig, "fig4_turn_length")


def fig5_alignment() -> None:
    el = [r for r in rows(RES / "alignment_earlier_later.csv") if r["variant"] == "all_pairs"]
    el = {r["condition"]: r for r in el}
    lc = by(RES / "alignment_length_controlled.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1, 1.3]})

    def line(ax, conds, color, label, lw=2.5):
        e = np.mean([float(el[c]["earlier"]) for c in conds])
        l = np.mean([float(el[c]["later"]) for c in conds])
        ax.plot([0, 1], [e, l], "-o", color=color, lw=lw, ms=8, label=label)

    line(a1, ["SB"], HUMAN, f"Humans (p = {float(el['SB']['wilcoxon_p']):.2f})", lw=3.5)
    for a in ARCHS:
        line(a1, [f"{a}-P1"], ARCH_COLOR[a],
             ARCH_NAME[a].replace("\n", " ") + f" (p = {float(el[a + '-P1']['wilcoxon_p']):.3f})")
    a1.set_xticks([0, 1], ["first half", "second half"])
    a1.set_xlim(-0.2, 1.2)
    a1.set_ylabel("conceptual alignment\n(ALIGN, adjacent turns)")
    a1.set_title("(a) Over a conversation (prompt P1):\nhumans stay flat, most LLM setups converge")
    a1.legend(fontsize=11.5, loc="upper center", bbox_to_anchor=(0.5, -0.12))

    vals = {c: float(lc[c]["adjusted_diff_vs_SB"]) for c in lc if c[0] == "C" and "noloop" not in c}
    raw = {c: float(lc[c]["raw_diff_vs_SB"]) for c in vals}
    width = 0.26
    for i, a in enumerate(ARCHS):
        for j, p in enumerate(PROMPTS):
            c, x = f"{a}-{p}", i + 1 + (j - 1) * width
            a2.bar(x, raw[c], width * 0.92, color="none", edgecolor=ARCH_COLOR[a], lw=1.5, ls="--")
            a2.bar(x, vals[c], width * 0.92, color=ARCH_COLOR[a], alpha=PROMPT_ALPHA[p])
    a2.axhline(0, color=HUMAN, lw=1.5)
    a2.set_xticks(range(1, 5), [ARCH_NAME[a] for a in ARCHS], fontsize=12)
    a2.set_ylabel("alignment minus humans")
    a2.set_title("(b) Gap vs humans: raw (dashed outline)\nvs controlling for turn length (filled)")
    fig.subplots_adjust(wspace=0.35)
    fig.suptitle("Long turns explain most of the LLMs' extra alignment", fontsize=20,
                 fontweight="bold", y=1.06)
    save(fig, "fig5_alignment")


if __name__ == "__main__":
    fig1_backchannels()
    fig2_act_mix()
    fig3_distance()
    fig4_turn_length()
    fig5_alignment()
