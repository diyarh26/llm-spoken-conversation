"""README figures and diagrams, rendered from the committed results in light and dark themes.

    python analysis/figures.py              # -> docs/assets/*-{light,dark}.svg
    python analysis/figures.py --out DIR    # render somewhere else (CI does this)
    python analysis/figures.py --extract    # refresh results/figure_data/turn_rhythm.json
                                            # (needs the licensed Switchboard corpus)

Every number comes from results/ (and data/generated/ for the LLM turn sequences), so the
figures rebuild without model access or the corpus. One color per architecture is used in
every figure (C1 blue, C2 orange, C3 aqua, C4 amber; humans in neutral ink). The palette
was checked for color-vision deficiency and contrast against GitHub's light and dark page
backgrounds; light-mode amber/aqua sit below 3:1 contrast, so every mark is also labelled.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import pathlib
import statistics
import sys
from dataclasses import dataclass
from html import escape

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RES = ROOT / "results"
DA = RES / "dialogue_acts"
RHYTHM = RES / "figure_data" / "turn_rhythm.json"
OUT = ROOT / "docs" / "assets"

ARCHS = ("C1", "C2", "C3", "C4")
PROMPTS = ("P0", "P1", "P2")
ARCH_NAME = {
    "C1": "All at once",
    "C2": "Turn by turn",
    "C3": "Two agents",
    "C4": "Two models",
}
ARCH_DESC = {
    "C1": "one model writes the whole call",
    "C2": "one model, one turn at a time",
    "C3": "two agents, same model",
    "C4": "Vicuna agent ↔ Mistral agent",
}
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"


@dataclass(frozen=True)
class Theme:
    name: str
    ink: str  # primary text
    muted: str  # secondary text
    faint: str  # axis labels, footnotes
    grid: str  # hairlines
    surface: str  # the page behind the (transparent) figure
    human: str  # human marks
    human_b: str
    arch: dict
    arch_b: dict  # second-speaker tints


def _mix(hex_a: str, hex_b: str, t: float) -> str:
    """Blend two colors: t=0 -> a, t=1 -> b."""
    a = [int(hex_a[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(hex_b[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b, strict=True))


def _theme(name, ink, muted, faint, grid, surface, human, arch, tint=0.5) -> Theme:
    return Theme(
        name=name,
        ink=ink,
        muted=muted,
        faint=faint,
        grid=grid,
        surface=surface,
        human=human,
        human_b=_mix(human, surface, tint + 0.05),
        arch=arch,
        arch_b={k: _mix(v, surface, tint) for k, v in arch.items()},
    )


THEMES = {
    "light": _theme(
        "light",
        ink="#1f2328",
        muted="#59636e",
        faint="#818b98",
        grid="#d1d9e0",
        surface="#ffffff",
        human="#57606a",
        arch={"C1": "#2a78d6", "C2": "#eb6834", "C3": "#1baf7a", "C4": "#eda100"},
    ),
    "dark": _theme(
        "dark",
        ink="#e6edf3",
        muted="#9198a1",
        faint="#6e7681",
        grid="#3d444d",
        surface="#0d1117",
        human="#b1bac4",
        arch={"C1": "#3987e5", "C2": "#d95926", "C3": "#199e70", "C4": "#c98500"},
        tint=0.4,
    ),
}


# --- data ----------------------------------------------------------------------------


def _rows(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def coarse_shares() -> dict[str, dict[str, float]]:
    return {
        r["condition"]: {k: float(v) for k, v in r.items() if k not in ("condition",)}
        for r in _rows(DA / "da_distribution_coarse_by_condition.csv")
    }


def jsd_with_ci() -> dict[str, tuple[float, float, float]]:
    block = (RES / "da_jsd_bootstrap.csv").read_text(encoding="utf-8").split("\n\n")[0]
    return {
        r["condition"]: (float(r["jsd"]), float(r["ci_low"]), float(r["ci_high"]))
        for r in csv.DictReader(block.strip().splitlines())
    }


def reference_lines() -> tuple[float, float]:
    """(tagger error on human speech, largest sampling-noise floor) for the coarse JSD."""
    rows = _rows(DA / "da_jsd_vs_sb.csv")
    tagger = next(
        float(r["jsd_dist"])
        for r in rows
        if r["label_set"] == "coarse" and r["view"] == "gold_vs_tagger_human_calibration"
    )
    floor = max(
        float(r["p95_jsd"]) for r in _rows(DA / "da_noise_floor.csv") if r["label_set"] == "coarse"
    )
    return tagger, floor


def words_per_turn_by_conversation() -> dict[str, list[float]]:
    records = json.loads((RES / "metrics_per_conversation.json").read_text(encoding="utf-8"))
    out: dict[str, list[float]] = {}
    for r in records:
        out.setdefault(r["condition"], []).append(float(r["mean_words_per_turn"]))
    return out


def summary() -> dict[str, dict]:
    return {r["condition"]: r for r in _rows(RES / "metrics_summary.csv")}


def listener_share() -> dict[str, float]:
    """Share of talk units that are ONLY a listener token ('uh-huh', 'yeah', 'right'...)."""
    return {
        r["condition"]: float(r["rule_backchannel_rate"])
        for r in _rows(DA / "da_rule_crosscheck.csv")
    }


# --- matplotlib figures --------------------------------------------------------------


def _mpl(theme: Theme):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10.5,
            "svg.fonttype": "path",
            "axes.edgecolor": theme.grid,
            "axes.labelcolor": theme.muted,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "xtick.color": theme.faint,
            "ytick.color": theme.ink,
            "xtick.labelcolor": theme.faint,
            "ytick.labelcolor": theme.ink,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "grid.color": theme.grid,
            "grid.linewidth": 0.6,
            "text.color": theme.ink,
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "savefig.transparent": True,
        }
    )
    return plt


def _headline(fig, theme: Theme, title: str, subtitle: str, y: float = 0.985) -> None:
    fig.text(
        0.012, y, title, ha="left", va="top", fontsize=14.5, fontweight="bold", color=theme.ink
    )
    fig.text(0.012, y - 0.085, subtitle, ha="left", va="top", fontsize=10.5, color=theme.muted)


def _save(fig, out: pathlib.Path, name: str, theme: Theme) -> pathlib.Path:
    path = out / f"{name}-{theme.name}.svg"
    # No tight bbox: every figure keeps its exact 10.2-inch canvas, so text renders at the
    # same size in the README whatever each figure's content width is.
    fig.savefig(path, format="svg", metadata={"Date": None})
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def fig_acts(theme: Theme, out: pathlib.Path) -> pathlib.Path:
    """What the talk is made of: three dialogue acts, humans vs each architecture."""
    plt = _mpl(theme)
    shares = coarse_shares()
    panels = [
        ("Listener backchannels", ["Backchannel"]),
        ("Opinions", ["Opinion"]),
        ("Questions", ["YesNoQuestion", "WhQuestion/OpenQuestion"]),
    ]
    rows = ["SB-tagger", *ARCHS]
    labels = ["Humans", *(f"{a}  {ARCH_NAME[a]}" for a in ARCHS)]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.7), sharey=True)
    fig.subplots_adjust(left=0.135, right=0.985, top=0.70, bottom=0.11, wspace=0.12)
    for ax, (title, keys) in zip(axes, panels, strict=True):

        def pct(cond, keys=keys):
            return 100 * sum(shares[cond][k] for k in keys)

        human = pct("SB-tagger")
        top = max(human, *(pct(f"{a}-{p}") for a in ARCHS for p in PROMPTS))
        ax.axvline(human, color=theme.human, lw=1, alpha=0.55, zorder=1)
        for i, row in enumerate(rows):
            y = len(rows) - 1 - i
            ax.axhline(y, color=theme.grid, lw=0.6, zorder=0)
            if row == "SB-tagger":
                ax.plot(human, y, "o", ms=9, color=theme.human, mec=theme.surface, mew=2, zorder=3)
                ax.text(human + top * 0.04, y + 0.28, f"{human:.0f}%", fontsize=10,
                        fontweight="bold", color=theme.ink, va="center")  # fmt: skip
                continue
            vals = [pct(f"{row}-{p}") for p in PROMPTS]
            color = theme.arch[row]
            ax.plot([min(vals), max(vals)], [y, y], color=color, lw=5, alpha=0.35,
                    solid_capstyle="round", zorder=2)  # fmt: skip
            mean = statistics.mean(vals)
            ax.plot(mean, y, "o", ms=9, color=color, mec=theme.surface, mew=2, zorder=3)
            ax.text(max(vals) + top * 0.04, y, f"{mean:.0f}%", fontsize=10, color=theme.ink,
                    va="center")  # fmt: skip
        ax.set_xlim(0, top * 1.28)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=theme.ink, pad=8)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.tick_params(axis="x", labelsize=9)
        ax.spines["bottom"].set_visible(False)
    axes[0].set_yticks(range(len(rows))[::-1], labels)
    for tick, row in zip(axes[0].get_yticklabels(), rows, strict=True):
        tick.set_fontweight("bold" if row == "SB-tagger" else "normal")
    _headline(
        fig,
        theme,
        "LLMs replace listening with opinions and questions",
        "Share of talk units by dialogue act (one automatic tagger for both). "
        "Dot = mean over prompts P0–P2, bar = range.",
    )
    return _save(fig, out, "fig-acts", theme)


def fig_distance(theme: Theme, out: pathlib.Path) -> pathlib.Path:
    """Distance of each condition's act mix from humans, with conversation-bootstrap CIs."""
    plt = _mpl(theme)
    ci = jsd_with_ci()
    tagger, floor = reference_lines()
    fig, ax = plt.subplots(figsize=(10.2, 4.9))
    fig.subplots_adjust(left=0.235, right=0.975, top=0.81, bottom=0.11)
    y, yticks, ylabels = 0.0, [], []
    centers = {}
    for a in reversed(ARCHS):
        block = []
        for p in reversed(PROMPTS):
            jsd, lo, hi = ci[f"{a}-{p}"]
            ax.plot([lo, hi], [y, y], color=theme.arch[a], lw=2, solid_capstyle="round")
            ax.plot(jsd, y, "o", ms=8.5, color=theme.arch[a], mec=theme.surface, mew=2, zorder=3)
            ax.text(hi + 0.004, y, f"{jsd:.3f}", va="center", fontsize=9, color=theme.muted)
            yticks.append(y)
            ylabels.append(p)
            block.append(y)
            y += 1
        centers[a] = statistics.mean(block)
        y += 0.8
    for a, c in centers.items():
        ax.text(-0.06, c, f"{a}  {ARCH_NAME[a]}", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=10.5, fontweight="bold", color=theme.ink)  # fmt: skip
    ax.set_yticks(yticks, ylabels, fontsize=9, color=theme.faint)
    ax.tick_params(axis="y", pad=4, labelcolor=theme.faint)
    ax.axvline(tagger, color=theme.muted, lw=1, ls=(0, (1, 2)))
    ax.text(tagger + 0.003, y - 0.5, f"tagger's own error\non human talk ({tagger:.3f})",
            fontsize=8.5, color=theme.muted, va="top")  # fmt: skip
    ax.axvspan(0, floor, color=theme.muted, alpha=0.15, lw=0)
    ax.set_xlim(0, 0.32)
    ax.set_ylim(-0.7, y - 0.5)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel(
        "Jensen–Shannon distance from human dialogue-act mix  (0 = identical)", fontsize=9.5
    )
    ax.tick_params(axis="x", labelsize=9)
    _headline(
        fig,
        theme,
        "Generating the whole call at once is closest to human structure",
        "95% CIs from 2,000 conversation-level bootstrap replicates; C1 is closer than "
        "every other architecture in all of them.",
        y=0.99,
    )
    return _save(fig, out, "fig-distance", theme)


def fig_turn_length(theme: Theme, out: pathlib.Path) -> pathlib.Path:
    """Per-conversation mean words per turn: humans vs every condition (log scale)."""
    plt = _mpl(theme)
    import numpy as np

    data = words_per_turn_by_conversation()
    human = data["SB"]
    q1, q3 = np.percentile(human, [25, 75])
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(10.2, 5.3))
    fig.subplots_adjust(left=0.215, right=0.975, top=0.84, bottom=0.10)
    rows = [("SB", "Humans", theme.human)] + [
        (f"{a}-{p}", p, theme.arch[a]) for a in ARCHS for p in PROMPTS
    ]
    ax.axvspan(q1, q3, color=theme.human, alpha=0.10, lw=0)
    positions, y = {}, 0.0
    for cond, _label, _color in rows:
        positions[cond] = y
        y += 1.35 if cond in ("SB", "C1-P2", "C2-P2", "C3-P2") else 1
    for cond, _label, color in rows:
        yy = positions[cond]
        vals = np.array(data[cond])
        jitter = rng.uniform(-0.28, 0.28, len(vals))
        ax.scatter(vals, yy + jitter, s=13, color=color, alpha=0.75, lw=0, zorder=2)
        med = float(np.median(vals))
        ax.plot([med, med], [yy - 0.38, yy + 0.38], color=theme.ink, lw=2, zorder=3)
        ax.text(330, yy, f"{med:.0f}", va="center", ha="right", fontsize=9.5, color=theme.ink)
    ax.text(330, -0.95, "median", va="center", ha="right", fontsize=8.5, color=theme.faint)
    ax.set_xscale("log")
    ax.set_xlim(3, 340)
    ticks = [5, 10, 20, 50, 100, 200]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_yticks([positions[c] for c, _, _ in rows], [lab for _, lab, _ in rows], fontsize=9)
    for tick, (cond, _, _) in zip(ax.get_yticklabels(), rows, strict=True):
        tick.set_color(theme.ink if cond == "SB" else theme.faint)
        tick.set_fontweight("bold" if cond == "SB" else "normal")
    for a in ARCHS:
        c = statistics.mean(positions[f"{a}-{p}"] for p in PROMPTS)
        ax.text(-0.045, c, f"{a}  {ARCH_NAME[a]}", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=10.5, fontweight="bold", color=theme.ink)  # fmt: skip
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel("Mean words per turn, per conversation  (log scale)", fontsize=9.5)
    ax.tick_params(axis="x", labelsize=9)
    _headline(
        fig,
        theme,
        "Separate agents write messages, not spoken turns",
        f"Each dot is one conversation; tick = median. Shaded band = the middle half "
        f"of the 50 human calls ({q1:.0f}–{q3:.0f} words).",
        y=0.99,
    )
    return _save(fig, out, "fig-turn-length", theme)


def fig_alignment(theme: Theme, out: pathlib.Path) -> pathlib.Path:
    """ALIGN conceptual alignment: convergence over the call, and the length-controlled gap."""
    plt = _mpl(theme)
    el = {
        r["condition"]: r
        for r in _rows(RES / "alignment_earlier_later.csv")
        if r["variant"] == "all_pairs"
    }
    lc = {r["condition"]: r for r in _rows(RES / "alignment_length_controlled.csv")}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.07, right=0.885, top=0.75, bottom=0.13, wspace=0.42)

    lines = [("SB", "Humans", theme.human)] + [(f"{a}-P1", a, theme.arch[a]) for a in ARCHS]
    ends = []
    for cond, label, color in lines:
        e, late = float(el[cond]["earlier"]), float(el[cond]["later"])
        p = float(el[cond]["wilcoxon_p"])
        lw = 2.6 if cond == "SB" else 2
        a1.plot([0, 1], [e, late], color=color, lw=lw, solid_capstyle="round", zorder=2)
        a1.plot([0, 1], [e, late], "o", ms=7.5, color=color, mec=theme.surface, mew=2, zorder=3)
        ptxt = "p < 0.001" if p < 0.001 else f"p = {p:.2f}" if p >= 0.01 else f"p = {p:.3f}"
        ends.append([late, late, f"{label}  ({ptxt})", cond == "SB"])
    # Spread end labels that would overlap; a hairline leader keeps each tied to its line.
    ends.sort(key=lambda r: r[0])
    min_gap = 0.026
    for prev, cur in itertools.pairwise(ends):
        cur[1] = max(cur[1], prev[1] + min_gap)
    for y_line, y_label, text, bold in ends:
        if abs(y_label - y_line) > 1e-6:
            a1.plot([1.03, 1.1], [y_line, y_label], color=theme.faint, lw=0.7)
        a1.text(1.13, y_label, text, va="center", fontsize=9.5,
                fontweight="bold" if bold else "normal", color=theme.ink)  # fmt: skip
    a1.set_xticks([0, 1], ["first half", "second half"], fontsize=9.5)
    a1.set_xlim(-0.12, 2.15)
    a1.set_ylabel("conceptual alignment (ALIGN)", fontsize=9.5)
    a1.tick_params(axis="y", labelsize=9, labelcolor=theme.faint)
    lo, hi = a1.get_ylim()
    a1.hlines(
        [t for t in a1.get_yticks() if lo <= t <= hi],
        -0.12,
        1.0,
        color=theme.grid,
        lw=0.6,
        zorder=0,
    )
    a1.set_ylim(lo, hi)
    a1.spines["bottom"].set_visible(False)
    a1.set_title("Over a call (prompt P1)", loc="left", fontsize=11, fontweight="bold", pad=8)

    rows = list(reversed(ARCHS))
    for i, a in enumerate(rows):
        raw = statistics.mean(float(lc[f"{a}-{p}"]["raw_diff_vs_SB"]) for p in PROMPTS)
        adj = statistics.mean(float(lc[f"{a}-{p}"]["adjusted_diff_vs_SB"]) for p in PROMPTS)
        color = theme.arch[a]
        a2.plot([raw, adj], [i, i], color=color, lw=2, alpha=0.5, zorder=1)
        a2.plot(raw, i, "o", ms=8, mfc=theme.surface, mec=color, mew=2, zorder=2)
        a2.plot(adj, i, "o", ms=8.5, color=color, mec=theme.surface, mew=2, zorder=3)
        a2.text(1.03, i, f"{raw:+.2f} → {adj:+.2f}", transform=a2.get_yaxis_transform(),
                va="center", fontsize=9.5, color=theme.ink)  # fmt: skip
    a2.text(1.03, len(rows) - 0.45, "raw → controlled", transform=a2.get_yaxis_transform(),
            va="center", fontsize=8.5, color=theme.faint)  # fmt: skip
    a2.axvline(0, color=theme.human, lw=1.2)
    a2.set_yticks(range(len(rows)), [f"{a}  {ARCH_NAME[a]}" for a in rows], fontsize=9.5)
    a2.set_xlim(-0.03, 0.36)
    a2.set_ylim(-0.6, len(rows) - 0.4)
    a2.grid(axis="x")
    a2.set_axisbelow(True)
    a2.tick_params(axis="x", labelsize=9)
    a2.set_xlabel("alignment minus humans   (○ raw   ● controlling for turn length)", fontsize=9.5)
    a2.set_title("Gap to humans", loc="left", fontsize=11, fontweight="bold", pad=8)
    _headline(
        fig,
        theme,
        "LLM speakers converge on each other; humans don't",
        "Humans stay flat across a call (Wilcoxon p = 0.89). Most of the agents' extra "
        "alignment comes from their long turns.",
        y=0.99,
    )
    return _save(fig, out, "fig-alignment", theme)


# --- hand-built SVG: hero and architecture diagram -----------------------------------


def _svg(width: int, height: int, body: list[str], title: str, desc: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="t d" '
        f'font-family="{SANS}">\n<title id="t">{escape(title)}</title>\n'
        f'<desc id="d">{escape(desc)}</desc>\n' + "\n".join(body) + "\n</svg>\n"
    )


def _text(x, y, s, size, fill, weight=400, anchor="start", extra="") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}"{extra}>{escape(s)}</text>'
    )


def hero_svg(theme: Theme, out: pathlib.Path, window: int = 300) -> pathlib.Path:
    """The same topic as a timeline of turns: one real call vs four generated versions."""
    data = json.loads(RHYTHM.read_text(encoding="utf-8"))
    stats, listen = summary(), listener_share()
    prompt = data["prompt"]
    rows = [("SB", "Humans", "real call (Switchboard)", theme.human, theme.human_b)] + [
        (f"{a}-{prompt}", a, ARCH_DESC[a], theme.arch[a], theme.arch_b[a]) for a in ARCHS
    ]
    W, H = 1200, 560
    x0, x1 = 300, 860  # timeline
    scale = (x1 - x0) / window
    top, row_h = 168, 62
    col_wpt, col_listen = 975, 1110
    t = theme
    body = [
        _text(40, 58, "Human phone calls are full of listening. LLM calls aren't.", 30, t.ink, 700),
        _text(
            40,
            92,
            f"One topic ({data['topic'].lower()}) as a timeline of turns: the first "
            f"{window} words of a real call and of four LLM-generated versions.",
            17,
            t.muted,
        ),
        _text(col_wpt, top - 26, "words / turn", 14, t.faint, 600, "middle"),
        _text(col_listen, top - 26, "listener turns", 14, t.faint, 600, "middle"),
    ]
    for i, (cond, name, desc, color_a, color_b) in enumerate(rows):
        y = top + i * row_h
        cy = y + 21
        if cond == "SB":
            body.append(_text(40, cy - 2, name, 19, t.ink, 700))
        else:
            body.append(
                f'<rect x="40" y="{cy - 17}" width="12" height="12" rx="3" fill="{color_a}"/>'
            )
            body.append(_text(60, cy - 6, name, 19, t.ink, 700))
            body.append(_text(92, cy - 6, ARCH_NAME[name[:2]], 16, t.ink, 400))
        body.append(_text(40 if cond == "SB" else 60, cy + 17, desc, 13.5, t.muted))
        body.append(
            f'<line x1="{x0}" y1="{cy}" x2="{x1}" y2="{cy}" stroke="{t.grid}" stroke-width="1"/>'
        )
        pos = 0
        listeners = []
        for spk, words, is_listener in data["turns"][cond]:
            if pos >= window:
                break
            start, end = pos, min(pos + words, window)
            pos += words
            if is_listener:
                listeners.append(x0 + (start + (end - start) / 2) * scale)
                continue
            xa = x0 + start * scale + 1
            w = max((end - start) * scale - 2, 3)
            fill = color_a if spk == "A" else color_b
            body.append(
                f'<rect x="{xa:.1f}" y="{cy - 14}" width="{w:.1f}" height="28" rx="4" fill="{fill}"/>'
            )
            if pos > window and words >= 100 and w > 150:  # a long turn running off the edge
                body.append(
                    _text(x1 - 12, cy + 5, f"one {words}-word turn →", 13.5, t.ink, 600, "end")
                )
        for lx in listeners:
            body.append(
                f'<circle cx="{lx:.1f}" cy="{cy}" r="6.5" fill="{t.ink}" stroke="{t.surface}" stroke-width="2.5"/>'
            )
        key = "SB" if cond == "SB" else cond
        wpt = float(stats[key]["mean_words_per_turn_mean"])
        share = 100 * listen[key]
        body.append(_text(col_wpt, cy + 6, f"{wpt:.0f}", 19, t.ink, 600, "middle"))
        body.append(_text(col_listen, cy + 6, f"{share:.0f}%" if share >= 1 else f"{share:.1f}%", 19,
                          t.ink, 600, "middle"))  # fmt: skip
    ly = top + len(rows) * row_h + 26
    body += [
        f'<rect x="40" y="{ly - 11}" width="26" height="14" rx="3" fill="{t.human}"/>',
        f'<rect x="70" y="{ly - 11}" width="26" height="14" rx="3" fill="{t.human_b}"/>',
        _text(104, ly + 1, "a turn (width = words; shade = speaker)", 14, t.muted),
        f'<circle cx="418" cy="{ly - 4}" r="6.5" fill="{t.ink}"/>',
        _text(432, ly + 1, "listener turn: only “uh-huh”, “yeah”, “right”…", 14, t.muted),
        _text(
            40,
            ly + 34,
            f"Right columns: averages over all 50 calls per condition (LLMs: prompt "
            f"{prompt}). Conversation {data['conversation_no']} was chosen as the most typical "
            "topic: closest to each condition's median",
            12.5,
            t.faint,
        ),
        _text(
            40,
            ly + 53,
            "turn length among calls with a clear human listener pattern.",
            12.5,
            t.faint,
        ),
    ]
    desc = (
        "Timelines of turns for one topic. The human call alternates long turns with many tiny "
        "listener turns; LLM conversations have no listener turns, and the two-agent versions "
        "use far longer turns."
    )
    path = out / f"hero-{theme.name}.svg"
    path.write_text(_svg(W, H, body, "Who's listening?", desc), encoding="utf-8")
    return path


def architectures_svg(theme: Theme, out: pathlib.Path) -> pathlib.Path:
    """The four generation architectures as information-flow schematics."""
    t = theme
    W, H = 1200, 470
    card_w, gap, x_start, y0 = 270, 22, 40, 34
    body = []

    def chip(x, y, w, label, color, sub=None):
        parts = [
            f'<rect x="{x}" y="{y}" width="{w}" height="44" rx="10" fill="none" '
            f'stroke="{color}" stroke-width="2"/>',
            _text(x + w / 2, y + (20 if sub else 28), label, 15, t.ink, 600, "middle"),
        ]
        if sub:
            parts.append(_text(x + w / 2, y + 36, sub, 12, t.muted, 400, "middle"))
        return parts

    def transcript(x, y, color_a, color_b, n=5, w=96):
        parts = [
            f'<rect x="{x}" y="{y}" width="{w}" height="{18 + n * 16}" rx="8" fill="none" '
            f'stroke="{t.grid}" stroke-width="1.5"/>'
        ]
        for k in range(n):
            ww = w - 28 - (k * 13) % 30
            fill = color_a if k % 2 == 0 else color_b
            parts.append(
                f'<rect x="{x + 14}" y="{y + 14 + k * 16}" width="{ww}" height="8" rx="4" fill="{fill}"/>'
            )
        return parts

    def arrow(x1, y1, x2, y2, both=False):
        mid = f'marker-end="url(#ah-{t.name})"' + (
            f' marker-start="url(#ah-{t.name})"' if both else ""
        )
        return [
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{t.muted}" stroke-width="1.8" {mid}/>'
        ]

    body.append(
        f'<defs><marker id="ah-{t.name}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
        f'fill="{t.muted}"/></marker></defs>'
    )
    captions = {
        "C1": ["One generation writes the", "entire conversation."],
        "C2": ["One model writes one turn at a", "time and sees both speakers."],
        "C3": ["Two separate chat sessions; each", "sees only its own side of the call."],
        "C4": ["The same two-agent setup, with", "two different models."],
    }
    for k, a in enumerate(ARCHS):
        x = x_start + k * (card_w + gap)
        ca, cb = t.arch[a], t.arch_b[a]
        body.append(
            f'<rect x="{x}" y="{y0}" width="{card_w}" height="352" rx="16" fill="none" '
            f'stroke="{t.grid}" stroke-width="1.5"/>'
        )
        body.append(f'<rect x="{x + 22}" y="{y0 + 25}" width="12" height="12" rx="3" fill="{ca}"/>')
        body.append(_text(x + 42, y0 + 37, a, 19, t.ink, 700))
        body.append(_text(x + 74, y0 + 37, ARCH_NAME[a], 17, t.ink, 400))
        sx, sy = x + 22, y0 + 72
        if a == "C1":
            body += chip(sx + 58, sy, 110, "Vicuna-13B", ca)
            body += arrow(sx + 113, sy + 50, sx + 113, sy + 88)
            body += transcript(sx + 65, sy + 96, ca, cb, n=6)
            body.append(
                _text(sx + 113, sy + 228, "whole call, one pass", 12.5, t.muted, 400, "middle")
            )
        elif a == "C2":
            body += chip(sx + 58, sy, 110, "Vicuna-13B", ca)
            body.append(
                f'<path d="M {sx + 175} {sy + 12} a 16 16 0 1 1 0 20" fill="none" '
                f'stroke="{t.muted}" stroke-width="1.8" marker-end="url(#ah-{t.name})"/>'
            )
            body += arrow(sx + 113, sy + 50, sx + 113, sy + 88)
            body += transcript(sx + 65, sy + 96, ca, cb, n=6)
            body.append(
                _text(
                    sx + 113, sy + 228, "next turn ← full transcript", 12.5, t.muted, 400, "middle"
                )
            )
        else:
            model_b = "Vicuna-13B" if a == "C3" else "Mistral-7B"
            body += chip(sx, sy, 100, "Speaker A", ca, "Vicuna-13B")
            body += chip(sx + 126, sy, 100, "Speaker B", cb, model_b)
            body += arrow(sx + 104, sy + 22, sx + 122, sy + 22, both=True)
            for j, (bx, fill) in enumerate(((sx, ca), (sx + 126, cb))):
                body.append(
                    f'<rect x="{bx}" y="{sy + 66}" width="100" height="118" rx="10" fill="none" '
                    f'stroke="{t.grid}" stroke-width="1.5" stroke-dasharray="4 4"/>'
                )
                body.append(_text(bx + 50, sy + 88, "private view", 12, t.muted, 400, "middle"))
                for m in range(4):
                    own = (m + j) % 2 == 0
                    body.append(
                        f'<rect x="{bx + (14 if own else 30)}" y="{sy + 102 + m * 18}" width="56" '
                        f'height="9" rx="4.5" fill="{fill if own else t.grid}"/>'
                    )
            body.append(
                _text(
                    sx + 113,
                    sy + 228,
                    "own persona card · own context",
                    12.5,
                    t.muted,
                    400,
                    "middle",
                )
            )
        for j, line in enumerate(captions[a]):
            body.append(_text(x + card_w / 2, y0 + 322 + j * 18, line, 13.5, t.ink, 400, "middle"))
    ay = y0 + 352 + 36
    body += arrow(x_start + 10, ay, W - 50, ay)
    body.append(
        _text(
            W / 2, ay + 30, "the two speakers become more independent", 14.5, t.muted, 400, "middle"
        )
    )
    desc = "Four generation architectures, from one model writing the whole call (C1) to two different models talking (C4)."
    path = out / f"architectures-{theme.name}.svg"
    path.write_text(_svg(W, H + 10, body, "Four generation architectures", desc), encoding="utf-8")
    return path


# --- corpus-dependent extraction -------------------------------------------------------


def extract_rhythm(conversation_no: int = 3508, prompt: str = "P1") -> None:
    """Write the hero's turn sequences (word counts and listener flags only; no text)."""
    from analysis.analyze import conversation_turns
    from analysis.evaluate_generated import BACKCHANNEL_STANDALONE_FULL
    from analysis.swda import conversation_no_of, iter_conversation_files, parse_conversation

    def encode(turns):
        return [
            [spk[-1], len(text.split()), bool(BACKCHANNEL_STANDALONE_FULL.match(text.strip()))]
            for spk, text in turns
        ]

    human = next(f for f in iter_conversation_files() if conversation_no_of(f) == conversation_no)
    turns = {"SB": encode(parse_conversation(human))}
    topic = None
    for a in ARCHS:
        rec = json.loads(
            (ROOT / "data" / "generated" / f"{a}-{prompt}" / f"{conversation_no}.json").read_text(
                encoding="utf-8"
            )
        )
        topic = rec["topic"]
        turns[f"{a}-{prompt}"] = encode(conversation_turns(rec))
    RHYTHM.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "conversation_no": conversation_no,
        "topic": topic,
        "prompt": prompt,
        "fields": ["speaker", "words", "listener_only"],
        "note": "Per-turn word counts and listener-only flags; no transcript text.",
        "turns": turns,
    }
    turns_json = ",\n".join(f'  "{k}": {json.dumps(v)}' for k, v in payload.pop("turns").items())
    head = json.dumps(payload, indent=1)[:-2]  # drop the closing "\n}"
    RHYTHM.write_text(f'{head},\n "turns": {{\n{turns_json}\n }}\n}}\n', encoding="utf-8")
    print("wrote", RHYTHM.relative_to(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    ap.add_argument(
        "--extract", action="store_true", help="refresh the hero data (needs the corpus)"
    )
    ap.add_argument("--theme", choices=sorted(THEMES), help="render one theme only")
    args = ap.parse_args()
    if args.extract:
        extract_rhythm()
    args.out.mkdir(parents=True, exist_ok=True)
    themes = [THEMES[args.theme]] if args.theme else list(THEMES.values())
    for theme in themes:
        for render in (
            hero_svg,
            architectures_svg,
            fig_acts,
            fig_distance,
            fig_turn_length,
            fig_alignment,
        ):
            print("wrote", render(theme, args.out))


if __name__ == "__main__":
    main()
