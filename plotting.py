"""Rendering for run_pipeline.py's recorded metrics.

Not part of the rlhf package on purpose: this is report tooling for the
one experiment run in this repository, not a reusable algorithmic
building block. Colors and chart forms follow a validated categorical
palette (fixed hue order, no chart ever uses two y axes on one plot;
metrics of different scale get their own small multiple panel instead).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GOOD = "#0ca30c"

SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]


def _style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.title.set_color(PRIMARY_INK)
    ax.xaxis.label.set_color(SECONDARY_INK)
    ax.yaxis.label.set_color(SECONDARY_INK)


def _new_figure(ncols=1, figsize_per_panel=(5.5, 3.6)):
    fig, axes = plt.subplots(
        1, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1]),
        facecolor=SURFACE,
    )
    if ncols == 1:
        axes = [axes]
    for ax in axes:
        _style_axis(ax)
    return fig, axes


def _line(ax, values, color=BLUE, label=None, x=None):
    x = list(range(len(values))) if x is None else x
    ax.plot(x, values, color=color, linewidth=2.2, solid_capstyle="round", label=label, zorder=3)


def plot_sft_loss(train_losses, val_loss, out_path):
    fig, (ax,) = _new_figure(1)
    _line(ax, train_losses, color=BLUE)
    ax.axhline(val_loss, color=MUTED, linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
    y_span = max(train_losses) - min(train_losses + [val_loss])
    ax.text(len(train_losses) * 0.5, val_loss + 0.04 * y_span, f"val loss {val_loss:.2f}",
            color=SECONDARY_INK, fontsize=9, va="bottom", ha="center",
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5})
    ax.set_title("Supervised finetuning loss", fontsize=12, weight="bold")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross entropy loss")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def plot_lora_loss(losses, out_path):
    fig, (ax,) = _new_figure(1)
    _line(ax, losses, color=BLUE)
    ax.set_title("LoRA adapter loss (hand rolled AdamW)", fontsize=12, weight="bold")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross entropy loss")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def plot_reward_model(losses, accuracies, out_path):
    fig, (ax_loss, ax_acc) = _new_figure(2)
    _line(ax_loss, losses, color=BLUE)
    ax_loss.set_title("Reward model loss", fontsize=12, weight="bold")
    ax_loss.set_xlabel("optimizer step")
    ax_loss.set_ylabel("pairwise loss")

    _line(ax_acc, [a * 100 for a in accuracies], color=BLUE)
    ax_acc.set_ylim(0, 105)
    ax_acc.axhline(50, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax_acc.text(0, 52, "chance", color=SECONDARY_INK, fontsize=9, va="bottom")
    ax_acc.set_title("Reward model pairwise accuracy", fontsize=12, weight="bold")
    ax_acc.set_xlabel("optimizer step")
    ax_acc.set_ylabel("percent correctly ranked")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def plot_ppo_training(policy_losses, value_losses, entropies, out_path):
    fig, (ax_p, ax_v, ax_e) = _new_figure(3, figsize_per_panel=(4.3, 3.6))
    _line(ax_p, policy_losses, color=BLUE)
    ax_p.set_title("Policy loss", fontsize=11, weight="bold")
    ax_p.set_xlabel("update")

    _line(ax_v, value_losses, color=BLUE)
    ax_v.set_title("Value loss", fontsize=11, weight="bold")
    ax_v.set_xlabel("update")

    _line(ax_e, entropies, color=BLUE)
    ax_e.set_title("Policy entropy", fontsize=11, weight="bold")
    ax_e.set_xlabel("update")

    fig.suptitle("PPO training signal", fontsize=13, weight="bold", color=PRIMARY_INK, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_dpo_training(losses, margins, out_path):
    fig, (ax_l, ax_m) = _new_figure(2)
    _line(ax_l, losses, color=BLUE)
    ax_l.axhline(0.693, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax_l.text(0, 0.71, "chance = ln(2)", color=SECONDARY_INK, fontsize=9, va="bottom")
    ax_l.set_title("DPO loss", fontsize=12, weight="bold")
    ax_l.set_xlabel("optimizer step")
    ax_l.set_ylabel("loss")

    _line(ax_m, margins, color=BLUE)
    ax_m.axhline(0, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax_m.set_title("Preference margin, policy vs reference", fontsize=12, weight="bold")
    ax_m.set_xlabel("optimizer step")
    ax_m.set_ylabel("(chosen minus rejected) log ratio gap")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def plot_preference_loss_comparison(loss_by_method, out_path):
    fig, (ax,) = _new_figure(1, figsize_per_panel=(6.5, 4.0))
    methods = list(loss_by_method.keys())
    values = [loss_by_method[m] for m in methods]
    colors = CATEGORICAL[: len(methods)]

    bars = ax.bar(methods, values, color=colors, width=0.6, zorder=3)
    ax.set_yscale("log")
    ax.set_title("Preference loss formulas on the same final batch", fontsize=12, weight="bold")
    ax.set_ylabel("loss (log scale)")

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.15, f"{value:.3g}",
                ha="center", va="bottom", fontsize=9, color=PRIMARY_INK)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def plot_win_rate_comparison(win_rates, out_path):
    fig, (ax,) = _new_figure(1, figsize_per_panel=(6.0, 4.0))
    labels = list(win_rates.keys())
    values = [win_rates[k] * 100 for k in labels]
    colors = CATEGORICAL[: len(labels)]

    bars = ax.bar(labels, values, color=colors, width=0.5, zorder=3)
    ax.axhline(50, color=MUTED, linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
    ax.text(len(labels) - 0.5, 52, "chance (50%)", color=SECONDARY_INK, fontsize=9, ha="right", va="bottom")
    ax.set_ylim(0, 100)
    ax.set_title("Aligned model win rate vs base\n(reward model judged)", fontsize=12, weight="bold")
    ax.set_ylabel("win rate, percent (ties count as half)")

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.0f}%",
                ha="center", va="bottom", fontsize=10, color=PRIMARY_INK, weight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor=SURFACE)
    plt.close(fig)
