"""Plot FL-AIGC experiment results from CSV/JSON files."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


METHOD_LABELS = {
    "no_aigc": "No AIGC",
    "random_incentive": "Random",
    "fixed_price": "Fixed Price",
    "data_size_proportional": "Data Size",
    "proposed_active_set": "Proposed",
    "NoAIGC": "No AIGC",
    "RandomIncentive": "Random",
    "FixedPrice": "Fixed Price",
    "DataSizeProportional": "Data Size",
    "ProposedActiveSet": "Proposed",
}

MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def _set_plot_style():
    """Apply a clean, publication-oriented Matplotlib style."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.15,
            "lines.markersize": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _label(name):
    """Return a display label for method names."""
    return METHOD_LABELS.get(str(name), str(name).replace("_", " ").title())


def _markevery(length: int) -> int:
    """Mark at most about a dozen points per curve."""
    return max(1, int(length) // 12)


def _finish_axes(ax):
    """Apply shared axis styling."""
    ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _read_csv(path):
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path)


def _read_json(path):
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save(fig, output_dir: Path, name: str):
    """Save a figure as PNG and PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _as_list(values):
    return [value for value in values if value is not None]


def plot_accuracy_vs_rounds(fl_frames, output_dir: Path):
    """Plot test accuracy and add a zoomed inset for the late-round region."""
    frames = [df for df in fl_frames if df is not None and {"round", "test_accuracy"}.issubset(df.columns)]
    if not frames:
        return False

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    plotted = []
    for idx, df in enumerate(frames):
        part = df[["round", "test_accuracy"] + (["method"] if "method" in df.columns else [])].dropna(
            subset=["test_accuracy"]
        )
        if part.empty:
            continue
        label = df["method"].iloc[0] if "method" in df.columns else f"run_{idx + 1}"
        line = ax.plot(
            part["round"],
            part["test_accuracy"],
            marker=MARKERS[idx % len(MARKERS)],
            markevery=_markevery(len(part)),
            linewidth=1.15,
            markersize=3.0,
            alpha=0.95,
            label=_label(label),
        )[0]
        plotted.append((part, line, _label(label)))

    if not plotted:
        plt.close(fig)
        return False

    ax.set_title("Accuracy vs Communication Rounds")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Test accuracy")

    all_rounds = pd.concat([part["round"] for part, _, _ in plotted], ignore_index=True)
    if len(all_rounds) >= 20:
        round_min = float(all_rounds.min())
        round_max = float(all_rounds.max())
        zoom_start = round_min + 0.85 * (round_max - round_min)
        zoom_values = []

        axins = inset_axes(ax, width="48%", height="46%", loc="lower left", borderpad=1.15)
        for part, line, _ in plotted:
            zoom_part = part[part["round"] >= zoom_start]
            if zoom_part.empty:
                continue
            axins.plot(
                zoom_part["round"],
                zoom_part["test_accuracy"],
                color=line.get_color(),
                linestyle=line.get_linestyle(),
                marker=line.get_marker(),
                markevery=_markevery(len(zoom_part)),
                linewidth=1.0,
                markersize=2.4,
                alpha=0.95,
            )
            zoom_values.extend(zoom_part["test_accuracy"].tolist())

        if zoom_values:
            y_min = min(zoom_values)
            y_max = max(zoom_values)
            y_pad = max((y_max - y_min) * 0.06, 0.0008)
            axins.set_xlim(zoom_start, round_max)
            axins.set_ylim(y_min - y_pad, y_max + y_pad)
            axins.tick_params(axis="both", labelsize=7)
            axins.grid(True, color="#d9d9d9", linewidth=0.45, alpha=0.65)
            axins.set_title("Late-round zoom", fontsize=8, pad=2)
            for spine in axins.spines.values():
                spine.set_linewidth(0.8)
                spine.set_edgecolor("#555555")
            mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="#555555", linewidth=0.8, alpha=0.95)
        else:
            axins.remove()

    _finish_axes(ax)
    ax.legend(frameon=False, ncol=2)
    _save(fig, output_dir, "accuracy_vs_rounds")
    return True


def plot_loss_vs_rounds(fl_frames, output_dir: Path, loss_column: str, title: str, output_name: str):
    """Plot train or test loss against communication rounds."""
    frames = [df for df in fl_frames if df is not None and {"round", loss_column}.issubset(df.columns)]
    if not frames:
        return False

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    has_data = False
    for idx, df in enumerate(frames):
        part = df[["round", loss_column] + (["method"] if "method" in df.columns else [])].dropna(
            subset=[loss_column]
        )
        if part.empty:
            continue
        has_data = True
        label = part["method"].iloc[0] if "method" in part.columns else f"run_{idx + 1}"
        ax.plot(
            part["round"],
            part[loss_column],
            marker=MARKERS[idx % len(MARKERS)],
            markevery=_markevery(len(part)),
            linewidth=1.15,
            markersize=3.0,
            alpha=0.95,
            label=_label(label),
        )

    if not has_data:
        plt.close(fig)
        return False

    ax.set_title(title)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Loss")
    _finish_axes(ax)
    ax.legend(frameon=False, ncol=2)
    _save(fig, output_dir, output_name)
    return True


def plot_lambda_before_after(fl_frames, output_dir: Path):
    rows = []
    for idx, df in enumerate(fl_frames):
        if df is None or not {"avg_lambda_before", "avg_lambda_after"}.issubset(df.columns):
            continue
        label = df["method"].iloc[0] if "method" in df.columns else f"run_{idx + 1}"
        rows.append(
            {
                "method": label,
                "before": df["avg_lambda_before"].iloc[-1],
                "after": df["avg_lambda_after"].iloc[-1],
            }
        )
    if not rows:
        return False

    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = range(len(data))
    width = 0.35
    ax.bar([value - width / 2 for value in x], data["before"], width=width, label="Before", alpha=0.85)
    ax.bar([value + width / 2 for value in x], data["after"], width=width, label="After", alpha=0.85)
    ax.set_title("Lambda Before vs After AIGC-Proxy")
    ax.set_xlabel("Method")
    ax.set_ylabel("Average TVD lambda")
    ax.set_xticks(list(x))
    ax.set_xticklabels([_label(value) for value in data["method"]], rotation=20, ha="right")
    _finish_axes(ax)
    ax.legend(frameon=False)
    _save(fig, output_dir, "lambda_before_after")
    return True


def plot_q_vs_lambda(client_frames, output_dir: Path):
    frames = [df for df in client_frames if df is not None and {"lambda_k", "q"}.issubset(df.columns)]
    if not frames:
        return False

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for idx, df in enumerate(frames):
        label = (
            df["baseline"].iloc[0]
            if "baseline" in df.columns and df["baseline"].nunique() == 1
            else f"clients_{idx + 1}"
        )
        if "baseline" in df.columns and df["baseline"].nunique() > 1:
            for baseline, part in df.groupby("baseline"):
                ax.scatter(part["lambda_k"], part["q"], s=18, alpha=0.65, label=_label(baseline))
        else:
            ax.scatter(df["lambda_k"], df["q"], s=18, alpha=0.65, label=_label(label))
    ax.set_title("q_k vs lambda_k")
    ax.set_xlabel("lambda_k")
    ax.set_ylabel("q_k")
    _finish_axes(ax)
    ax.legend(frameon=False)
    _save(fig, output_dir, "q_vs_lambda")
    return True


def _summary_rows(summary):
    if summary is None:
        return pd.DataFrame()
    rows = []
    for key, value in summary.items():
        if key == "meta" or not isinstance(value, dict):
            continue
        row = {"method": value.get("baseline", value.get("solver", key))}
        row.update(value)
        if "budget_ratio" not in row and "budget_ratio" in summary.get("meta", {}):
            row["budget_ratio"] = summary["meta"]["budget_ratio"]
        if "num_clients" not in row and "num_clients" in summary.get("meta", {}):
            row["num_clients"] = summary["meta"]["num_clients"]
        rows.append(row)
    return pd.DataFrame(rows)


def _concat_frames(frames):
    """Concatenate non-empty frames, returning an empty frame when none exist."""
    valid_frames = [df for df in frames if df is not None and not df.empty]
    if not valid_frames:
        return pd.DataFrame()
    return pd.concat(valid_frames, ignore_index=True)


def plot_budget_utilization(summary_frames, output_dir: Path):
    data = _concat_frames(summary_frames)
    if data.empty or not {"method", "budget_utilization"}.issubset(data.columns):
        return False

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    data = data.sort_values("method")
    ax.bar([_label(value) for value in data["method"]], data["budget_utilization"], alpha=0.85)
    ax.set_title("Budget Utilization by Method")
    ax.set_xlabel("Method")
    ax.set_ylabel("Budget utilization")
    ax.set_ylim(0, max(1.0, float(data["budget_utilization"].max()) * 1.1))
    ax.tick_params(axis="x", rotation=20)
    _finish_axes(ax)
    _save(fig, output_dir, "budget_utilization")
    return True


def plot_server_utility_comparison(summary_frames, output_dir: Path):
    """Plot server utility by method."""
    data = _concat_frames(summary_frames)
    if data.empty or not {"method", "server_utility"}.issubset(data.columns):
        return False

    data = data.sort_values("server_utility", ascending=False)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar([_label(value) for value in data["method"]], data["server_utility"], alpha=0.85)
    ax.axhline(0.0, color="#666666", linewidth=0.8)
    ax.set_title("Server Utility by Method")
    ax.set_xlabel("Method")
    ax.set_ylabel("Server utility")
    ax.tick_params(axis="x", rotation=20)
    _finish_axes(ax)
    _save(fig, output_dir, "server_utility_comparison")
    return True


def plot_server_utility_vs_budget(sweep_df, summary_frames, output_dir: Path):
    if sweep_df is not None and {"budget_ratio", "server_utility"}.issubset(sweep_df.columns):
        data = sweep_df
    else:
        data = _concat_frames(summary_frames)
    if data.empty or not {"budget_ratio", "server_utility"}.issubset(data.columns):
        return False

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    if "method" in data.columns:
        for idx, (method, part) in enumerate(data.groupby("method")):
            part = part.sort_values("budget_ratio")
            ax.plot(
                part["budget_ratio"],
                part["server_utility"],
                marker=MARKERS[idx % len(MARKERS)],
                linewidth=1.15,
                markersize=3.0,
                label=_label(method),
            )
    else:
        data = data.sort_values("budget_ratio")
        ax.plot(data["budget_ratio"], data["server_utility"], marker="o", linewidth=1.15, markersize=3.0)
    ax.set_title("Server Utility vs Budget Ratio")
    ax.set_xlabel("Budget ratio")
    ax.set_ylabel("Server utility")
    _finish_axes(ax)
    ax.legend(frameon=False)
    _save(fig, output_dir, "server_utility_vs_budget_ratio")
    return True


def plot_runtime_vs_n(sweep_df, summary_frames, output_dir: Path):
    if sweep_df is not None and {"num_clients", "runtime"}.issubset(sweep_df.columns):
        data = sweep_df
    else:
        data = _concat_frames(summary_frames)
    if data.empty or not {"num_clients", "runtime"}.issubset(data.columns):
        return False

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    if "method" in data.columns:
        for idx, (method, part) in enumerate(data.groupby("method")):
            part = part.sort_values("num_clients")
            ax.plot(
                part["num_clients"],
                part["runtime"],
                marker=MARKERS[idx % len(MARKERS)],
                linewidth=1.15,
                markersize=3.0,
                label=_label(method),
            )
    else:
        data = data.sort_values("num_clients")
        ax.plot(data["num_clients"], data["runtime"], marker="o", linewidth=1.15, markersize=3.0)
    ax.set_title("Runtime vs Number of Clients")
    ax.set_xlabel("Number of clients")
    ax.set_ylabel("Runtime (seconds)")
    _finish_axes(ax)
    ax.legend(frameon=False)
    _save(fig, output_dir, "runtime_vs_n")
    return True


def plot_optimality_gap(sweep_df, summary_frames, output_dir: Path):
    if sweep_df is not None and {"num_clients", "optimality_gap"}.issubset(sweep_df.columns):
        data = sweep_df
    else:
        combined = _concat_frames(summary_frames)
        if combined.empty or not {"method", "server_utility", "num_clients"}.issubset(combined.columns):
            return False
        exact = combined[combined["method"].str.contains("exact", case=False, na=False)]
        if exact.empty:
            return False
        exact_utility = float(exact["server_utility"].max())
        data = combined.copy()
        data["optimality_gap"] = exact_utility - data["server_utility"]
    data = data[data["num_clients"] <= 12]
    if data.empty:
        return False

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    if "method" in data.columns:
        for idx, (method, part) in enumerate(data.groupby("method")):
            part = part.sort_values("num_clients")
            ax.plot(
                part["num_clients"],
                part["optimality_gap"],
                marker=MARKERS[idx % len(MARKERS)],
                linewidth=1.15,
                markersize=3.0,
                label=_label(method),
            )
    else:
        data = data.sort_values("num_clients")
        ax.plot(data["num_clients"], data["optimality_gap"], marker="o", linewidth=1.15, markersize=3.0)
    ax.set_title("Optimality Gap for N <= 12")
    ax.set_xlabel("Number of clients")
    ax.set_ylabel("Exact utility - method utility")
    _finish_axes(ax)
    ax.legend(frameon=False)
    _save(fig, output_dir, "optimality_gap_n_le_12")
    return True


def main():
    _set_plot_style()
    parser = argparse.ArgumentParser(description="Plot FL-AIGC experiment results")
    parser.add_argument("--fl-csv", action="append", default=[], help="FL round CSV; can be repeated")
    parser.add_argument("--mechanism-csv", action="append", default=[], help="Mechanism client CSV; can be repeated")
    parser.add_argument("--baseline-csv", action="append", default=[], help="Baseline client CSV; can be repeated")
    parser.add_argument("--mechanism-json", action="append", default=[], help="Mechanism summary JSON; can be repeated")
    parser.add_argument("--baseline-json", action="append", default=[], help="Baseline summary JSON; can be repeated")
    parser.add_argument("--sweep-csv", default=None, help="Optional sweep CSV with budget_ratio/runtime/gap columns")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for PNG/PDF figures")
    args = parser.parse_args()

    fl_frames = [_read_csv(path) for path in args.fl_csv]
    client_frames = [_read_csv(path) for path in args.mechanism_csv + args.baseline_csv]
    summaries = [_read_json(path) for path in args.mechanism_json + args.baseline_json]
    summary_frames = [_summary_rows(summary) for summary in summaries]
    sweep_df = _read_csv(args.sweep_csv)

    plotted = {
        "accuracy_vs_rounds": plot_accuracy_vs_rounds(fl_frames, args.output_dir),
        "test_loss_vs_rounds": plot_loss_vs_rounds(
            fl_frames,
            args.output_dir,
            "test_loss",
            "Test Loss vs Communication Rounds",
            "test_loss_vs_rounds",
        ),
        "train_loss_vs_rounds": plot_loss_vs_rounds(
            fl_frames,
            args.output_dir,
            "train_loss",
            "Train Loss vs Communication Rounds",
            "train_loss_vs_rounds",
        ),
        "server_utility_vs_budget_ratio": plot_server_utility_vs_budget(sweep_df, summary_frames, args.output_dir),
        "q_vs_lambda": plot_q_vs_lambda(client_frames, args.output_dir),
        "runtime_vs_n": plot_runtime_vs_n(sweep_df, summary_frames, args.output_dir),
        "optimality_gap_n_le_12": plot_optimality_gap(sweep_df, summary_frames, args.output_dir),
        "lambda_before_after": plot_lambda_before_after(fl_frames, args.output_dir),
        "budget_utilization": plot_budget_utilization(summary_frames, args.output_dir),
        "server_utility_comparison": plot_server_utility_comparison(summary_frames, args.output_dir),
    }

    for name, did_plot in plotted.items():
        status = "created" if did_plot else "skipped"
        print(f"{status}: {name}")


if __name__ == "__main__":
    main()
