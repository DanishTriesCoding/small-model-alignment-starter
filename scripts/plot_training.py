##making my training plots for the report from the metrics files my training scripts already wrote
"""
some commands for me
    python scripts/plot_training.py \
        --sft_dir /kaggle/input/<sft-output>/outputs/sft \
        --dpo_hh_dir /kaggle/input/<dpo-hh-output>/outputs/dpo_hh \
        --dpo_uf_dir /kaggle/input/<dpo-uf-output>/outputs/dpo_uf \
        --persona_dir /kaggle/input/<persona-output>/outputs/persona_dpo \
        --output_dir /kaggle/working/outputs/plots

    # any run can be left out, that plot just gets skipped
    # each dir is the run's output_dir: SFT/DPO need metrics.jsonl (+ config.json), persona needs trainer_state.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")  ##no screen on kaggle, just saving files
import matplotlib.pyplot as plt

TRAIN_COLOR = "#2a78d6"  ##blue = train everywhere
VAL_COLOR = "#eb6834"  ##orange = validation everywhere, same colors in every figure so the report reads consistently
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e7e6e2"
AXIS_COLOR = "#c9c8c2"
REFERENCE_COLOR = "#8a8984"  ##gray for the best-checkpoint line + 0.5 / 0 reference lines

DPO_METRICS = [
    ("loss", "DPO loss"),
    ("reward_accuracy", "reward accuracy"),
    ("reward_margin", "reward margin (r_w - r_l)"),
]
##3 panels per DPO run, manual 14.4 wants DPO loss + at least one preference diagnostic, im doing both


def parse_args():
    parser = argparse.ArgumentParser()
    sft_dir_argument = parser.add_argument("--sft_dir", default=None, help="SFT output_dir w/ metrics.jsonl")
    dpo_hh_dir_argument = parser.add_argument("--dpo_hh_dir", default=None, help="HH-RLHF DPO output_dir w/ metrics.jsonl")
    dpo_uf_dir_argument = parser.add_argument("--dpo_uf_dir", default=None, help="UltraFeedback DPO output_dir w/ metrics.jsonl")
    persona_dir_argument = parser.add_argument("--persona_dir", default=None, help="persona LoRA output_dir w/ trainer_state.json")
    output_dir_argument = parser.add_argument("--output_dir", default="outputs/plots", help="where the png + pdf files go")
    ##all runs optional so i can plot whatever is attached in the notebook
    parsed_arguments = parser.parse_args()
    return parsed_arguments


##reading my own metrics.jsonl (one dict per line from log_metrics in train_sft.py / train_dpo.py)
def read_metrics_jsonl(run_dir, run_config):
    metrics_path = run_dir if run_dir.endswith(".jsonl") else os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(metrics_path):
        raise ValueError(f"No metrics.jsonl found at '{metrics_path}'.")
    with open(metrics_path, "r", encoding="utf-8") as metrics_file:
        all_records = [json.loads(one_line) for one_line in metrics_file if one_line.strip() != ""]
    ##log_metrics APPENDS, so an earlier run (smoke test / crashed try) into the same folder leaves old lines in here

    last_restart_index = 0
    previous_step_by_split = {}
    previous_elapsed_minutes = None
    for record_index, one_record in enumerate(all_records):
        record_split = one_record["split"]
        step_went_backwards = one_record["step"] <= previous_step_by_split.get(record_split, 0)
        clock_went_backwards = (record_split == "train" and "elapsed_min" in one_record
                                and previous_elapsed_minutes is not None
                                and one_record["elapsed_min"] < previous_elapsed_minutes)
        if step_went_backwards or clock_went_backwards:
            last_restart_index = record_index
            previous_step_by_split = {}
        previous_step_by_split[record_split] = one_record["step"]
        if record_split == "train" and "elapsed_min" in one_record:
            previous_elapsed_minutes = one_record["elapsed_min"]
    if last_restart_index > 0:
        print(f"'{metrics_path}': a new run starts at line {last_restart_index + 1}, only plotting that last run")
        all_records = all_records[last_restart_index:]
    ##catches a 2nd run that started over (steps or elapsed time going back to small numbers)

    log_every = run_config.get("log_every")
    eval_every = run_config.get("eval_every")
    last_val_step = max((one_record["step"] for one_record in all_records if one_record["split"] == "val"), default=None)
    kept_records = []
    for one_record in all_records:
        if one_record["split"] == "train" and log_every and one_record["step"] % log_every != 0:
            continue
        if one_record["split"] == "val" and eval_every and one_record["step"] % eval_every != 0 \
                and one_record["step"] != last_val_step:
            continue
        kept_records.append(one_record)
    ##a smoke run logged BEFORE the real one (steps 1-4) looks like a normal start, but its steps arent multiples of
    ##the real run's log_every / eval_every from config.json, so those lines get dropped (final val step is always kept)
    number_of_dropped_lines = len(all_records) - len(kept_records)
    if number_of_dropped_lines > 0:
        print(f"'{metrics_path}': dropped {number_of_dropped_lines} lines that dont match log_every/eval_every (old smoke run?)")

    train_records = [one_record for one_record in kept_records if one_record["split"] == "train"]
    val_records = [one_record for one_record in kept_records if one_record["split"] == "val"]
    return train_records, val_records


##config.json is saved by train_sft.py / train_dpo.py so labels can say the real values instead of guesses
def read_config(run_dir):
    config_path = os.path.join(run_dir, "config.json") if not run_dir.endswith(".jsonl") else None
    if config_path is None or not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


##best checkpoint = last val line where my training script said "best": true, thats exactly when best/ got overwritten
def find_best_step(val_records):
    best_steps = [one_record["step"] for one_record in val_records if one_record.get("best")]
    if not best_steps:
        return None
    return best_steps[-1]


def column(records, key):
    steps = [one_record["step"] for one_record in records if one_record.get(key) is not None]
    values = [one_record[key] for one_record in records if one_record.get(key) is not None]
    return steps, values


##same look on every panel: light y grid, no top/right box, gray ticks so the lines are what stands out
def style_axis(axis):
    axis.set_facecolor("white")
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(AXIS_COLOR)
    axis.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    axis.xaxis.label.set_color(TEXT_SECONDARY)
    axis.yaxis.label.set_color(TEXT_SECONDARY)
    axis.xaxis.label.set_size(9)
    axis.yaxis.label.set_size(9)


def mark_best_step(axis, best_step):
    if best_step is None:
        return None
    best_line = axis.axvline(best_step, color=REFERENCE_COLOR, linestyle="--", linewidth=1.0, zorder=1)
    return best_line


def save_figure(figure, output_dir, file_stem):
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f"{file_stem}.png")
    pdf_path = os.path.join(output_dir, f"{file_stem}.pdf")
    figure.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")  ##pdf stays sharp if the report is latex
    plt.close(figure)
    print(f"saved {png_path} (+ .pdf)")


##SFT: train + val loss on one axis (same units so one axis is fine) + where best/ came from
def plot_sft(run_dir, output_dir):
    run_config = read_config(run_dir)
    train_records, val_records = read_metrics_jsonl(run_dir, run_config)
    log_every = run_config.get("log_every", 10)
    val_size = run_config.get("val_size", None)
    train_size = run_config.get("train_size", None)
    model_name = run_config.get("model_name", "Qwen/Qwen2.5-0.5B")

    figure, axis = plt.subplots(figsize=(6.2, 3.2))
    style_axis(axis)
    train_steps, train_losses = column(train_records, "loss")
    val_steps, val_losses = column(val_records, "loss")
    axis.plot(train_steps, train_losses, color=TRAIN_COLOR, linewidth=1.4,
              label=f"train (mean over each {log_every}-step log window)")
    val_label = f"validation ({val_size:,} examples)" if val_size else "validation"
    axis.plot(val_steps, val_losses, color=VAL_COLOR, linewidth=1.8, marker="o", markersize=4.5, label=val_label)
    ##train is already averaged per log window by train_sft.py, so no extra smoothing on top

    best_step = find_best_step(val_records)
    best_line = mark_best_step(axis, best_step)
    if best_line is not None:
        best_line.set_label(f"best checkpoint (step {best_step})")

    axis.set_xlabel("optimizer step")
    axis.set_ylabel("loss (per response token)")
    title_size_text = f", {train_size:,} examples" if train_size else ""
    axis.set_title(f"SFT: {model_name.split('/')[-1]}{title_size_text}", color=TEXT_PRIMARY, fontsize=10, loc="left")
    axis.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY)
    save_figure(figure, output_dir, "sft_loss")

    best_val_record = next((one_record for one_record in val_records if one_record["step"] == best_step), None)
    sft_summary = {
        "last_logged_train_step": train_steps[-1] if train_steps else None,  ##NOT total steps, real total is in the training log (937 for my run)
        "best_step": best_step,
        "best_val_loss": best_val_record["loss"] if best_val_record else None,
        "last_val_step": val_steps[-1] if val_steps else None,
        "last_val_loss": val_losses[-1] if val_losses else None,
    }
    return sft_summary


##turning my DPO metrics.jsonl into the same shape the persona TRL logs get turned into, so one plot func does both
def load_my_dpo_run(run_dir, run_name):
    run_config = read_config(run_dir)
    train_records, val_records = read_metrics_jsonl(run_dir, run_config)
    log_every = run_config.get("log_every", 10)
    val_size = run_config.get("val_size", None)
    dpo_run = {
        "name": run_name,
        "train_records": train_records,
        "val_records": val_records,
        "best_step": find_best_step(val_records),  ##train_dpo.py marks best by val reward accuracy
        "train_label": f"train (mean over each {log_every}-step log window)",
        "val_label": f"validation ({val_size:,} pairs)" if val_size else "validation",
        "best_label": "best checkpoint (val reward accuracy)",
        "skipped_optimizer_steps": sum(1 for one_record in train_records if one_record.get("optimizer_step_skipped")),
    }
    return dpo_run


##TRL saves everything in trainer_state.json -> log_history, w/ its own key names
def load_persona_run(run_dir):
    state_path = run_dir if run_dir.endswith(".json") else os.path.join(run_dir, "trainer_state.json")
    if not os.path.exists(state_path):
        checkpoint_state_paths = sorted(
            glob.glob(os.path.join(run_dir, "checkpoint-*", "trainer_state.json")),
            key=lambda one_path: int(one_path.split("checkpoint-")[-1].split(os.sep)[0].split("/")[0]),
        )
        if not checkpoint_state_paths:
            raise ValueError(f"No trainer_state.json found in '{run_dir}' or its checkpoint-* folders.")
        state_path = checkpoint_state_paths[-1]  ##latest checkpoint has the longest history
    with open(state_path, "r", encoding="utf-8") as state_file:
        log_history = json.load(state_file)["log_history"]

    train_records = []
    val_records_by_step = {}
    for one_entry in log_history:
        if "loss" in one_entry and "train_runtime" not in one_entry:
            train_records.append({
                "step": one_entry["step"],
                "loss": one_entry.get("loss"),
                "reward_accuracy": one_entry.get("rewards/accuracies"),
                "reward_margin": one_entry.get("rewards/margins"),
            })
        if "eval_loss" in one_entry:
            val_records_by_step[one_entry["step"]] = {
                "step": one_entry["step"],
                "loss": one_entry.get("eval_loss"),
                "reward_accuracy": one_entry.get("eval_rewards/accuracies"),
                "reward_margin": one_entry.get("eval_rewards/margins"),
            }
            ##my script calls trainer.evaluate() again after training -> same step logged twice, dict keeps only one
    val_records = [val_records_by_step[one_step] for one_step in sorted(val_records_by_step)]
    persona_run = {
        "name": "Persona LoRA-DPO (TRL)",
        "train_records": train_records,
        "val_records": val_records,
        "best_step": None,  ##TRL run saves the FINAL adapter, theres no best-by-val pick to mark
        "train_label": "train (TRL logging window)",
        "val_label": "validation (held-out split)",
        "best_label": None,
        "skipped_optimizer_steps": None,
    }
    return persona_run


##one row per run, 3 columns (loss / reward accuracy / reward margin), separate small panels instead of a 2-axis mess
def plot_dpo_runs(dpo_runs, output_dir, file_stem):
    number_of_rows = len(dpo_runs)
    figure, axes = plt.subplots(number_of_rows, 3, figsize=(10.5, 2.7 * number_of_rows), squeeze=False)
    legend_handles = {}
    for row_index, dpo_run in enumerate(dpo_runs):
        for column_index, (metric_key, metric_label) in enumerate(DPO_METRICS):
            axis = axes[row_index][column_index]
            style_axis(axis)
            train_steps, train_values = column(dpo_run["train_records"], metric_key)
            val_steps, val_values = column(dpo_run["val_records"], metric_key)
            train_line, = axis.plot(train_steps, train_values, color=TRAIN_COLOR, linewidth=1.4)
            val_line, = axis.plot(val_steps, val_values, color=VAL_COLOR, linewidth=1.8, marker="o", markersize=4.5)
            legend_handles.setdefault(dpo_run["train_label"], train_line)
            legend_handles.setdefault(dpo_run["val_label"], val_line)

            if metric_key == "reward_accuracy":
                axis.axhline(0.5, color=REFERENCE_COLOR, linestyle=":", linewidth=0.9, zorder=1)
                axis.set_ylim(-0.03, 1.05)  ##little headroom so a 1.0 accuracy line isnt cut by the frame
                ##0.5 = no consistent preference direction (manual 10.5), above it = learning the right way
            if metric_key == "reward_margin":
                axis.axhline(0.0, color=REFERENCE_COLOR, linestyle=":", linewidth=0.9, zorder=1)
            best_line = mark_best_step(axis, dpo_run["best_step"])
            if best_line is not None and dpo_run["best_label"]:
                legend_handles.setdefault(dpo_run["best_label"], best_line)

            axis.set_ylabel(metric_label)
            if row_index == number_of_rows - 1:
                axis.set_xlabel("optimizer step")
            if column_index == 0:
                best_step_text = f"  (best checkpoint: step {dpo_run['best_step']})" if dpo_run["best_step"] is not None else ""
                axis.set_title(dpo_run["name"] + best_step_text, color=TEXT_PRIMARY, fontsize=10, loc="left")

    figure.tight_layout()
    figure.legend(list(legend_handles.values()), list(legend_handles.keys()), loc="lower center",
                  bbox_to_anchor=(0.5, 1.0), ncol=min(len(legend_handles), 4), frameon=False,
                  fontsize=8, labelcolor=TEXT_SECONDARY)
    ##one shared legend sitting right above the panels instead of repeating it in 6 panels (bbox_inches="tight" keeps it in the file)
    save_figure(figure, output_dir, file_stem)


def summarize_dpo_run(dpo_run):
    val_records = dpo_run["val_records"]
    best_step = dpo_run["best_step"]
    best_val_record = next((one_record for one_record in val_records if one_record["step"] == best_step), None)
    last_val_record = val_records[-1] if val_records else None
    last_train_step = dpo_run["train_records"][-1]["step"] if dpo_run["train_records"] else None
    run_summary = {
        "last_logged_step": last_train_step,
        "best_step": best_step,
        "best_val": {key: best_val_record.get(key) for key, _ in DPO_METRICS} if best_val_record else None,
        "last_val_step": last_val_record["step"] if last_val_record else None,
        "last_val": {key: last_val_record.get(key) for key, _ in DPO_METRICS} if last_val_record else None,
        "skipped_optimizer_steps_in_logs": dpo_run["skipped_optimizer_steps"],
    }
    ##numbers for the report tables so i dont have to read them off the plots
    return run_summary


def main():
    args = parse_args()
    training_summary = {}

    if args.sft_dir:
        training_summary["sft"] = plot_sft(args.sft_dir, args.output_dir)

    my_dpo_runs = []
    if args.dpo_hh_dir:
        my_dpo_runs.append(("dpo_hh", load_my_dpo_run(args.dpo_hh_dir, "HH-RLHF DPO (from scratch)")))
    if args.dpo_uf_dir:
        my_dpo_runs.append(("dpo_uf", load_my_dpo_run(args.dpo_uf_dir, "UltraFeedback DPO (from scratch)")))
    if my_dpo_runs:
        plot_dpo_runs([one_run for _, one_run in my_dpo_runs], args.output_dir, "dpo_training")
        for run_key, one_run in my_dpo_runs:
            training_summary[run_key] = summarize_dpo_run(one_run)

    if args.persona_dir:
        persona_run = load_persona_run(args.persona_dir)
        plot_dpo_runs([persona_run], args.output_dir, "persona_dpo_training")
        training_summary["persona_lora"] = summarize_dpo_run(persona_run)

    if not training_summary:
        raise ValueError("Give at least one of --sft_dir / --dpo_hh_dir / --dpo_uf_dir / --persona_dir.")
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(training_summary, summary_file, indent=2)
    print(json.dumps(training_summary, indent=2))
    print(f"summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
