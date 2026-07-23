#!/usr/bin/env python3
"""Overnight experiment queue: distribute (benchmark, system) jobs across GPUs.

Keeps a worker on every GPU busy until the job list is exhausted, continues on
individual failures, logs progress, and collates a master table at the end.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
from _report import METRICS  # noqa: E402

# Heavy diffusion jobs first so both GPUs stay saturated; classical fills the tail.
DIFFUSION = ["gt_diffusion", "gt_diffusion_temporal", "gt_bg",
             "grounded_diffusion", "grounded2_diffusion", "grounded2_bg"]
CLASSICAL = ["naive_copy", "classical_spatial", "gt_flow", "auto_heuristic"]


def ts():
    return time.strftime("%H:%M:%S")


def build_jobs(benchmarks):
    jobs = [(b, s) for b in benchmarks for s in DIFFUSION]
    # temporal-blend ablation on the first benchmark only
    jobs += [(benchmarks[0], "gt_diffusion_t03"),
             (benchmarks[0], "gt_diffusion_t07")]
    jobs += [(b, s) for b in benchmarks for s in CLASSICAL]
    return jobs


def launch(gpu, bench, system, steps, work_size, scorer):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               HF_HOME=os.path.join(ROOT, "weights", "hf"),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    name = os.path.basename(bench.rstrip("/"))
    log_dir = os.path.join(ROOT, "runs", "overnight")
    os.makedirs(log_dir, exist_ok=True)
    log = open(os.path.join(log_dir, f"{name}__{system}.log"), "w")
    p = subprocess.Popen(
        [sys.executable, os.path.join(HERE, "real_worker.py"),
         "--benchmark", bench, "--system", system, "--steps", str(steps),
         "--work-size", str(work_size), "--scorer", scorer],
        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return p, log


def collate(jobs, out_dir):
    lines = ["# Overnight results", ""]
    by_bench = {}
    for bench, system in jobs:
        name = os.path.basename(bench.rstrip("/"))
        rj = os.path.join(ROOT, "runs", name, system, "results.json")
        if os.path.exists(rj):
            by_bench.setdefault(name, {})[system] = json.load(open(rj))["aggregate"]
    for name, systems in by_bench.items():
        lines += [f"## {name}", "", "| system | " + " | ".join(METRICS) + " |",
                  "|" + "---|" * (len(METRICS) + 1)]
        for system, agg in systems.items():
            lines.append("| " + system + " | " + " | ".join(
                "n/a" if agg[m] is None else f"{agg[m]:.3f}" for m in METRICS) + " |")
        lines.append("")
    with open(os.path.join(out_dir, "MASTER.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "MASTER.json"), "w") as f:
        json.dump(by_bench, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", required=True, help="comma-separated roots")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--work-size", type=int, default=512)
    ap.add_argument("--scorer", default="clip_vad")
    args = ap.parse_args()

    benchmarks = args.benchmarks.split(",")
    gpus = [int(g) for g in args.gpus.split(",")]
    jobs = build_jobs(benchmarks)
    out_dir = os.path.join(ROOT, "runs", "overnight")
    os.makedirs(out_dir, exist_ok=True)
    prog = open(os.path.join(out_dir, "progress.log"), "w", buffering=1)

    def note(msg):
        line = f"[{ts()}] {msg}"
        print(line, flush=True)
        prog.write(line + "\n")

    note(f"start: {len(jobs)} jobs on GPUs {gpus}")
    free = list(gpus)
    running = {}
    i = 0
    done = 0
    while i < len(jobs) or running:
        while free and i < len(jobs):
            gpu = free.pop(0)
            bench, system = jobs[i]
            i += 1
            p, log = launch(gpu, bench, system, args.steps, args.work_size, args.scorer)
            running[gpu] = (p, log, bench, system)
            note(f"launch gpu{gpu} {os.path.basename(bench)}/{system} "
                 f"(job {i}/{len(jobs)})")
        for gpu, (p, log, bench, system) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                log.close()
                done += 1
                note(f"done   gpu{gpu} {os.path.basename(bench)}/{system} rc={rc} "
                     f"({done}/{len(jobs)})")
                free.append(gpu)
                del running[gpu]
        time.sleep(10)

    collate(jobs, out_dir)
    note(f"ALL DONE ({done}/{len(jobs)}); master -> runs/overnight/MASTER.md")
    prog.close()


if __name__ == "__main__":
    main()
