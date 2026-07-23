#!/usr/bin/env python3
"""Run two real diffusion systems concurrently, one per GPU, then print both tables."""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
from _report import METRICS, fmt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--work-size", type=int, default=512)
    ap.add_argument("--scorer", default="clip_vad")
    ap.add_argument("--systems", default="gt_diffusion,grounded_diffusion")
    args = ap.parse_args()

    gpus = args.gpus.split(",")
    systems = args.systems.split(",")
    worker = os.path.join(HERE, "real_worker.py")
    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)

    procs = []
    for gpu, system in zip(gpus, systems):
        env = dict(os.environ,
                   CUDA_VISIBLE_DEVICES=gpu,
                   HF_HOME=os.path.join(ROOT, "weights", "hf"),
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        log_path = os.path.join(ROOT, "runs", f"_real_{system}.log")
        log = open(log_path, "w")
        p = subprocess.Popen(
            [sys.executable, worker, "--benchmark", args.benchmark,
             "--system", system, "--steps", str(args.steps),
             "--work-size", str(args.work_size), "--scorer", args.scorer],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        procs.append((system, gpu, p, log, log_path))
        print(f"launched {system} on GPU {gpu} (pid {p.pid}) -> {log_path}")

    for system, gpu, p, log, log_path in procs:
        p.wait()
        log.close()
        print(f"{system} (GPU {gpu}) exit {p.returncode}")

    name = os.path.basename(args.benchmark.rstrip("/"))
    print(f"\n# Real diffusion results — {name}\n")
    print("| system | " + " | ".join(METRICS) + " |")
    print("|" + "---|" * (len(METRICS) + 1))
    for system in systems:
        rj = os.path.join(ROOT, "runs", name, system, "results.json")
        if not os.path.exists(rj):
            print(f"| {system} | (failed — see runs/_real_{system}.log) |")
            continue
        agg = json.load(open(rj))["aggregate"]
        print("| " + system + " | "
              + " | ".join(fmt(agg[m]) for m in METRICS) + " |")


if __name__ == "__main__":
    main()
