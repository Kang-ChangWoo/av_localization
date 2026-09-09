#!/usr/bin/env python3
"""Run a queue of visual training jobs across a pool of GPUs.

Jobs are described in a YAML file (see ``configs/queue_visual.yaml``). Each job
asks for ``devices`` GPUs and may declare ``after`` dependencies; the scheduler
starts a job as soon as its dependencies have succeeded and enough GPUs are
free, so the wide reproduction runs and the narrow sweep runs share one pool.

State is written continuously to ``outputs/queue_status.json`` so progress can
be inspected (or the queue resumed) without attaching to the process.

    python scripts/run_queue.py --jobs configs/queue_visual.yaml
    python scripts/run_queue.py --jobs configs/queue_visual.yaml --status
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Job:
    def __init__(self, spec: dict, defaults: dict):
        merged = {**defaults, **spec}
        self.name: str = merged["name"]
        self.net: str = merged["net"]
        self.devices: int = int(merged.get("devices", 1))
        self.after: list[str] = list(merged.get("after", []))
        self.config: str | None = merged.get("config")
        self.args: dict = dict(merged.get("args", {}))
        self.eval_datasets: list[str] = list(merged.get("eval", []))
        # "eval": BatchNorm uses running statistics (correct inference).
        # "upstream": reproduces eval_observation.py, which leaves mono/mv in
        # train mode; that is the protocol the published numbers were measured
        # with, so it is the one to compare against the paper.
        self.eval_bn_modes: list[str] = list(merged.get("eval_bn_modes", ["eval"]))
        self.eval_dataset_root: str | None = merged.get("eval_dataset_root")
        # "train" jobs fit a checkpoint; "eval" jobs score one that already
        # exists. Evaluation is scheduled like any other job so it runs on a
        # free GPU instead of blocking the scheduler.
        self.kind: str = merged.get("kind", "train")
        self.eval_of: str | None = merged.get("eval_of")
        self.status = "queued"
        self.gpus: list[str] = []
        self.started: str | None = None
        self.finished: str | None = None
        self.returncode: int | None = None
        self.metrics: dict = {}
        self.process: subprocess.Popen | None = None
        self.log_path: Path | None = None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "net": self.net,
            "status": self.status,
            "devices": self.devices,
            "gpus": self.gpus,
            "started": self.started,
            "finished": self.finished,
            "returncode": self.returncode,
            "metrics": self.metrics,
            "log": str(self.log_path) if self.log_path else None,
        }

    def command(self, gpus: list[str]) -> list[str]:
        if self.kind == "eval":
            return [
                PYTHON, "scripts/eval_job.py",
                "--run-name", self.eval_of or self.name,
                "--net", self.net,
                "--gpu", gpus[0],
                "--datasets", *self.eval_datasets,
                "--bn-modes", *self.eval_bn_modes,
                "--skip-existing",
            ] + (["--dataset-root", self.eval_dataset_root] if self.eval_dataset_root else [])
        cmd = [PYTHON, "scripts/train_visual.py", "--net", self.net, "--gpus", ",".join(gpus)]
        if self.config:
            cmd += ["--config", self.config]
        cmd += ["--run-name", self.name]
        for key, value in self.args.items():
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
            elif key == "set":
                # --set is an append flag: one occurrence per override
                for item in value:
                    cmd += [flag, str(item)]
            elif isinstance(value, (list, tuple)):
                cmd += [flag] + [str(v) for v in value]
            else:
                cmd += [flag, str(value)]
        return cmd


def metrics_filename(job_name: str, dataset: str, bn_mode: str) -> str:
    suffix = "" if bn_mode == "eval" else f"_{bn_mode}"
    return f"{job_name}_{dataset}{suffix}.json"


def collect_metrics(job: Job, gpu: str, out_root: Path, log_dir: Path) -> dict:
    """Evaluate a finished job and return its recall metrics."""
    metrics = {}
    ckpt = out_root / job.name / f"{job.net}.ckpt"
    if not ckpt.exists():
        return metrics
    for dataset in job.eval_datasets:
        for bn_mode in job.eval_bn_modes:
            key = dataset if bn_mode == "eval" else f"{dataset}:{bn_mode}"
            out_json = out_root / "metrics" / metrics_filename(job.name, dataset, bn_mode)
            log_path = log_dir / f"eval_{job.name}_{dataset}_{bn_mode}.log"
            cmd = [
                PYTHON, "scripts/eval_visual.py",
                "--net", job.net,
                "--dataset", dataset,
                "--ckpt", str(ckpt),
                "--gpu", gpu,
                "--bn-mode", bn_mode,
                "--out", str(out_json),
            ]
            # upstream's protocol is defined at batch size 1
            if bn_mode == "upstream":
                cmd += ["--batch-size", "1"]
            with open(log_path, "w") as handle:
                subprocess.run(cmd, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT)
            if out_json.exists():
                metrics[key] = json.loads(out_json.read_text())
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=Path, default=REPO_ROOT / "configs" / "queue_visual.yaml")
    parser.add_argument("--gpus", default=None, help="override the GPU pool, e.g. '1,2,3'")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--status", action="store_true", help="print queue_status.json and exit")
    parser.add_argument("--poll", type=int, default=20, help="scheduler poll interval in seconds")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="treat a job as done when its checkpoint already exists, so an "
        "interrupted queue resumes instead of retraining what finished",
    )
    args = parser.parse_args()
    # the queue is meant to be tailed while it runs, so don't block-buffer
    sys.stdout.reconfigure(line_buffering=True)

    out_root = args.output_dir
    status_path = out_root / "queue_status.json"
    if args.status:
        print(status_path.read_text() if status_path.exists() else "{}")
        return 0

    spec = yaml.safe_load(args.jobs.read_text())
    pool = [g.strip() for g in (args.gpus or spec.get("gpus", "1,2,3,4,5,6,7")).split(",")]
    defaults = spec.get("defaults", {})
    jobs = [Job(job_spec, defaults) for job_spec in spec["jobs"]]
    by_name = {job.name: job for job in jobs}

    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (out_root / "metrics").mkdir(parents=True, exist_ok=True)

    free_gpus = list(pool)
    running: list[Job] = []

    def write_status() -> None:
        status_path.write_text(
            json.dumps(
                {
                    "updated": now(),
                    "pool": pool,
                    "free_gpus": free_gpus,
                    "jobs": [job.as_dict() for job in jobs],
                },
                indent=2,
            )
        )

    def queue_eval_job(train_job: Job) -> None:
        name = f"{train_job.name}:eval"
        if name in by_name:
            return
        ev = Job(
            {
                "name": name,
                "net": train_job.net,
                "kind": "eval",
                "eval_of": train_job.name,
                "devices": 1,
                "eval": train_job.eval_datasets,
                "eval_bn_modes": train_job.eval_bn_modes,
                "eval_dataset_root": train_job.eval_dataset_root,
            },
            {},
        )
        # evaluations are short and one-GPU: put them at the front so results
        # land while the long training jobs are still running
        jobs.insert(0, ev)
        by_name[name] = ev
        print(f"[queue] queued evaluation for {train_job.name}")

    def load_metrics(eval_job: Job) -> None:
        target = by_name.get(eval_job.eval_of or "")
        for dataset in eval_job.eval_datasets:
            for bn_mode in eval_job.eval_bn_modes:
                key = dataset if bn_mode == "eval" else f"{dataset}:{bn_mode}"
                path = out_root / "metrics" / metrics_filename(eval_job.eval_of, dataset, bn_mode)
                if not path.exists():
                    continue
                m = json.loads(path.read_text())
                eval_job.metrics[key] = m
                if target is not None:
                    target.metrics[key] = m
                print(f"[queue]   {eval_job.eval_of} {key}: 1m={m['recall_1m']:.3f} "
                      f"0.5m={m['recall_0.5m']:.3f} 0.1m={m['recall_0.1m']:.3f}")

    def deps_ok(job: Job) -> bool:
        return all(by_name[dep].status == "done" for dep in job.after if dep in by_name)

    def deps_dead(job: Job) -> bool:
        return any(by_name[dep].status in ("failed", "skipped") for dep in job.after if dep in by_name)

    print(f"[queue] {len(jobs)} jobs, GPU pool {pool}")

    if args.skip_existing:
        needs_eval: list[Job] = []
        for job in jobs:
            if not (out_root / job.name / f"{job.net}.ckpt").exists():
                continue
            job.status = "done"
            job.returncode = 0
            job.finished = now()
            missing = []
            for dataset in job.eval_datasets:
                for bn_mode in job.eval_bn_modes:
                    key = dataset if bn_mode == "eval" else f"{dataset}:{bn_mode}"
                    metrics_json = out_root / "metrics" / metrics_filename(job.name, dataset, bn_mode)
                    if metrics_json.exists():
                        job.metrics[key] = json.loads(metrics_json.read_text())
                    elif dataset not in missing:
                        missing.append(dataset)
            print(f"[queue] {job.name}: checkpoint present, skipping training"
                  + (f" (still needs eval on {', '.join(missing)})" if missing else ""))
            if missing:
                job.eval_datasets = missing
                needs_eval.append(job)
        write_status()

        # A run interrupted between training and evaluation still owes metrics;
        # schedule those as ordinary jobs so they run alongside training.
        for job in needs_eval:
            queue_eval_job(job)

    write_status()

    while True:
        # reap finished jobs
        for job in list(running):
            if job.process.poll() is None:
                continue
            job.returncode = job.process.returncode
            job.finished = now()
            job.status = "done" if job.returncode == 0 else "failed"
            print(f"[queue] {job.name} {job.status} (rc={job.returncode})")
            gpus, job.process = job.gpus, None
            running.remove(job)
            free_gpus.extend(gpus)
            if job.kind == "train" and job.status == "done" and job.eval_datasets:
                queue_eval_job(job)
            if job.kind == "eval":
                load_metrics(job)
            write_status()

        # mark jobs whose dependencies can never succeed
        for job in jobs:
            if job.status == "queued" and deps_dead(job):
                job.status = "skipped"
                print(f"[queue] {job.name} skipped (dependency failed)")
                write_status()

        # start what fits, in declaration order
        for job in jobs:
            if job.status != "queued" or not deps_ok(job):
                continue
            if len(free_gpus) < job.devices:
                continue
            job.gpus = [free_gpus.pop(0) for _ in range(job.devices)]
            job.log_path = log_dir / f"{job.name}.log"
            job.started = now()
            job.status = "running"
            cmd = job.command(job.gpus)
            print(f"[queue] start {job.name} on GPU {','.join(job.gpus)}")
            with open(job.log_path, "w") as handle:
                handle.write(" ".join(cmd) + "\n\n")
            job.process = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                stdout=open(job.log_path, "a"),
                stderr=subprocess.STDOUT,
                env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            running.append(job)
            write_status()

        if not running and all(job.status in ("done", "failed", "skipped") for job in jobs):
            break
        time.sleep(args.poll)

    write_status()
    done = sum(1 for j in jobs if j.status == "done")
    print(f"[queue] finished: {done}/{len(jobs)} succeeded")
    print(f"[queue] status: {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
