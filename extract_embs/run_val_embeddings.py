#!/usr/bin/env python3
"""Background-friendly launcher with a lock, persistent log and exit status."""
import datetime
import fcntl
import os
from pathlib import Path
import subprocess
import sys

from extract_val_embeddings import write_json


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "resultados/embeddings_val_last_layer"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("An extraction launcher is already running", file=sys.stderr)
            return 2
        command = [sys.executable, "-u", str(root / "extract_embs/extract_val_embeddings.py"),
                   "--root", str(root), "--output", str(folder), "--gpu", "0",
                   "--fallback-cache-dir", str(root / ".cache/embeddings_models")]
        state = {"status": "running", "pid": os.getpid(), "command": command,
                 "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "log": str(folder / "extraction.log")}
        write_json(folder / "runner.json", state)
        try:
            with (folder / "extraction.log").open("a", buffering=1) as log:
                print("\nSTART " + state["started_at"], file=log, flush=True)
                environment = os.environ.copy()
                environment.update(HF_HUB_OFFLINE="0", HF_HUB_DISABLE_XET="1",
                                   HF_XET_CACHE=str(root / ".cache/embeddings_xet"))
                result = subprocess.run(command, cwd=root, stdout=log,
                                        stderr=subprocess.STDOUT, env=environment)
                code = result.returncode
        except BaseException as error:
            code = 1
            state["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            state.update(status="succeeded" if code == 0 else "failed", exit_code=code,
                         finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
            write_json(folder / "runner.json", state)
        return code


if __name__ == "__main__":
    sys.exit(main())
