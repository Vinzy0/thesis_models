"""Run the full pipeline: build features (if missing) -> train all 7 symptoms -> summary."""
import subprocess
import sys
import time

from common import FEATURES_PATH, ROOT, SYMPTOMS


def run(script):
    print(f"\n{'=' * 80}\n>>> python {script}\n{'=' * 80}", flush=True)
    return subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT).returncode


def main():
    t0 = time.time()
    if not FEATURES_PATH.exists():
        if run("build_features.py") != 0:
            sys.exit("build_features.py failed")
    else:
        print(f"{FEATURES_PATH.name} exists, skipping feature building "
              f"(delete it to rebuild)")

    failed = [s for s in SYMPTOMS if run(f"train_{s}.py") != 0]
    run("summary.py")
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")
    if failed:
        sys.exit(f"FAILED: {failed}")


if __name__ == "__main__":
    main()
