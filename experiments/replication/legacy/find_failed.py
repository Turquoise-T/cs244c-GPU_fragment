#!/usr/bin/env python3
"""Find experiments that didn't complete successfully."""

import json
import os

def main():
    with open("experiments.json") as f:
        exps = json.load(f)

    failed = []
    for i, exp in enumerate(exps):
        v100s, p100s, k80s = exp["cluster_spec"].split(":")
        log_path = os.path.join(
            "results",
            exp["figure"],
            f"v100={v100s}.p100={p100s}.k80={k80s}",
            exp["policy"],
            f"seed={exp['seed']}",
            f"lambda={exp['lambda']:.6f}.log"
        )

        if os.path.exists(log_path):
            with open(log_path) as f:
                content = f.read()
                if "Cluster utilization:" not in content:
                    failed.append(i)
                    print(f"{i}: INCOMPLETE - {exp['figure']} {exp['policy']} seed={exp['seed']} lambda={exp['lambda']}")
        else:
            failed.append(i)
            print(f"{i}: MISSING - {exp['figure']} {exp['policy']} seed={exp['seed']} lambda={exp['lambda']}")

    print(f"\nFailed indices: {failed}")
    print(f"Total: {len(failed)}")

    # Output for sbatch array
    if failed:
        print(f"\nFor sbatch --array: {','.join(map(str, failed))}")

if __name__ == "__main__":
    main()
