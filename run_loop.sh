#!/usr/bin/env bash

set -euo pipefail

INPUT_FILE="parameters.dat"

for temp in 0.5 0.8 1.0 1.3 2.0; do
    sed -i -E \
        "s/^[[:space:]]*temp:[[:space:]]*.*/temp: $temp/" \
        "$INPUT_FILE"

    for mu in $(seq 0 1 3); do
        sed -i -E \
            "s/^[[:space:]]*mu:[[:space:]]*.*/mu: $mu/" \
            "$INPUT_FILE"

        for r0 in 1.1 1.3 1.5 1.7 1.9; do
            sed -i -E \
                "s/^[[:space:]]*r0:[[:space:]]*.*/r0: $r0/" \
                "$INPUT_FILE"

            for eps in 0.0 0.7 1.0 1.4 2.5; do
                sed -i -E \
                    "s/^[[:space:]]*eps:[[:space:]]*.*/eps: $eps/" \
                    "$INPUT_FILE"

                echo "Running: temp=$temp, mu=$mu, r0=$r0, eps=$eps"
                time python3 solver.py
            done
        done
    done
done
