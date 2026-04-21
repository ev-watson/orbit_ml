#!/bin/bash
# Simulate SLURM_ARRAY_JOB_ID
optimizers=("sgd" "adamw" "nadam" "radam" "adabound" "swats" "lion")
JOB_ID=$(echo -n "33916221" | sha256sum | xxd -r -p)

# Shuffle using your method
mapfile -t shuffled1 < <(shuf --random-source=<(printf "%s" "$JOB_ID") -e "${optimizers[@]}")
mapfile -t shuffled2 < <(shuf --random-source=<(printf "%s" "$JOB_ID") -e "${optimizers[@]}")

# Compare the shuffled arrays
if [[ "${shuffled1[*]}" == "${shuffled2[*]}" ]]; then
    echo "Shuffle is deterministic."
else
    echo "Shuffle is not deterministic."
fi

echo "Changing input from 33916221 to 33916222, only last digit change"
JOB_ID1=$(echo -n "33916221" | sha256sum | xxd -r -p)
JOB_ID2=$(echo -n "33916222" | sha256sum | xxd -r -p)

mapfile -t shuffled1 < <(shuf --random-source=<(printf "%s" "$JOB_ID1") -e "${optimizers[@]}")
mapfile -t shuffled2 < <(shuf --random-source=<(printf "%s" "$JOB_ID2") -e "${optimizers[@]}")

echo "Shuffle 1 is ${shuffled1[*]}"
echo "Shuffle 2 is ${shuffled2[*]}"