#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

train_log="${1:-${SCRIPT_DIR}/train.log}"
train_metric="${2:-${SCRIPT_DIR}/train.csv}"

if [ ! -f "${train_log}" ]; then
  printf '[FAIL] Train log does not exist: %s\n' "${train_log}" >&2
  exit 1
fi

mkdir -p "$(dirname "${train_metric}")"

printf '%s\n' \
  "target,hyper_split,search_method,split,seed,test_size,model,descriptor,R2_train,R2_test,R_train,R_test,Spearmanr_train,Spearmanr_test,MSE_train,MSE_test,MAE_train,MAE_test,RMSE_train,RMSE_test" \
  > "${train_metric}"

awk '
function reset_context() {
    target = "";
    hyper = "";
    search_method = "";
    split_name = "";
    seed = "";
    test_size = "";
    model = "";
    desc = "";
}

function update_split_fields(n, parts) {
    seed = "";
    test_size = "";
    if (split_name == "") {
        return;
    }
    n = split(split_name, parts, "_");
    if (n >= 4 && parts[1] == "seed" && parts[3] == "test") {
        seed = parts[2];
        test_size = parts[4];
        gsub(/-/, ".", test_size);
    }
}

BEGIN {
    reset_context();
}

/^\[INFO\] Train / {
    reset_context();
    for (i = 1; i <= NF; i++) {
        split($i, kv, "=");
        if (kv[1] == "target") {
            target = kv[2];
        } else if (kv[1] == "hyper") {
            hyper = kv[2];
        } else if (kv[1] == "search") {
            search_method = kv[2];
        } else if (kv[1] == "split") {
            split_name = kv[2];
        } else if (kv[1] == "model") {
            model = kv[2];
        } else if (kv[1] == "descriptor") {
            desc = kv[2];
        }
    }
    update_split_fields();
    next;
}

/^Train: model/ {
    reset_context();
    split($2, model_part, /\[|\]/);
    split($3, desc_part, /\[|\]/);
    model = model_part[2];
    desc = desc_part[2];
    next;
}

/^Info\[iaw\]:> Result/ {
    line = $0;
    sub(/^Info\[iaw\]:> Result, /, "", line);
    gsub(/, /, ",", line);
    print target "," hyper "," search_method "," split_name "," seed "," test_size "," model "," desc "," line;
}
' "${train_log}" >> "${train_metric}"

row_count="$(($(wc -l < "${train_metric}") - 1))"
printf '[OK] Extracted %s rows to %s\n' "${row_count}" "${train_metric}"
