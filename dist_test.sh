#!/bin/bash
config=config/val.py
gpus=$(nvidia-smi --list-gpus | wc -l)

master_port=$(python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))          # 0 => pick an available port
print(s.getsockname()[1])
s.close()
PY
)

torchrun \
  --nproc_per_node="$gpus" \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=localhost \
  --master_port="$master_port" \
  zedd_test/zedd_test.py \
  --config "$config" \
  "$@"