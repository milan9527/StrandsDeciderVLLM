#!/bin/bash
set -euo pipefail
python3 - <<'PY'
from pathlib import Path
original = Path("/opt/decider-batchtokens-20261013/original.service").read_text()
start = original[original.index("ExecStart="):original.index("ExecStop=")]
assert start.count("--max-num-batched-tokens 8192") == 1
start = start.replace("--max-num-batched-tokens 8192", "--max-num-batched-tokens __TOKENS__")
path = Path("/run/systemd/system/decider-vllm.service.d/90-batchtokens.conf")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text("[Service]\nExecStart=\n" + start)
PY
date -u +%FT%TZ > /opt/decider-batchtokens-20261013/batch-__TOKENS__-restart.txt
systemctl daemon-reload
systemctl restart decider-vllm.service
for attempt in $(seq 1 90); do
  if curl --max-time 4 -fsS http://127.0.0.1:8100/health > /opt/decider-batchtokens-20261013/batch-__TOKENS__-health.json 2>/dev/null; then
    docker inspect --format '{{json .Config.Cmd}}' decider-vllm-service > /opt/decider-batchtokens-20261013/batch-__TOKENS__-command.json
    docker logs --tail 100 decider-vllm-service > /opt/decider-batchtokens-20261013/batch-__TOKENS__-startup.log 2>&1
    date -u +%FT%TZ > /opt/decider-batchtokens-20261013/batch-__TOKENS__-ready.txt
    echo "Batch budget __TOKENS__ is healthy."
    exit 0
  fi
  sleep 5
done
docker logs --tail 160 decider-vllm-service > /opt/decider-batchtokens-20261013/batch-__TOKENS__-failure.log 2>&1 || true
journalctl -u decider-vllm.service --since "-10min" --no-pager > /opt/decider-batchtokens-20261013/batch-__TOKENS__-journal.log
exit 1
