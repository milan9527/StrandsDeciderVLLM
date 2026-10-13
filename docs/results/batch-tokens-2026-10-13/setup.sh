#!/bin/bash
set -euo pipefail
root=/opt/decider-batchtokens-20261013
install -d -m 0755 "$root"
test ! -e "$root/original.service"
test ! -e /run/systemd/system/decider-vllm.service.d/90-batchtokens.conf
cp /etc/systemd/system/decider-vllm.service "$root/original.service"
sha256sum /etc/systemd/system/decider-vllm.service > "$root/original-service-sha256.txt"
docker inspect --format '{{json .Config.Cmd}}' decider-vllm-service > "$root/original-command.json"
docker inspect --format '{{.Image}}' decider-vllm-service > "$root/image-id.txt"
curl --max-time 5 -fsS http://127.0.0.1:8100/health > "$root/original-health.json"
python3 - <<'PY'
import json
from pathlib import Path
args = json.loads(Path("/opt/decider-batchtokens-20261013/original-command.json").read_text())
for flag, value in {
    "--max-num-seqs": "128", "--max-inflight-questions": "256",
    "--max-pending-requests": "256", "--max-num-batched-tokens": "8192",
    "--requests-per-minute": "120000", "--rate-limit-burst": "512",
}.items():
    assert args[args.index(flag) + 1] == value, (flag, args)
PY
cat > "$root/restore.sh" <<'SH'
#!/bin/bash
set -euo pipefail
root=/opt/decider-batchtokens-20261013
rm -f /run/systemd/system/decider-vllm.service.d/90-batchtokens.conf
sha256sum --check "$root/original-service-sha256.txt"
systemctl daemon-reload
systemctl restart decider-vllm.service
for attempt in $(seq 1 90); do
  if curl --max-time 4 -fsS http://127.0.0.1:8100/health > "$root/final-health.json" 2>/dev/null; then
    docker inspect --format '{{json .Config.Cmd}}' decider-vllm-service > "$root/final-command.json"
    cmp "$root/original-command.json" "$root/final-command.json"
    systemctl stop decider-vllm-batchtokens-rollback.timer || true
    systemctl stop decider-vllm-batchtokens-monitor.service || true
    date -u +%FT%TZ > "$root/restored-at.txt"
    aws s3 sync "$root/" s3://__BUCKET__/output/batchtokens-20261013/ --only-show-errors
    echo "Restored batch budget 8192; sequence limit remains 128."
    exit 0
  fi
  sleep 5
done
exit 1
SH
chmod 0700 "$root/restore.sh"
systemd-run --unit=decider-vllm-batchtokens-rollback --on-active=45m "$root/restore.sh"
systemd-run --unit=decider-vllm-batchtokens-monitor \
  /bin/bash -c 'nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv -l 1 > /opt/decider-batchtokens-20261013/gpu-monitor.csv'
cp "$root/original-command.json" "$root/batch-8192-command.json"
cp "$root/original-health.json" "$root/batch-8192-health.json"
echo "Current config verified; rollback and GPU monitoring armed."
