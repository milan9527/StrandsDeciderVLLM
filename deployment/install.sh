#!/bin/bash
set -euo pipefail
export AWS_DEFAULT_REGION=us-east-1
install -d -m 0755 /opt/decider-vllm /opt/decider-model /opt/decider-results /opt/decider-cache
install -d -m 0700 /opt/decider-secrets

finish() {
  aws s3 sync /opt/decider-results/ s3://__BUCKET__/output/install/ --only-show-errors || true
}
trap finish EXIT

aws s3 cp s3://__BUCKET__/input/source.tar.gz /opt/decider-source.tar.gz --only-show-errors
tar xzf /opt/decider-source.tar.gz -C /opt/decider-vllm
aws s3 sync s3://__BUCKET__/input/model/ /opt/decider-model/ --only-show-errors
cd /opt/decider-vllm
docker build -t decider-vllm:service . > /opt/decider-results/build.log 2>&1
docker run --rm --memory=4g -v /opt/decider-model:/models/decider-vllm:ro \
  decider-vllm:service verify-artifacts --model /models/decider-vllm \
  > /opt/decider-results/model-integrity.json

# Fetch outside the inference container; the application never receives AWS credentials.
cat > /opt/decider-vllm/fetch-key.sh <<'SCRIPT'
#!/bin/bash
set -eu
umask 077
aws secretsmanager get-secret-value --region us-east-1 --secret-id '__SECRET_ARN__' \
  --query SecretString --output text > /opt/decider-secrets/payload.json
python3 - <<'PY'
import json
from pathlib import Path
root = Path("/opt/decider-secrets")
key = json.loads((root / "payload.json").read_text())["api_key"]
assert len(key) >= 32 and not any(character.isspace() for character in key)
(root / "api-key").write_text(key + "\n")
(root / "api-key").chmod(0o600)
(root / "payload.json").unlink()
PY
SCRIPT
chmod 0700 /opt/decider-vllm/fetch-key.sh

cat > /etc/systemd/system/decider-vllm.service <<'UNIT'
[Unit]
Description=Authenticated Strands Decider vLLM inference service
After=docker.service network-online.target
Requires=docker.service
Wants=network-online.target

[Service]
Type=simple
ExecStartPre=/opt/decider-vllm/fetch-key.sh
ExecStartPre=-/usr/bin/docker rm -f decider-vllm-service
ExecStart=/usr/bin/docker run --name decider-vllm-service \
  --gpus all --shm-size=2g --memory=14g --cpuset-cpus=0-3 \
  --cap-drop=ALL --security-opt=no-new-privileges:true \
  -p 8100:8100 \
  -v /opt/decider-model:/models/decider-vllm:ro \
  -v /opt/decider-cache:/root/.cache \
  -v /opt/decider-secrets/api-key:/run/secrets/api-key:ro \
  --log-driver=awslogs --log-opt awslogs-region=us-east-1 \
  --log-opt awslogs-group=__LOG_GROUP__ --log-opt awslogs-stream=model \
  --health-cmd="curl -fsS --max-time 4 http://127.0.0.1:8100/health >/dev/null" \
  --health-interval=30s --health-timeout=5s --health-retries=3 --health-start-period=600s \
  decider-vllm:service serve --model /models/decider-vllm --host 0.0.0.0 --port 8100 \
  --max-num-seqs 8 --max-inflight-questions 8 --max-pending-requests 8 \
  --max-num-batched-tokens 8192 --gpu-memory-utilization 0.7 \
  --timeout-seconds 25 --api-key-file /run/secrets/api-key \
  --requests-per-minute 1200 --rate-limit-burst 16
ExecStop=/usr/bin/docker stop --time 30 decider-vllm-service
Restart=always
RestartSec=10
TimeoutStartSec=0
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
UNIT

cat > /opt/decider-vllm/check-health.sh <<'SCRIPT'
#!/bin/bash
set -eu
health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{end}}' decider-vllm-service 2>/dev/null || true)"
if [ "$health" = unhealthy ]; then
  logger -t decider-vllm "Container health checks failed; restarting model service"
  systemctl restart decider-vllm.service
fi
SCRIPT
chmod 0755 /opt/decider-vllm/check-health.sh
cat > /etc/systemd/system/decider-vllm-health.service <<'UNIT'
[Unit]
Description=Restart an unhealthy Decider inference container
[Service]
Type=oneshot
ExecStart=/opt/decider-vllm/check-health.sh
UNIT
cat > /etc/systemd/system/decider-vllm-health.timer <<'UNIT'
[Unit]
Description=Check Decider inference health
[Timer]
OnBootSec=10min
OnUnitActiveSec=1min
Unit=decider-vllm-health.service
[Install]
WantedBy=timers.target
UNIT
systemctl daemon-reload
systemctl enable decider-vllm.service decider-vllm-health.timer
systemctl restart decider-vllm.service
systemctl start decider-vllm-health.timer

ready=0
for attempt in $(seq 1 300); do
  if curl -fsS --max-time 5 http://127.0.0.1:8100/health > /opt/decider-results/health.json 2>/dev/null; then
    ready=1
    break
  fi
  sleep 3
done
if [ "$ready" != 1 ]; then
  journalctl -u decider-vllm.service --no-pager -n 60
  docker logs --tail 100 decider-vllm-service || true
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv \
  > /opt/decider-results/gpu.csv
docker image inspect decider-vllm:service --format '{{.Id}}' > /opt/decider-results/image-id.txt
registry="__REPOSITORY__"
version="service-$(date -u +%Y%m%dT%H%M%SZ)"
aws ecr get-login-password --region us-east-1 |
  docker login --username AWS --password-stdin "${registry%%/*}" > /opt/decider-results/registry-login.log
docker tag decider-vllm:service "$registry:$version"
docker push "$registry:$version" > /opt/decider-results/image-push.log 2>&1
docker image inspect "$registry:$version" --format '{{json .RepoDigests}}' > /opt/decider-results/registry-digests.json
docker logout "${registry%%/*}" > /opt/decider-results/registry-logout.log 2>&1
printf 'Model service is healthy.\n'
