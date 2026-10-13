# AWS deployment and operations

English | [简体中文](README.zh-CN.md)

The independent public API runs on an always-on, On-Demand `g6.xlarge` in `us-east-1`. CloudFront provides HTTPS and reaches EC2 through a private VPC origin. The instance security group permits port 8100 only from the CloudFront VPC origin service security group. There is no public SSH or direct inference access.

See [API usage](../docs/API.md) and the [deployment record](../docs/DEPLOYED.md).

## Deploy

Prerequisites: AWS credentials with permission to create the listed resources, Python 3.11+, `uv`, and exported weights in `models/decider-vllm`. Verify the weights before uploading.

`manage.py` contains this deployment's VPC, subnet, AMI, region, and resource-name defaults. For another account, review and replace these constants first. The selected subnets need outbound internet access, and this script expects the AWS-managed `CloudFront-VPCOrigins-Service-SG` to exist in the VPC. It tries On-Demand `g6.xlarge` capacity across the configured subnets, then `g5.xlarge`; it does not launch a larger instance.

```bash
uv run --no-project --with boto3 python deployment/manage.py provision
uv run --no-project --with boto3 python deployment/manage.py upload
uv run --no-project --with boto3 python deployment/manage.py command deployment/install.sh
uv run --no-project --with boto3 python deployment/manage.py expose
uv run --no-project --with boto3 python deployment/manage.py status
```

Provisioning creates an encrypted 160 GiB gp3 root disk, private S3 bucket, private ECR repository, Secrets Manager secret, IAM instance profile, security group, and CloudWatch log group. SSM runs installation without SSH. Installation verifies model hashes, builds and starts the container, waits for model readiness, and pushes the image to ECR.

CloudFront VPC origin provisioning is asynchronous. Run `expose` again after the origin reports `Deployed` to create the distribution, then wait for the distribution to deploy. The HTTPS URL is saved in `.deployment/state.json` and `.deployment/api-credentials.json`. The latter also holds the API key and must remain private. Both files are ignored by Git.

Verify the deployed endpoint:

```bash
uv run --no-sync python scripts/verify_service.py \
  --credentials .deployment/api-credentials.json \
  --overload-requests 512 \
  --output artifacts/public-api-verification
```

Use a new output directory for each run. This checks authentication, schema validation, body limits, HTTPS, original-model parity, overload handling, and paced batches of eight concurrent requests. It does not establish a sustained-load SLA.

The current deployment admits 256 HTTP requests and 256 questions, with vLLM `max_num_seqs=128`, a 2,000 requests/second token bucket, and burst 512. The verification burst must exceed admission capacity to test 429 responses. The [1–128 capacity report](../docs/CONCURRENCY_128.md) records actual throughput and one numeric-tolerance failure; the higher configured rate is a rejection threshold, not GPU throughput.

## Operate

Use SSM Session Manager or `manage.py command PATH_TO_SCRIPT` for host operations. The latter saves command output to the private S3 bucket; never put secret values in a script or print them.

| Component | Location / command on EC2 |
|---|---|
| Container | `docker ps --filter name=decider-vllm-service` |
| Model readiness | `curl -fsS http://127.0.0.1:8100/health` |
| Service | `systemctl status decider-vllm.service` |
| Restart | `systemctl restart decider-vllm.service` |
| Automatic health recovery | `systemctl status decider-vllm-health.timer` |
| Application logs | CloudWatch `/strands-decider-vllm/model`, stream `model`; 14-day retention |
| Startup logs | `journalctl -u decider-vllm.service` |
| Weights | `/opt/decider-model`, mounted read-only |
| Persistent runtime cache | `/opt/decider-cache` |
| Private key file | `/opt/decider-secrets/api-key`, mode 0600 |
| Installation evidence | `/opt/decider-results` and the bucket's `output/install/` prefix |

The application logs request IDs, stage timings, and status codes. It does not log request bodies or authorization headers. Correlate the client's `X-Request-ID` with `http_completed`, `preprocessing_completed`, `engine_submitted`, and `engine_completed` events.

There is no idle shutdown. Stopping EC2 retains the root disk, weights, caches, and private IP, but the API is unavailable until the instance and model restart. Systemd starts the model on boot. A public IPv4 address is used for outbound traffic and can change after a stop/start; CloudFront uses the private origin.

## Update and rotate credentials

For an application update, run `upload`, then `command deployment/install.sh`. Model files are skipped when their remote sizes match; installation checks the export manifest. For changed weights, use a fresh export, ensure the corresponding S3 objects are replaced, and confirm integrity before serving. Installation restarts this independent API, causing a brief interruption. Keep the previous ECR image digest for rollback.

To rotate the key, generate a new random key of at least 32 non-whitespace characters, update the `api_key` field in Secrets Manager, and restart `decider-vllm.service`. Startup retrieves the secret outside the container and mounts a read-only key file. Securely update the local credentials file and authorized clients. The initial deployment supports one shared key; rotation replaces it immediately and limits are aggregate across callers.

## Scope and resource lifecycle

This is one GPU in one Availability Zone, without automatic scaling or multi-AZ failover. The deployment is separate from the original payment demo's ECS backend and GPU. Costs continue while the GPU runs; EBS, ECR, Secrets Manager, CloudFront, S3, logs, and public IPv4 are billed separately as applicable.

The script has no automatic teardown. If retiring the service, first disable and delete the CloudFront distribution and its VPC origin, then terminate EC2 and review its disk, IAM, security group, secret, image, log, and S3 resources. Preserve any required evidence before deletion. Do not delete the AWS-managed CloudFront service security group, which may serve other distributions.
