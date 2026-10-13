# AWS 部署与运维

[English](README.md) | 简体中文

独立公共 API 部署在 `us-east-1` 常驻按需 `g6.xlarge` 上。CloudFront 提供 HTTPS，通过 VPC 源站私网访问 EC2。实例安全组只允许 CloudFront VPC 源站服务安全组访问 8100 端口，不开放公网 SSH 或直接推理访问。

接口使用见 [API 文档](../docs/API.zh-CN.md)，实际环境与验证结果见 [部署记录](../docs/DEPLOYED.zh-CN.md)。

## 部署

准备具有创建相关资源权限的 AWS 凭证、Python 3.11+、`uv`，以及 `models/decider-vllm` 中的导出权重。上传前校验权重。

`manage.py` 内置本次部署使用的 VPC、子网、AMI、区域和资源名称。部署到其他账号前须检查并替换这些常量。所选子网需要出站互联网访问，脚本要求 VPC 中已有 AWS 托管的 `CloudFront-VPCOrigins-Service-SG` 安全组。脚本先在配置的子网中尝试按需 `g6.xlarge`，容量不足再尝试 `g5.xlarge`，不会自动申请更大实例。

```bash
uv run --no-project --with boto3 python deployment/manage.py provision
uv run --no-project --with boto3 python deployment/manage.py upload
uv run --no-project --with boto3 python deployment/manage.py command deployment/install.sh
uv run --no-project --with boto3 python deployment/manage.py expose
uv run --no-project --with boto3 python deployment/manage.py status
```

脚本创建加密 160 GiB gp3 系统盘、私有 S3 桶、私有 ECR 仓库、Secrets Manager 密钥、IAM 实例角色、安全组和 CloudWatch 日志组。安装通过 SSM 执行，无需 SSH。安装过程校验模型哈希、构建并启动容器，等待模型就绪后将镜像推送到 ECR。

CloudFront VPC 源站异步部署。其状态变为 `Deployed` 后再次运行 `expose`，创建分发并等待部署完成。HTTPS 地址保存在 `.deployment/state.json` 和 `.deployment/api-credentials.json` 中。后者同时包含 API Key，必须保密；两个文件均被 Git 忽略。

验证公网服务：

```bash
uv run --no-sync python scripts/verify_service.py \
  --credentials .deployment/api-credentials.json \
  --overload-requests 512 \
  --output artifacts/public-api-verification
```

每次使用新的输出目录。脚本检查鉴权、输入校验、请求体限制、HTTPS、原版模型结果对齐、过载处理和限速下的 8 并发批次，不能据此认定长时间负载下的 SLA。

当前部署允许 256 个 HTTP 请求和 256 个问题，vLLM `max_num_seqs=128`，令牌桶每秒 2,000 次，突发容量 512。验证突发需超过准入容量才能检查 429。[1–128 容量报告](../docs/CONCURRENCY_128.zh-CN.md) 保留了实际吞吐和一项数值阈值失败；较高的速率配置是拒绝阈值，不代表 GPU 吞吐量。

## 运维

通过 SSM Session Manager 或 `manage.py command PATH_TO_SCRIPT` 操作主机。后者将命令输出存入私有 S3 桶，脚本中不要包含或打印密钥值。

| 组件 | EC2 上的位置 / 命令 |
|---|---|
| 容器 | `docker ps --filter name=decider-vllm-service` |
| 模型就绪状态 | `curl -fsS http://127.0.0.1:8100/health` |
| 服务状态 | `systemctl status decider-vllm.service` |
| 重启服务 | `systemctl restart decider-vllm.service` |
| 自动健康恢复 | `systemctl status decider-vllm-health.timer` |
| 应用日志 | CloudWatch `/strands-decider-vllm/model`，日志流 `model`；保留 14 天 |
| 启动日志 | `journalctl -u decider-vllm.service` |
| 权重 | `/opt/decider-model`，只读挂载 |
| 持久化运行缓存 | `/opt/decider-cache` |
| 私有密钥文件 | `/opt/decider-secrets/api-key`，权限 0600 |
| 安装记录 | `/opt/decider-results` 及桶内 `output/install/` 前缀 |

应用记录请求 ID、阶段耗时和状态码，不记录请求体或鉴权头。通过客户端收到的 `X-Request-ID`，关联 `http_completed`、`preprocessing_completed`、`engine_submitted` 和 `engine_completed` 事件。

服务不会空闲关机。停止 EC2 会保留系统盘、权重、缓存和私有 IP，但 API 在实例及模型重新启动前不可用。Systemd 在开机时启动模型。公网 IPv4 用于出站流量，停止再启动后可能变化；CloudFront 使用私网源站。

## 更新与密钥轮换

更新程序时先运行 `upload`，再运行 `command deployment/install.sh`。远端大小一致的模型文件会跳过上传，安装时检查导出清单。更换权重时使用新导出，确保相应 S3 对象已替换，并在提供服务前通过完整性校验。安装会重启这个独立 API，造成短暂中断；保留上一个 ECR 镜像摘要以便回退。

轮换密钥时生成至少 32 个非空白字符的随机 Key，更新 Secrets Manager 中的 `api_key` 字段，再重启 `decider-vllm.service`。启动时由宿主机获取密钥并将文件只读挂载到容器。安全更新本地凭证文件及授权客户端。初始部署使用一个共享 Key，轮换立即替换旧 Key，各调用方共用限额。

## 范围与资源生命周期

当前是单可用区、单 GPU 部署，没有自动扩缩容或跨可用区故障切换，与原支付演示的 ECS 后端和 GPU 相互独立。GPU 常驻产生持续费用；EBS、ECR、Secrets Manager、CloudFront、S3、日志及公网 IPv4 按实际使用分别计费。

脚本不自动清理资源。下线时先禁用并删除 CloudFront 分发及 VPC 源站，再终止 EC2，并检查磁盘、IAM、安全组、密钥、镜像、日志和 S3 资源。删除前保存需要的记录。不要删除可能被其他分发共用的 AWS 托管 CloudFront 服务安全组。
