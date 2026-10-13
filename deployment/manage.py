"""Provision and operate the independent authenticated Decider vLLM API."""

import argparse
import concurrent.futures
import json
import os
import secrets
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / ".deployment"
STATE = PRIVATE / "state.json"
NAME = "strands-decider-vllm"
REGION = "us-east-1"
VPC = "vpc-0c036c30b82b3703f"
AMI = "ami-028357be7d5b15c53"
SUBNETS = [
    "subnet-0693778771c7fe672",
    "subnet-062a148d1354577b3",
    "subnet-023880de85fb9261b",
    "subnet-029c0264a7e188a46",
    "subnet-035e3e2820bda44e3",
]


def client(service):
    return boto3.client(service, region_name=REGION)


def save(state):
    PRIVATE.mkdir(mode=0o700, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2, default=str) + "\n")
    STATE.chmod(0o600)


def load():
    return json.loads(STATE.read_text())


def provision():
    if STATE.exists():
        state = load()
    else:
        account = client("sts").get_caller_identity()["Account"]
        state = {
            "name": NAME,
            "region": REGION,
            "account": account,
            "bucket": f"{NAME}-{account}",
            "created_at": datetime.now(UTC).isoformat(),
        }
        save(state)
    account, bucket = state["account"], state["bucket"]
    s3, iam, ec2 = client("s3"), client("iam"), client("ec2")
    tags = [{"Key": "Project", "Value": NAME}, {"Key": "Name", "Value": NAME}]
    if not state.get("bucket_created"):
        s3.create_bucket(Bucket=bucket)
        s3.put_public_access_block(
            Bucket=bucket,
            PublicAccessBlockConfiguration={
                key: True
                for key in (
                    "BlockPublicAcls",
                    "IgnorePublicAcls",
                    "BlockPublicPolicy",
                    "RestrictPublicBuckets",
                )
            },
        )
        s3.put_bucket_encryption(
            Bucket=bucket,
            ServerSideEncryptionConfiguration={
                "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
            },
        )
        s3.put_bucket_policy(
            Bucket=bucket,
            Policy=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Deny",
                            "Principal": "*",
                            "Action": "s3:*",
                            "Resource": [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/*"],
                            "Condition": {"Bool": {"aws:SecureTransport": "false"}},
                        }
                    ],
                }
            ),
        )
        state["bucket_created"] = True
        save(state)
    if not state.get("secret_arn"):
        key = "sdv_" + secrets.token_urlsafe(32)
        credentials = PRIVATE / "api-credentials.json"
        descriptor = os.open(credentials, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            json.dump({"api_key": key}, handle)
            handle.write("\n")
        result = client("secretsmanager").create_secret(
            Name=f"{NAME}/api-key",
            SecretString=json.dumps({"api_key": key}),
            Tags=tags,
        )
        state["secret_arn"] = result["ARN"]
        save(state)
    if not state.get("repository"):
        result = client("ecr").create_repository(
            repositoryName=NAME,
            imageScanningConfiguration={"scanOnPush": True},
            tags=tags,
        )
        state["repository"] = result["repository"]["repositoryUri"]
        save(state)
    if not state.get("log_group"):
        group = f"/{NAME}/model"
        client("logs").create_log_group(logGroupName=group, tags={"Project": NAME})
        client("logs").put_retention_policy(logGroupName=group, retentionInDays=14)
        state["log_group"] = group
        save(state)
    if not state.get("role"):
        iam.create_role(
            RoleName=NAME,
            AssumeRolePolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "ec2.amazonaws.com"},
                            "Action": "sts:AssumeRole",
                        }
                    ],
                }
            ),
            Tags=tags,
        )
        state["role"] = NAME
        save(state)
    iam.attach_role_policy(
        RoleName=NAME, PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
    )
    iam.put_role_policy(
        RoleName=NAME,
        PolicyName="ServiceArtifacts",
        PolicyDocument=json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": "s3:ListBucket",
                        "Resource": f"arn:aws:s3:::{bucket}",
                    },
                    {
                        "Effect": "Allow",
                        "Action": "s3:GetObject",
                        "Resource": f"arn:aws:s3:::{bucket}/input/*",
                    },
                    {
                        "Effect": "Allow",
                        "Action": "s3:PutObject",
                        "Resource": f"arn:aws:s3:::{bucket}/output/*",
                    },
                    {
                        "Effect": "Allow",
                        "Action": "secretsmanager:GetSecretValue",
                        "Resource": state["secret_arn"],
                    },
                    {"Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*"},
                    {
                        "Effect": "Allow",
                        "Action": [
                            "ecr:BatchGetImage",
                            "ecr:GetDownloadUrlForLayer",
                            "ecr:BatchCheckLayerAvailability",
                            "ecr:InitiateLayerUpload",
                            "ecr:UploadLayerPart",
                            "ecr:CompleteLayerUpload",
                            "ecr:PutImage",
                        ],
                        "Resource": f"arn:aws:ecr:{REGION}:{account}:repository/{NAME}",
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "logs:CreateLogStream",
                            "logs:PutLogEvents",
                            "logs:DescribeLogStreams",
                        ],
                        "Resource": f"arn:aws:logs:{REGION}:{account}:log-group:{state['log_group']}:*",
                    },
                ],
            }
        ),
    )
    if not state.get("profile"):
        iam.create_instance_profile(InstanceProfileName=NAME)
        iam.add_role_to_instance_profile(InstanceProfileName=NAME, RoleName=NAME)
        state["profile"] = NAME
        save(state)
        time.sleep(10)
    if not state.get("security_group"):
        group = ec2.create_security_group(
            GroupName=NAME,
            Description="Decider vLLM: CloudFront VPC origin only",
            VpcId=VPC,
            TagSpecifications=[{"ResourceType": "security-group", "Tags": tags}],
        )["GroupId"]
        state["security_group"] = group
        save(state)
        managed = ec2.describe_security_groups(
            Filters=[
                {"Name": "vpc-id", "Values": [VPC]},
                {"Name": "group-name", "Values": ["CloudFront-VPCOrigins-Service-SG"]},
            ]
        )["SecurityGroups"]
        if not managed:
            raise RuntimeError("CloudFront VPC origin service security group is required")
        ec2.authorize_security_group_ingress(
            GroupId=group,
            IpPermissions=[
                {
                    "IpProtocol": "tcp",
                    "FromPort": 8100,
                    "ToPort": 8100,
                    "UserIdGroupPairs": [{"GroupId": managed[0]["GroupId"]}],
                }
            ],
        )
        state["cloudfront_service_security_group"] = managed[0]["GroupId"]
        save(state)
    if not state.get("instance"):
        launch(state)
    print(
        json.dumps({key: state.get(key) for key in ("name", "instance", "instance_type", "bucket")})
    )


def launch(state):
    ec2 = boto3.client("ec2", region_name=REGION, config=Config(retries={"max_attempts": 0}))
    userdata = """#!/bin/bash
set -eu
exec >> /var/log/decider-bootstrap.log 2>&1
systemctl enable --now docker
systemctl start amazon-ssm-agent || systemctl start snap.amazon-ssm-agent.amazon-ssm-agent || true
install -d -m 0755 /opt/decider-vllm /opt/decider-model /opt/decider-results /opt/decider-cache
docker pull vllm/vllm-openai:v0.31.0
"""
    for instance_type in ("g6.xlarge", "g5.xlarge"):
        for subnet in SUBNETS:
            try:
                result = ec2.run_instances(
                    ImageId=AMI,
                    InstanceType=instance_type,
                    MinCount=1,
                    MaxCount=1,
                    ClientToken=f"{NAME}-{instance_type}-{subnet}-20261013",
                    IamInstanceProfile={"Name": state["profile"]},
                    MetadataOptions={"HttpTokens": "required", "HttpPutResponseHopLimit": 1},
                    InstanceInitiatedShutdownBehavior="stop",
                    NetworkInterfaces=[
                        {
                            "DeviceIndex": 0,
                            "SubnetId": subnet,
                            "Groups": [state["security_group"]],
                            "AssociatePublicIpAddress": True,
                            "DeleteOnTermination": True,
                        }
                    ],
                    BlockDeviceMappings=[
                        {
                            "DeviceName": "/dev/sda1",
                            "Ebs": {
                                "VolumeSize": 160,
                                "VolumeType": "gp3",
                                "Encrypted": True,
                                "Iops": 3000,
                                "Throughput": 250,
                                "DeleteOnTermination": True,
                            },
                        }
                    ],
                    TagSpecifications=[
                        {
                            "ResourceType": kind,
                            "Tags": [
                                {"Key": "Project", "Value": NAME},
                                {"Key": "Name", "Value": NAME},
                            ],
                        }
                        for kind in ("instance", "volume")
                    ],
                    UserData=userdata,
                )
                instance = result["Instances"][0]
                state.update(
                    instance=instance["InstanceId"], instance_type=instance_type, subnet=subnet
                )
                save(state)
                print("Launched", instance["InstanceId"], instance_type, flush=True)
                return
            except ClientError as error:
                code = error.response["Error"]["Code"]
                print(instance_type, subnet, code, flush=True)
                if code != "InsufficientInstanceCapacity":
                    raise
    raise RuntimeError(
        "No xlarge GPU capacity in the attempted zones; larger instances were not launched"
    )


def upload():
    state, s3 = load(), client("s3")
    archive_path = PRIVATE / "source.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name in ("pyproject.toml", "Dockerfile", "src", "deployment", "examples"):
            archive.add(
                ROOT / name,
                arcname=name,
                filter=lambda item: (
                    None if "__pycache__" in item.name or item.name.endswith(".pyc") else item
                ),
            )
    s3.upload_file(str(archive_path), state["bucket"], "input/source.tar.gz")
    files = [p for p in (ROOT / "models/decider-vllm").rglob("*") if p.is_file()]

    def put(path):
        key = "input/model/" + str(path.relative_to(ROOT / "models/decider-vllm"))
        try:
            if (
                s3.head_object(Bucket=state["bucket"], Key=key)["ContentLength"]
                == path.stat().st_size
            ):
                return
        except ClientError as error:
            if error.response["Error"]["Code"] not in ("404", "NoSuchKey"):
                raise
        s3.upload_file(str(path), state["bucket"], key)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(put, files))
    print("Uploaded source and", len(files), "model files")


def command(path):
    state = load()
    script = path.read_text()
    for token, key in (
        ("__BUCKET__", "bucket"),
        ("__REPOSITORY__", "repository"),
        ("__SECRET_ARN__", "secret_arn"),
        ("__LOG_GROUP__", "log_group"),
    ):
        script = script.replace(token, state[key])
    result = client("ssm").send_command(
        InstanceIds=[state["instance"]],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": [script], "executionTimeout": ["7200"]},
        OutputS3BucketName=state["bucket"],
        OutputS3KeyPrefix="output/ssm",
        Comment="Independent Decider vLLM service deployment",
    )
    identifier = result["Command"]["CommandId"]
    state.setdefault("commands", []).append({"id": identifier, "script": str(path)})
    save(state)
    print(identifier)


def status():
    state = load()
    if state.get("instance"):
        instance = client("ec2").describe_instances(InstanceIds=[state["instance"]])[
            "Reservations"
        ][0]["Instances"][0]
        print(
            "Instance",
            instance["State"]["Name"],
            instance["InstanceType"],
            instance.get("PrivateIpAddress"),
        )
        managed = client("ssm").describe_instance_information(
            Filters=[{"Key": "InstanceIds", "Values": [state["instance"]]}]
        )["InstanceInformationList"]
        print("SSM", [item["PingStatus"] for item in managed])
    for item in state.get("commands", [])[-2:]:
        try:
            result = client("ssm").get_command_invocation(
                CommandId=item["id"], InstanceId=state["instance"]
            )
            print(item["id"], result["Status"])
            print(result["StandardOutputContent"][-2500:], result["StandardErrorContent"][-1000:])
        except ClientError as error:
            print(error.response["Error"]["Code"])
    if state.get("vpc_origin"):
        print(
            "VPC origin",
            client("cloudfront").get_vpc_origin(Id=state["vpc_origin"])["VpcOrigin"]["Status"],
        )
    if state.get("distribution"):
        result = client("cloudfront").get_distribution(Id=state["distribution"])["Distribution"]
        print("HTTPS", result["Status"], "https://" + result["DomainName"])


def expose():
    state, cloudfront = load(), client("cloudfront")
    instance = client("ec2").describe_instances(InstanceIds=[state["instance"]])["Reservations"][0][
        "Instances"
    ][0]
    if not state.get("vpc_origin"):
        result = cloudfront.create_vpc_origin(
            VpcOriginEndpointConfig={
                "Name": NAME,
                "Arn": f"arn:aws:ec2:{REGION}:{state['account']}:instance/{state['instance']}",
                "HTTPPort": 8100,
                "HTTPSPort": 443,
                "OriginProtocolPolicy": "http-only",
                "OriginSslProtocols": {"Quantity": 1, "Items": ["TLSv1.2"]},
            },
            Tags={"Items": [{"Key": "Project", "Value": NAME}]},
        )
        state["vpc_origin"] = result["VpcOrigin"]["Id"]
        save(state)
    origin = cloudfront.get_vpc_origin(Id=state["vpc_origin"])["VpcOrigin"]
    if origin["Status"] != "Deployed":
        print("VPC origin is", origin["Status"], "- run expose again after deployment")
        return
    if not state.get("distribution"):
        cache = next(
            item["CachePolicy"]["Id"]
            for item in cloudfront.list_cache_policies(Type="managed")["CachePolicyList"]["Items"]
            if item["CachePolicy"]["CachePolicyConfig"]["Name"] == "Managed-CachingDisabled"
        )
        policy = next(
            item["OriginRequestPolicy"]["Id"]
            for item in cloudfront.list_origin_request_policies(Type="managed")[
                "OriginRequestPolicyList"
            ]["Items"]
            if item["OriginRequestPolicy"]["OriginRequestPolicyConfig"]["Name"]
            == "Managed-AllViewerExceptHostHeader"
        )
        result = cloudfront.create_distribution_with_tags(
            DistributionConfigWithTags={
                "DistributionConfig": {
                    "CallerReference": NAME + "-20261013",
                    "Comment": "Authenticated Strands Decider vLLM inference API",
                    "Enabled": True,
                    "HttpVersion": "http2",
                    "IsIPV6Enabled": True,
                    "PriceClass": "PriceClass_100",
                    "Origins": {
                        "Quantity": 1,
                        "Items": [
                            {
                                "Id": NAME,
                                "DomainName": instance["PrivateDnsName"],
                                "VpcOriginConfig": {
                                    "VpcOriginId": state["vpc_origin"],
                                    "OriginReadTimeout": 60,
                                    "OriginKeepaliveTimeout": 5,
                                },
                                "ConnectionAttempts": 1,
                                "ConnectionTimeout": 10,
                            }
                        ],
                    },
                    "DefaultCacheBehavior": {
                        "TargetOriginId": NAME,
                        "ViewerProtocolPolicy": "https-only",
                        "TrustedSigners": {"Enabled": False, "Quantity": 0},
                        "TrustedKeyGroups": {"Enabled": False, "Quantity": 0},
                        "AllowedMethods": {
                            "Quantity": 7,
                            "Items": ["GET", "HEAD", "OPTIONS", "PUT", "PATCH", "POST", "DELETE"],
                            "CachedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"]},
                        },
                        "CachePolicyId": cache,
                        "OriginRequestPolicyId": policy,
                        "Compress": True,
                    },
                    "ViewerCertificate": {
                        "CloudFrontDefaultCertificate": True,
                        "MinimumProtocolVersion": "TLSv1.2_2021",
                    },
                    "CustomErrorResponses": {
                        "Quantity": 6,
                        "Items": [
                            {"ErrorCode": code, "ErrorCachingMinTTL": 0}
                            for code in (403, 404, 500, 502, 503, 504)
                        ],
                    },
                },
                "Tags": {"Items": [{"Key": "Project", "Value": NAME}]},
            },
        )
        distribution = result["Distribution"]
        state.update(
            distribution=distribution["Id"],
            endpoint="https://" + distribution["DomainName"],
        )
        save(state)
        credentials = PRIVATE / "api-credentials.json"
        secret = json.loads(credentials.read_text())
        secret.update(endpoint=state["endpoint"], secret_arn=state["secret_arn"])
        credentials.write_text(json.dumps(secret, indent=2) + "\n")
        credentials.chmod(0o600)
    print("HTTPS endpoint:", state["endpoint"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("provision", "upload", "command", "status", "expose"))
    parser.add_argument("path", type=Path, nargs="?")
    args = parser.parse_args()
    if args.action == "command":
        if args.path is None:
            parser.error("command requires a script path")
        command(args.path)
    else:
        globals()[args.action]()


if __name__ == "__main__":
    main()
