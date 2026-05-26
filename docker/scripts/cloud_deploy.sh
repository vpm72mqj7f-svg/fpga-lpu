#!/bin/bash
# ============================================================================
# cloud_deploy.sh — Deploy FPGA LPU builder to cloud VM
#
# Prerequisites: aws-cli or gcloud installed and configured
# ============================================================================
set -euo pipefail

CLOUD="${1:-aws}"
INSTANCE_NAME="fpgalpu-builder-$(date +%Y%m%d)"

echo "Deploying FPGA LPU Builder to ${CLOUD}..."

case "$CLOUD" in
    aws)
        INSTANCE_TYPE="${AWS_INSTANCE_TYPE:-c6i.16xlarge}"
        REGION="${AWS_REGION:-us-east-1}"

        echo "  Type:     ${INSTANCE_TYPE}"
        echo "  Region:   ${REGION}"
        echo "  Name:     ${INSTANCE_NAME}"

        # Create instance
        INSTANCE_ID=$(aws ec2 run-instances \
            --instance-type "${INSTANCE_TYPE}" \
            --region "${REGION}" \
            --image-id "${AWS_AMI:-ami-0c7217cd0322c0b99}" \
            --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=500,VolumeType=gp3,Throughput=1000,Iops=16000}' \
            --key-name "${AWS_KEY_NAME:-fpgalpu}" \
            --security-group-ids "${AWS_SG:-sg-fpgalpu}" \
            --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${INSTANCE_NAME}}]" \
            --query 'Instances[0].InstanceId' \
            --output text)

        echo "  Instance: ${INSTANCE_ID}"
        echo ""
        echo "  Wait for instance to start:"
        echo "    aws ec2 wait instance-running --instance-ids ${INSTANCE_ID} --region ${REGION}"
        echo ""
        echo "  Get IP:"
        echo "    aws ec2 describe-instances --instance-ids ${INSTANCE_ID} --region ${REGION} --query 'Reservations[0].Instances[0].PublicIpAddress'"
        echo ""
        echo "  SSH:"
        echo "    ssh -i ${AWS_KEY_NAME}.pem ubuntu@<ip>"
        ;;

    gcp)
        MACHINE_TYPE="${GCP_MACHINE_TYPE:-c2-standard-60}"
        ZONE="${GCP_ZONE:-us-central1-a}"

        echo "  Type:     ${MACHINE_TYPE}"
        echo "  Zone:     ${ZONE}"
        echo "  Name:     ${INSTANCE_NAME}"

        gcloud compute instances create "${INSTANCE_NAME}" \
            --zone="${ZONE}" \
            --machine-type="${MACHINE_TYPE}" \
            --boot-disk-size=500GB \
            --boot-disk-type=pd-ssd \
            --image-family=ubuntu-2204-lts \
            --image-project=ubuntu-os-cloud

        echo ""
        echo "  SSH:"
        echo "    gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE}"
        ;;

    *)
        echo "Usage: $0 <aws|gcp>"
        exit 1
        ;;
esac

echo ""
echo "After SSH, run:"
echo "  curl -fsSL https://get.docker.com | sh"
echo "  git clone <repo-url> /workspace"
echo "  cd /workspace"
echo "  make build-all"
