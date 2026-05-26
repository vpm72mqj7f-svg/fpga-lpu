# ============================================================================
# FPGA LPU — Top-Level Build Makefile
#
# Usage:
#   make build-all        Build all 8 FPGA projects (docker)
#   make build PROJ=master Build a single project
#   make sim              Run all simulations (iverilog)
#   make sim-e2e          Run E2E cluster simulation
#   make cloud-aws        Deploy builder to AWS EC2
#   make cloud-gcp        Deploy builder to GCP Compute Engine
#   make lint             Lint all RTL with Verilator
#   make clean            Remove build artifacts
# ============================================================================

SHELL := /bin/bash
PROJ_ROOT := $(shell pwd)

# Docker settings
DOCKER_IMAGE := fpgalpu-builder
DOCKER_TAG := latest
QUARTUS_DIR ?= /opt/intelFPGA_pro
LICENSE_FILE ?= $(HOME)/.quartus/license.dat

# Cloud settings
AWS_INSTANCE_TYPE ?= c6i.16xlarge
AWS_REGION ?= us-east-1
AWS_AMI ?= ami-0c7217cd0322c0b99  # Ubuntu 22.04 LTS
GCP_MACHINE_TYPE ?= c2-standard-60
GCP_ZONE ?= us-central1-a

# Icarus Verilog
IVERILOG := iverilog
VVP := vvp
IVERILOG_FLAGS := -g2012 -I$(PROJ_ROOT)/rtl/include -I$(PROJ_ROOT)/rtl/interfaces

# Verilator
VERILATOR := verilator

# ---------------------------------------------------------------------------
# Docker Build
# ---------------------------------------------------------------------------
.PHONY: docker-build
docker-build:
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) docker/

.PHONY: docker-shell
docker-shell:
	docker run --rm -it \
		-v $(PROJ_ROOT):/workspace \
		-v $(QUARTUS_DIR):/opt/intelFPGA_pro:ro \
		-v $(LICENSE_FILE):/opt/flexlm/license.dat:ro \
		--tmpfs /tmp:exec,size=32G \
		$(DOCKER_IMAGE):$(DOCKER_TAG) shell

# ---------------------------------------------------------------------------
# FPGA Build (Docker)
# ---------------------------------------------------------------------------
.PHONY: build-all
build-all: docker-build
	docker run --rm \
		-v $(PROJ_ROOT):/workspace \
		-v $(QUARTUS_DIR):/opt/intelFPGA_pro:ro \
		-v $(LICENSE_FILE):/opt/flexlm/license.dat:ro \
		--tmpfs /tmp:exec,size=32G \
		--cpus=32 --memory=128g \
		$(DOCKER_IMAGE):$(DOCKER_TAG) build-all

.PHONY: build
build: docker-build
	@if [ -z "$(PROJ)" ]; then \
		echo "Usage: make build PROJ=<project_name>"; \
		echo "Projects: bringup hbm_char dsp_char pcie_test c2c_test full_stack master slave"; \
		exit 1; \
	fi
	docker run --rm \
		-v $(PROJ_ROOT):/workspace \
		-v $(QUARTUS_DIR):/opt/intelFPGA_pro:ro \
		-v $(LICENSE_FILE):/opt/flexlm/license.dat:ro \
		--tmpfs /tmp:exec,size=32G \
		$(DOCKER_IMAGE):$(DOCKER_TAG) build $(PROJ) fpga_lpu_$(PROJ)

.PHONY: check-timing
check-timing:
	docker run --rm \
		-v $(PROJ_ROOT):/workspace \
		-v $(QUARTUS_DIR):/opt/intelFPGA_pro:ro \
		-v $(LICENSE_FILE):/opt/flexlm/license.dat:ro \
		$(DOCKER_IMAGE):$(DOCKER_TAG) check-timing

# ---------------------------------------------------------------------------
# Simulation (Icarus Verilog)
# ---------------------------------------------------------------------------
RTL_SRC := $(PROJ_ROOT)/rtl
SIM_DIR := $(PROJ_ROOT)/rtl/sim
DSP_SRC := $(RTL_SRC)/dsp/fp4_mac.sv $(RTL_SRC)/dsp/fp4_scale_reader.sv \
	$(RTL_SRC)/dsp/fp4_systolic_tile.sv $(RTL_SRC)/dsp/fp4_scaled_tile.sv \
	$(RTL_SRC)/dsp/fp4_systolic_array.sv $(RTL_SRC)/dsp/fp4_linear_engine.sv
ATTN_SRC := $(RTL_SRC)/attention/mla_kv_cache.sv $(RTL_SRC)/attention/mla_qkv_proj.sv \
	$(RTL_SRC)/attention/mla_rope.sv $(RTL_SRC)/attention/mla_attention_v2.sv
ACT_SRC := $(RTL_SRC)/activation/rms_norm.sv $(RTL_SRC)/activation/silu_q12_lut.sv \
	$(RTL_SRC)/activation/q12_to_fp8_e4m3.sv
MOE_SRC := $(RTL_SRC)/moe/router_topk.sv $(RTL_SRC)/moe/expert_ffn_engine_fp4_down.sv
LAYER_SRC := $(RTL_SRC)/layer/full_transformer_layer.sv $(RTL_SRC)/layer/mhc_mixer.sv
ALL_RTL := $(DSP_SRC) $(ATTN_SRC) $(ACT_SRC) $(MOE_SRC) $(LAYER_SRC)

.PHONY: sim
sim:
	@echo "=== Unit Tests ==="
	cd $(SIM_DIR) && \
	$(IVERILOG) $(IVERILOG_FLAGS) -o tb_mac.vvp $(RTL_SRC)/dsp/fp4_mac.sv tb_fp4_mac.sv && $(VVP) tb_mac.vvp && \
	$(IVERILOG) $(IVERILOG_FLAGS) -o tb_rt.vvp $(RTL_SRC)/moe/router_topk.sv tb_router_topk.sv && $(VVP) tb_rt.vvp && \
	$(IVERILOG) $(IVERILOG_FLAGS) -o tb_rn.vvp $(RTL_SRC)/activation/rms_norm.sv tb_rms_norm.sv && $(VVP) tb_rn.vvp && \
	echo "=== All Unit Tests Passed ==="

.PHONY: sim-e2e
sim-e2e:
	cd $(SIM_DIR) && \
	$(IVERILOG) $(IVERILOG_FLAGS) -o tb_e2e.vvp $(ALL_RTL) tb_full_transformer_layer.sv && \
	$(VVP) tb_e2e.vvp

.PHONY: sim-cluster
sim-cluster:
	cd $(SIM_DIR) && \
	$(IVERILOG) $(IVERILOG_FLAGS) -o tb_cluster.vvp $(ALL_RTL) tb_cluster_384.sv && \
	$(VVP) tb_cluster.vvp

# ---------------------------------------------------------------------------
# Lint (Verilator)
# ---------------------------------------------------------------------------
.PHONY: lint
lint:
	$(VERILATOR) --lint-only -Wall \
		+incdir+$(RTL_SRC)/include +incdir+$(RTL_SRC)/interfaces \
		$(ALL_RTL)

# ---------------------------------------------------------------------------
# Cloud Deployment
# ---------------------------------------------------------------------------
.PHONY: cloud-aws
cloud-aws:
	@echo "Deploying FPGA LPU Builder to AWS EC2..."
	@echo "  Instance: $(AWS_INSTANCE_TYPE)"
	@echo "  Region:   $(AWS_REGION)"
	@echo ""
	@echo "Manual steps (run on cloud shell or local aws-cli):"
	@echo ""
	@echo "  # 1. Launch instance with 500 GB gp3 disk"
	@echo "  aws ec2 run-instances \\"
	@echo "    --instance-type $(AWS_INSTANCE_TYPE) \\"
	@echo "    --region $(AWS_REGION) \\"
	@echo "    --image-id $(AWS_AMI) \\"
	@echo "    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=500,VolumeType=gp3}' \\"
	@echo "    --key-name my-fpga-key"
	@echo ""
	@echo "  # 2. SSH to instance, install docker"
	@echo "  ssh -i my-fpga-key.pem ubuntu@<instance-ip>"
	@echo "  curl -fsSL https://get.docker.com | sh"
	@echo ""
	@echo "  # 3. Clone repo, mount Quartus, run build"
	@echo "  git clone <repo-url> /workspace"
	@echo "  cd /workspace && make build-all"
	@echo ""
	@echo "  # 4. Download results"
	@echo "  scp -r ubuntu@<ip>:/workspace/build_results ./"

.PHONY: cloud-gcp
cloud-gcp:
	@echo "Deploying FPGA LPU Builder to GCP Compute Engine..."
	@echo "  Machine: $(GCP_MACHINE_TYPE)"
	@echo "  Zone:    $(GCP_ZONE)"
	@echo ""
	@echo "  # 1. Create instance"
	@echo "  gcloud compute instances create fpgalpu-builder \\"
	@echo "    --zone=$(GCP_ZONE) \\"
	@echo "    --machine-type=$(GCP_MACHINE_TYPE) \\"
	@echo "    --boot-disk-size=500GB \\"
	@echo "    --boot-disk-type=pd-ssd \\"
	@echo "    --image-family=ubuntu-2204-lts \\"
	@echo "    --image-project=ubuntu-os-cloud"
	@echo ""
	@echo "  # 2. SSH to instance"
	@echo "  gcloud compute ssh fpgalpu-builder --zone=$(GCP_ZONE)"

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
.PHONY: clean
clean:
	find $(PROJ_ROOT) -name "*.vvp" -delete
	find $(PROJ_ROOT) -name "*.vcd" -delete
	find $(PROJ_ROOT) -name "*.log" -delete
	find $(PROJ_ROOT) -name "work" -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf $(PROJ_ROOT)/rtl/sim/build/*.vvp 2>/dev/null || true

.PHONY: info
info:
	@echo "============================================"
	@echo " FPGA LPU — Build System"
	@echo "============================================"
	@echo " Projects: bringup hbm_char dsp_char pcie_test c2c_test full_stack master slave"
	@echo ""
	@echo " Quick start:"
	@echo "   make sim             Run all simulations"
	@echo "   make sim-cluster     Run 32-chip/384-layer E2E"
	@echo "   make lint             Lint RTL"
	@echo "   make build PROJ=XXX   Build one FPGA project"
	@echo "   make build-all        Build all FPGA projects"
	@echo ""

.DEFAULT_GOAL := info
