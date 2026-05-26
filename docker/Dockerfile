# ============================================================================
# FPGA LPU — Docker Build Environment for Quartus Prime Pro 24.3
#
# Target: Intel Agilex 7 M-Series (AGMF039R47A1E2VR0)
#
# Usage:
#   docker build -t fpgalpu-builder .
#   docker run --rm \
#     -v $(pwd):/workspace \
#     -v /path/to/quartus:/opt/intelFPGA_pro \
#     -v /path/to/license.dat:/opt/flexlm/license.dat \
#     -e LM_LICENSE_FILE=/opt/flexlm/license.dat \
#     fpgalpu-builder build-all
# ============================================================================

FROM ubuntu:22.04

LABEL description="FPGA LPU Quartus Prime Pro 24.3 Build Environment"
LABEL target="Intel Agilex 7 M-Series AGMF039"
LABEL maintainer="fpgalpu"

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    # Quartus runtime deps
    libc6-dev-i386 \
    libx11-6 \
    libxext6 \
    libxft2 \
    libxau6 \
    libxdmcp6 \
    libice6 \
    libsm6 \
    libxt6 \
    libfontconfig1 \
    libfreetype6 \
    libncurses5 \
    libstdc++6 \
    lib32z1 \
    libbz2-1.0 \
    # Build tools
    make \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    wget \
    unzip \
    # Utilities
    procps \
    htop \
    vim \
    tree \
    time \
    jq \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Quartus environment
# ---------------------------------------------------------------------------
ENV QUARTUS_HOME=/opt/intelFPGA_pro/quartus
ENV QUARTUS_ROOTDIR=/opt/intelFPGA_pro/quartus
ENV PATH="${QUARTUS_HOME}/bin:${QUARTUS_HOME}/sopc_builder/bin:${QUARTUS_HOME}/syscon/bin:${PATH}"
ENV LM_LICENSE_FILE=/opt/flexlm/license.dat
ENV QSYS_ROOTDIR=/opt/intelFPGA_pro/qsys

# Quartus needs /tmp with exec permissions
RUN chmod 1777 /tmp

# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------
WORKDIR /workspace

# ---------------------------------------------------------------------------
# Build scripts
# ---------------------------------------------------------------------------
COPY scripts/ /usr/local/bin/
RUN chmod +x /usr/local/bin/*.sh

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["build-all"]
