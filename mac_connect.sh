#!/bin/bash

# === CONFIG ===
VPN_BIN="/opt/cisco/secureclient/bin/vpn"
VPN_SERVER="umvpn.umnet.umich.edu/umvpn-all-traffic-alt"
GIT_REPO_DIR="$HOME/Documents/GitHub/mmint-research"
REPO_URL="git@github.com:mmint-research/mmint-research.git"  # replace with your repo
SSH_USER="maria"  # e.g., maria
CONDA_ENV="collection_env"  # Name of the conda environment to activate


# === 1. Clone or update IP repo ===
if [ ! -d "$GIT_REPO_DIR" ]; then
    echo "📥 Cloning repo..."
    git clone "$REPO_URL" "$GIT_REPO_DIR"
else
    echo "🔄 Pulling latest IP from GitHub..."
    cd "$GIT_REPO_DIR" || exit 1
    git pull origin main
fi

# === 2. Read VPN IP ===
VPN_IP=$(cat "$GIT_REPO_DIR/vpn_ip.txt")

if [ -z "$VPN_IP" ]; then
    echo "❌ Failed to read VPN IP from vpn_ip.txt"
    exit 1
fi

echo "📡 VPN IP is: $VPN_IP"

# === 4. SSH into remote machine and activate conda environment ===
echo "🔐 Connecting via SSH and activating conda environment '$CONDA_ENV'..."
ssh -X "$SSH_USER@$VPN_IP" 

