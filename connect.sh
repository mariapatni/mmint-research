#!/bin/bash

# === CONFIG ===
VPN_BIN="/opt/cisco/secureclient/bin/vpn"
VPN_SERVER="umvpn.umnet.umich.edu/umvpn-all-traffic-alt"
REPO_DIR="$HOME/mmint-research"
BRANCH="main"

# === 1. Kill any existing VPN GUI process ===
pkill -f vpnui 2>/dev/null

# === 2. Run the VPN CLI connection command ===
echo "🔐 Connecting to UMICH VPN..."
sudo "$VPN_BIN" connect "$VPN_SERVER"

# === 3. Wait for VPN interface to appear ===
echo "🔄 Waiting for VPN tunnel (cscotun0)..."
for i in {1..10}; do
    sleep 2
    if ip a show cscotun0 &>/dev/null; then
        echo "✅ VPN connected."
        break
    fi
    if [[ $i -eq 10 ]]; then
        echo "❌ VPN did not connect. Exiting."
        exit 1
    fi
done

# === 4. Get VPN IP ===
VPN_IP=$(ip -4 addr show cscotun0 | grep inet | awk '{print $2}' | cut -d/ -f1)
if [ -z "$VPN_IP" ]; then
    echo "❌ Failed to get VPN IP."
    exit 1
fi
echo "📡 VPN IP: $VPN_IP"

# === 5. Push to GitHub ===
cd "$REPO_DIR" || { echo "❌ Git repo not found at $REPO_DIR"; exit 1; }

echo "$VPN_IP" > vpn_ip.txt
git add vpn_ip.txt
git commit -m "Update VPN IP to $VPN_IP"
git push origin "$BRANCH"

echo "✅ Pushed IP to GitHub."
