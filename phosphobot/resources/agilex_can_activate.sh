#!/bin/bash

# agilex_can_activate.sh
set -euo pipefail

TARGET_INTERFACE="${1:-}"
DEFAULT_BITRATE="${2:-1000000}"

log_success() { echo -e "\e[32m[SUCCESS] $1\e[0m"; }
log_error() { echo -e "\e[31m[ERROR] $1\e[0m" >&2; }
log_info() { echo -e "\e[33m[INFO] $1\e[0m"; }


# Check if required utilities are installed
for pkg in ethtool can-utils; do
  if ! dpkg -s "$pkg" &> /dev/null; then
    log_error "Error: $pkg not detected. Please install it: sudo apt update && sudo apt install $pkg"
    exit 1
  fi
done
log_success "All required packages are installed."

# Retrieve CAN interfaces
mapfile -t CAN_INTERFACES < <(ip -br link show type can | awk '{print $1}')

if [ -n "$TARGET_INTERFACE" ]; then
    FOUND_INTERFACE=0
    for iface in "${CAN_INTERFACES[@]}"; do
        if [ "$iface" = "$TARGET_INTERFACE" ]; then
            FOUND_INTERFACE=1
            CAN_INTERFACES=("$TARGET_INTERFACE")
            break
        fi
    done

    if [ "$FOUND_INTERFACE" -eq 0 ]; then
        log_error "CAN interface '$TARGET_INTERFACE' was not detected."
        exit 1
    fi
fi

CAN_COUNT=${#CAN_INTERFACES[@]}

if [ "$CAN_COUNT" -eq 0 ]; then
    log_error "No CAN interfaces detected!"
    exit 1
fi
log_success "Detected $CAN_COUNT CAN interface(s)."

# Configure interfaces
for iface in "${CAN_INTERFACES[@]}"; do
    BUS_INFO=$(sudo ethtool -i "$iface" | grep "bus-info" | awk '{print $2}')

    log_info "Configuring $iface (USB: $BUS_INFO)..."

    sudo ip link set "$iface" down
    sudo ip link set "$iface" type can bitrate "$DEFAULT_BITRATE"
    sudo ip link set "$iface" up
    log_success "Configured $iface"
done

log_success "All CAN interfaces active"
# Return success
exit 0
