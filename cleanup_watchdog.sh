#!/bin/bash
# Safety Watchdog for VERA
# Run in terminal: bash cleanup_watchdog.sh

while true; do
    # Get available space on the root partition in GB
    AVAILABLE=$(df -BG / | tail -1 | awk '{print $4}' | sed 's/G//')        
    echo "$(date): Available space: ${AVAILABLE}G"

    # If space drops below 40GB, start emergency cleanup
    if [ "$AVAILABLE" -lt 40 ]; then
        echo "WARNING: Space low. Deleting"
        
        # 1. Clear the OSMnx cache (safest to delete)
        rm -rf ./cache
        
        # 2. Clear Step 01 and 02 outputs if you are already at Step 04
        if [ -d "output/03_sampled" ]; then
            rm -f output/01_raw/*.csv
            rm -f output/02_cleaned/*.csv
        fi
    fi

    # Sleep for 5 minutes 
    sleep 300
done