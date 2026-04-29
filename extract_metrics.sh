#!/bin/bash

INPUT_DIR=$1
OUTPUT_FILE=$2

> ${OUTPUT_FILE}.txt

for file in $INPUT_DIR/*.txt; do
    fname=$(basename $file)

    ########################################
    # IPC (Throughput)
    ########################################
    IPC=$(grep "Simulation finished" $file | grep -o 'IPC: [0-9.]*' | grep -o '[0-9.]*')

    ########################################
    # L1D CACHE
    ########################################
    L1_BLOCK=$(awk '/cpu0_L1D/,/cpu0_L2C/' $file)

    L1_HIT=$(echo "$L1_BLOCK" | grep -i "hit" | head -1 | grep -o '[0-9]\+')
    L1_MISS=$(echo "$L1_BLOCK" | grep -i "miss" | head -1 | grep -o '[0-9]\+')

    [[ -z "$L1_HIT" ]] && L1_HIT=0
    [[ -z "$L1_MISS" ]] && L1_MISS=0

    ########################################
    # PREFETCH
    ########################################
    PREF_LINE=$(grep "Num Con Useful" $file | head -1)

    USEFUL=$(echo "$PREF_LINE" | grep -o 'Useful: *[0-9]*' | grep -o '[0-9]*')
    USELESS=$(echo "$PREF_LINE" | grep -o 'Useless: *[0-9]*' | grep -o '[0-9]*')
    ACC=$(echo "$PREF_LINE" | grep -o 'Accuracy: *[0-9.]*' | grep -o '[0-9.]*')

    [[ -z "$USEFUL" ]] && USEFUL=0
    [[ -z "$USELESS" ]] && USELESS=0
    [[ -z "$ACC" ]] && ACC=0

    TOTAL_PREF=$((USEFUL + USELESS))

    ########################################
    # POLLUTION
    ########################################
    if [[ $TOTAL_PREF -gt 0 ]]; then
        POLLUTION=$(awk "BEGIN {printf \"%.2f\", ($USELESS/$TOTAL_PREF)*100}")
    else
        POLLUTION=0
    fi

    ########################################
    # COVERAGE
    ########################################
    if [[ $L1_MISS -gt 0 ]]; then
        COVERAGE=$(awk "BEGIN {printf \"%.2f\", ($USEFUL/$L1_MISS)}")
    else
        COVERAGE=0
    fi

    ########################################
    # DRAM TRAFFIC (approx)
    ########################################
    DRAM=$(grep -i "dram" $file | grep -o '[0-9]\+' | head -1)
    [[ -z "$DRAM" ]] && DRAM=0

    ########################################
    # PRINT
    ########################################
    {
        echo "========================================"
        echo "Benchmark: $fname"
        echo "----------------------------------------"
        echo "Throughput (IPC): $IPC"
        echo "L1 Hits: $L1_HIT"
        echo "L1 Misses: $L1_MISS"
        echo "Prefetch Useful: $USEFUL"
        echo "Prefetch Useless: $USELESS"
        echo "Accuracy: $ACC %"
        echo "Cache Pollution: $POLLUTION %"
        echo "Coverage: $COVERAGE %"
        echo "Power: Not available in ChampSim"
        echo "Energy: Not available in ChampSim"
        echo ""
    } >> ${OUTPUT_FILE}.txt

done

echo "----------------------------------------"
echo "✅ Done. Output saved to ${OUTPUT_FILE}.txt"
echo "----------------------------------------"