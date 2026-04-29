#!/bin/bash

TRACE_DIR=/Users/srikarkopanati/Documents/project/Gaze-Spatial-Prefetcher/traces
BIN=./ChampSim/bin/champsim_1core_no_prefetch

WARMUP=20000000
SIM=50000000

mkdir -p results_no_prefetech

TRACES=(

# ---- compute (1) ----
compute_fp_1_new.xz

# ---- ligra (1) ----
ligra_BFS.com-lj.ungraph.gcc_6.3.0_O3.drop_500M.length_250M.champsimtrace.xz

# ---- gap (1) ----
gap.bfs.twitter-10B.champsimtrace.xz

# ---- parsec (1) ----
parsec_2.1.facesim.simlarge.prebuilt.drop_1500M.length_250M.champsimtrace.xz

# ---- srv (1) ----
srv_1_new.xz

)

for trace in "${TRACES[@]}"; do
    echo "=============================="
    echo "Running $trace"
    echo "=============================="

    START=$(date +%s)

    $BIN \
      --warmup_instructions $WARMUP \
      --simulation_instructions $SIM \
      --trace $TRACE_DIR/$trace \
      > results_no_prefetech/${trace}_no.txt 2>&1

    END=$(date +%s)
    echo "Finished $trace in $((END-START)) seconds"
done