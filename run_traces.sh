#!/bin/bash

TRACE_DIR=~/traces
BIN=./bin/champsim_8core_gaze

WARMUP=200000000
SIM=200000000

mkdir -p results

TRACES=(

# ---- compute (3) ----
compute_fp_1_new.xz
compute_fp_6_new.xz
compute_int_11_new.xz

# ---- ligra (3) ----
ligra_BFS.com-lj.ungraph.gcc_6.3.0_O3.drop_500M.length_250M.champsimtrace.xz
ligra_PageRank.com-lj.ungraph.gcc_6.3.0_O3.drop_500M.length_250M.champsimtrace.xz
ligra_BellmanFord.com-lj.ungraph.gcc_6.3.0_O3.drop_4000M.length_250M.champsimtrace.xz

# ---- gap (2) ----
gap.bfs.twitter-10B.champsimtrace.xz
gap.pr.web-10B.champsimtrace.xz

# ---- parsec (2) ----
parsec_2.1.facesim.simlarge.prebuilt.drop_1500M.length_250M.champsimtrace.xz
parsec_2.1.streamcluster.simlarge.prebuilt.drop_4750M.length_250M.champsimtrace.xz

# ---- srv (3) ----
srv_1_new.xz
srv_27_new.xz
srv_60_new.xz

)

for trace in "${TRACES[@]}"; do
    echo "Running $trace"

    $BIN \
      --warmup_instructions $WARMUP \
      --simulation_instructions $SIM \
      --trace $TRACE_DIR/$trace \
      > results/${trace}.txt

    echo "Finished $trace"
done

