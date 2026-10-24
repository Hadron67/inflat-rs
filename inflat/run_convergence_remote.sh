#!/bin/sh
set -e

export PATH="/usr/local/bin:$HOME/.cargo/bin:$PATH"
cd "$HOME/computations-cfy"

mkdir -p out/gauss_bonnet_convergence

# The server network is slow; all crates are already in the local cargo cache
# (plotly was uploaded manually), so build offline.
cargo build --release --offline --bin gauss_bonnet_convergence > out/gauss_bonnet_convergence/build.log 2>&1

BIN=./target/release/gauss_bonnet_convergence
BASE=baseline/case1-final.output.npz

# Run the two cases strictly one after another.
$BIN --baseline "$BASE" --run --config dt1_n64 > out/gauss_bonnet_convergence/dt1_n64.run.log 2>&1
$BIN --baseline "$BASE" --run --config n128 > out/gauss_bonnet_convergence/n128.run.log 2>&1

# Cheap post-processing: now that both bincode files exist, regenerate the
# time-step comparison with both points.
$BIN --baseline "$BASE" --plot-only --group timestep > out/gauss_bonnet_convergence/timestep.plot.log 2>&1
