#!/bin/bash
# Everything still to run, in priority order. Each part skips finished results, so just rerun after a stop.
cd /home/claude/comp0245_bc
./run_objects.sh >> objects.log 2>&1
./run_nav.sh >> nav.log 2>&1
./run_final.sh >> final.log 2>&1
