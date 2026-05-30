#!/bin/bash
set +e
pkill -9 -f review_api.py
sleep 2
cd /root/HarborTrajectoryGen_v2
nohup /root/venv/bin/python3 -u review_api.py --port 8082 > /root/review_api.log 2>&1 &
disown
sleep 3
pgrep -fa review_api
exit 0
