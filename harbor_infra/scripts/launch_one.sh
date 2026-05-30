#!/bin/bash
set +e
cd /root/ai-scientist-v3
export PATH="/root/venv/bin:$PATH"
nohup ./run.sh ideas/idea_cpu_calibration.json --model deepseek-v4-pro --env e2b --timeout 14400 --use-upstream-agent \
    > /root/ai_scientist_idea_cpu_calibration.log 2>&1 &
echo "launched: calibration (pid $!)"
disown
exit 0
