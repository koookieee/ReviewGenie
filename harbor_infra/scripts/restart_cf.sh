#!/bin/bash
set +e
pkill -9 -f "cloudflared tunnel"
sleep 3
nohup cloudflared tunnel run search_api > /root/cloudflared.log 2>&1 &
disown
sleep 5
pgrep -fa cloudflared
exit 0
