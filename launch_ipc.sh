#!/bin/bash
# Assume hosts are as follows:
"""
 127.0.0.1   hbmpc_0
 127.0.0.1   hbmpc_1
 127.0.0.1   hbmpc_2
 127.0.0.1   hbmpc_3
"""

CMD="python -m honeybadgermpc.ipc 4 1"
set -x
tmux new-session     "ALIAS=hbmpc_0 ${CMD}; sh" \; \
     splitw -h -p 50 "ALIAS=hbmpc_1 ${CMD}; sh" \; \
     splitw -v -p 50 "ALIAS=hbmpc_2 ${CMD}; sh" \; \
     selectp -t 0 \; \
     splitw -v -p 50 "ALIAS=hbmpc_3 ${CMD}; sh"
