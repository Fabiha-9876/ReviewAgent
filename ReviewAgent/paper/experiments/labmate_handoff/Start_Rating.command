#!/bin/bash
# Double-click this file to start the rating tool.
# (If macOS asks "open from unidentified developer", right-click → Open.)

cd "$(dirname "$0")"
clear
echo "Loading rating tool..."
echo
python3 rate.py
echo
echo "Press Return to close this window."
read
