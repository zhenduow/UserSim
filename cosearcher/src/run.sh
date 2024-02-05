#!/bin/bash
for f in /raid/zhenduow/USi/output/*.csv
do
    echo $f
    python3 src/main.py --output_file_path $f > "${f}.json"
done