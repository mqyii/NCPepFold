#!/bin/bash

# 设定文件夹A的路径
folderA="/home/light/mqy/ncaa/data/training_set_v1/linear/missing"

# 定义一个子文件夹名的数组
declare -a folderList=("3bh8C" "3bh9C" "4lorB" "4lorD" "1g6gF" "1o9sK" "2x4wB" "2x4xD" "2x4xF" "2x4yB" "2x4yL" "2y5tG" "3avrB" "3ejhE" "3hnaQ" "4apjP" "4gagP" "4glrA")
# 遍历文件夹A
for subdir in "$folderA"/*; do
    if [ -d "$subdir" ]; then  # 确保是一个目录
        folderName=$(basename "$subdir")
        # 检查子文件夹名是否在列表中
        for validFolder in "${folderList[@]}"; do
            if [ "$folderName" == "$validFolder" ]; then
                # 如果在列表中，则运行pdbfixer
                echo "Running pdbfixer on $folderName"
                pdbfixer "$subdir/${folderName}_clean.pdb" --add-atoms=heavy --output="$subdir/${folderName}_clean_1.pdb"
            fi
        done
    fi
done
