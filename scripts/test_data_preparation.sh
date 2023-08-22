#!/bin/bash
# script to copy scans_test data
# e.g. bash scripts/test_data_preparation.sh /mnt/ScanNet/public/v2/scans /mnt/raid/scanpb/ScanNet_Source/public/v2

# path to scannet v2 scans folder that contains *_vh_clean_2.ply for evaluation
SOURCE_PATH=$1
POST_FIX="_vh_clean_2.ply"

# path to preprocessed scannet v2 dataset that will be used by the program
DEST_PATH=$2

# create scans_test folder
mkdir -p "$DEST_PATH/scans_test"

scenes=()

# reading the test list from dest path
while read -r line; do
  scenes+=("$line")
done < $DEST_PATH/scans_test.txt

# copy all scenes to destination
for s in "${scenes[@]}"; do
  echo $SOURCE_PATH/$s/$s$POST_FIX
  echo $DEST_PATH/scans_test/$s
  cp -r "$DEST_PATH/scans/$s" "$DEST_PATH/scans_test/$s"
  cp "$SOURCE_PATH/$s/$s$POST_FIX" "$DEST_PATH/scans_test/$s/$s$POST_FIX"
done
