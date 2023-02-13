#!/bin/bash

SPLIT=${1:-100}
printf -v HSPLIT "%04x" "$SPLIT"

head -c $((8 + ${SPLIT})) t10k-labels-idx1-ubyte | xxd -p | sed "/2710/s//${HSPLIT}/" | xxd -r -p > "t${SPLIT}-labels-idx1-ubyte"
head -c $((16 + ${SPLIT}*28*28)) t10k-images-idx3-ubyte | xxd -p | sed "/2710/s//${HSPLIT}/" | xxd -r -p > "t${SPLIT}-images-idx3-ubyte"
