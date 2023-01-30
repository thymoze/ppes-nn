#!/bin/bash
TEMP_DIR=/tmp/tmp.NkpqUvjSwS #$(mktemp -d)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

wget -P "$TEMP_DIR" http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget -P "$TEMP_DIR" http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget -P "$TEMP_DIR" http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget -P "$TEMP_DIR" http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

for f in "$TEMP_DIR"/*.gz; do gzip -d "$f"; done
cp "$TEMP_DIR"/* "$SCRIPT_DIR/"