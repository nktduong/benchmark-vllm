#!/bin/bash

python benchmark.py --transcript-file different_10_batches.json  --inference-type "vllm" --request-handle "parallel" --request-handle "parallel"