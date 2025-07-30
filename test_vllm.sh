#!/bin/bash

python benchmark.py --transcript-file meetings_10_batches.json  --inference-type "vllm" --request-handle "parallel"