#!/bin/bash


CUDA_VISIBLE_DEVICES=1 python generate_explanation.py --task=0
CUDA_VISIBLE_DEVICES=1 python evaluate_explanation.py --task=0

