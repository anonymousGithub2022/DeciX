#!/bin/bash


CUDA_VISIBLE_DEVICES=2 python generate_explanation.py --task=1
CUDA_VISIBLE_DEVICES=2 python evaluate_explanation.py --task=1