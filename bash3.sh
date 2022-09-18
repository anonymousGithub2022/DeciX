#!/bin/bash


CUDA_VISIBLE_DEVICES=3 python generate_explanation.py --task=2
CUDA_VISIBLE_DEVICES=3 python evaluate_explanation.py --task=2