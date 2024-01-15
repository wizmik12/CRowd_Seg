#!/bin/bash

export CUDA_DEVICE_ORDER="PCI_BUS_ID"    
export CUDA_VISIBLE_DEVICES=0
source activate seg_crowd_env

nohup python -u ../src/main.py -dc config.yaml -ef supervised/1 > supervised/1/log.out 
nohup python -u ../src/main.py -dc config.yaml -ef supervised/2 > cr_image/2/log.out 
nohup python -u ../src/main.py -dc config.yaml -ef supervised/3 > supervised/3/log.out 

nohup python -u ../src/main.py -dc config.yaml -ef cr_image/1 > cr_image/1/log.out 
nohup python -u ../src/main.py -dc config.yaml -ef cr_image/2 > cr_image/2/log.out 
nohup python -u ../src/main.py -dc config.yaml -ef cr_image/3 > cr_image/3/log.out 

nohup python -u ../src/main.py -dc config.yaml -ef cr_pixel/1 > cr_pixel/1/log.out 
nohup python -u ../src/main.py -dc config.yaml -ef cr_pixel/2 > cr_pixel/2/log.out 
nohup python -u ../src/main.py -dc config.yaml -ef cr_pixel/3 > cr_pixel/3/log.out 

nohup python -u ../src/main.py -dc config.yaml -ef cr_global/1 > cr_global/1/log.out 
nohup python -u ../src/main.py -dc config.yaml -ef cr_global/2 > cr_global/2/log.out 
nohup python -u ../src/main.py -dc config.yaml -ef cr_global/3 > cr_global/3/log.out 

nohup python -u ../src/main.py -dc config.yaml -ef mv/1 > mv/1/log.out 
nohup python -u ../src/main.py -dc config.yaml -ef mv/2 > mv/2/log.out 
nohup python -u ../src/main.py -dc config.yaml -ef mv/3 > mv/3/log.out 

nohup python -u ../src/main.py -dc config.yaml -ef staple/1 > staple/1/log.out 
nohup python -u ../src/main.py -dc config.yaml -ef staple/2 > staple/2/log.out 
nohup python -u ../src/main.py -dc config.yaml -ef staple/3 > staple/3/log.out 
