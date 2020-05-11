#!/bin/bash
#SBATCH --account=def-wangk
#SBATCH --time=12:00:00
#SBATCH --job-name=G_DS1_0_16G

#SBATCH --cpus-per-task=8
#SBATCH --cores-per-socket=8
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=16G

#SBATCH -o /home/aduraira/projects/def-wangk/aduraira/cc_out/job%j.out
#SBATCH -e /home/aduraira/projects/def-wangk/aduraira/cc_out/job%j.err
#SBATCH --mail-user=aduraira@sfu.ca
#SBATCH --mail-type=ALL

python main.py
