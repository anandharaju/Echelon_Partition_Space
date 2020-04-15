#!/bin/bash
#SBATCH --account=def-wangk
#SBATCH --time=3:00:00
#SBATCH --job-name=TAIL_ABS_RELAX

#SBATCH --cpus-per-task=8
#SBATCH --cores-per-socket=8
#SBATCH --gres=gpu:2
#SBATCH --mem=6G

#SBATCH -o /home/aduraira/projects/def-wangk/aduraira/cc_out/job%j.out
#SBATCH -e /home/aduraira/projects/def-wangk/aduraira/cc_out/job%j.err
#SBATCH --mail-user=aduraira@sfu.ca
#SBATCH --mail-type=ALL

python main.py
