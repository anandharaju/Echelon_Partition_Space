#!/bin/bash
#SBATCH --account=def-wangk
#SBATCH --time=4:00:00
#SBATCH --job-name=PDS1_BIDIV_P100

#SBATCH --cpus-per-task=1
#SBATCH --cores-per-socket=1
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=8G

#SBATCH -o /home/aduraira/projects/def-wangk/aduraira/cc_out/job%j.out
#SBATCH -e /home/aduraira/projects/def-wangk/aduraira/cc_out/job%j.err
#SBATCH --mail-user=aduraira@sfu.ca
#SBATCH --mail-type=ALL

python main.py
