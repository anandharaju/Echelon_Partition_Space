#!/bin/bash
#SBATCH --account=def-wangk
#SBATCH --time=02:00:00
#SBATCH --job-name=echelon_boosting_test_5th_fold

#SBATCH --cpus-per-task=12
#SBATCH --cores-per-socket=12
#SBATCH --gres=gpu:1
#SBATCH --mem=24G

#SBATCH -o /home/aduraira/projects/def-wangk/aduraira/Echelon_Linux/src/job%j.out
#SBATCH -e /home/aduraira/projects/def-wangk/aduraira/Echelon_Linux/src/job%j.err
#SBATCH --mail-user=aduraira@sfu.ca
#SBATCH --mail-type=ALL

python main.py
