#!/bin/bash
#SBATCH --account=def-hefeeda
#SBATCH --time=03:00:00
#SBATCH --job-name=testthetrain_multiscale_test_with_gpu
#SBATCH --cpus-per-task=12
#SBATCH --cores-per-socket=12
#SBATCH --gres=gpu:1
#SBATCH --mem=122G
#SBATCH -o /home/neha512/project/test_by_gpu/job%j.out
#SBATCH -e /home/neha512/project/job%j.err
#SBATCH --mail-user=nsa84r@sfu.ca
#SBATCH --mail-type=ALL

#!/bin/bash
SBATCH --account=def-wangk
SBATCH --time=03:00:00
SBATCH --job-name=echelon_test_single_fold
SBATCH --cpus-per-task=12
SBATCH --cores-per-socket=12
SBATCH --gres=gpu:1
SBATCH --mem=122G
SBATCH -o /home/aduraira/projects/def-wangk/aduraira/cc/job%j.out
SBATCH -e /home/aduraira/projects/def-wangk/aduraira/cc/job%j.err
SBATCH --mail-user=aduraira@sfu.ca
SBATCH --mail-type=ALL
