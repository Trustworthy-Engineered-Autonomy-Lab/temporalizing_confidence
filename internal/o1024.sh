#!/bin/bash
#SBATCH --job-name=gm31024
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mail
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64000
#SBATCH --time=48:00:00
#SBATCH --output=fastapi_server_%j.out
#SBATCH --error=fastapi_server_%j.err
#SBATCH --gres=gpu:2               # see the other file
#SBATCH --partition=hpg-turin     # better to use B200

module load conda
conda activate tea

python -u -m uvicorn o1024:app --host 0.0.0.0 --port 8000 --workers 1 --timeout-keep-alive 120

date

