#!/bin/bash
#SBATCH --job-name=gm3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64000
#SBATCH --time=48:00:00
#SBATCH --output=fastapi_server_%j.out
#SBATCH --error=fastapi_server_%j.err
#SBATCH --gpus=b200:1
#SBATCH --partition=hpg-b200

module load conda
conda activate tea

python -u -m uvicorn file_name_no_py:app --host 0.0.0.0 --port 8000 --workers 1 --timeout-keep-alive 120

date

