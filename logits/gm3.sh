#!/bin/bash
#SBATCH --job-name=gm3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUREMAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --time=48:00:00
#SBATCH --output=fastapi_server_%j.out
#SBATCH --error=fastapi_server_%j.err
#SBATCH --gres=gpu:1               # see the other file
#SBATCH --partition=hpg-turin     # better to use B200

module load conda
conda activate tea

export HF_HOME=/blue/iruchkin/YOUR_USERNAME/.cache/huggingface # modify this to your own username
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u -m uvicorn gemma3:app --host 0.0.0.0 --port 8000 --workers 1 --timeout-keep-alive 120

date

