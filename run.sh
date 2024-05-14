pip install -r requirements.txt

wandb init
python EntropySetup.py build_ext --inplace
pip install .

# Run train
# python -m src.scripts.train --config_file configs/train/residual_ae.yaml

# Run inference
# python -B -m src.scripts.inference --config_file ./configs/inference/residual_ae.yaml