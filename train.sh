python train.py --dataset-dir "./data/div2k" --output-dir "./results/output" --model "WDSR-A" --scale 2 --n-feats 32 --n-res-blocks 16 --expansion-ratio 4 --res-scale 1.0 --lr 1e-3
python train.py --dataset-dir "./data/div2k" --output-dir "./results/output" --model "WDSR-Deconv" --scale 2 --n-feats 32 --n-res-blocks 16 --expansion-ratio 4 --res-scale 1.0 --lr 1e-3
python train.py --dataset-dir "./data/div2k" --output-dir "./results/output" --model "WDSR-Norm_Deconv" --scale 2 --n-feats 32 --n-res-blocks 16 --expansion-ratio 4 --res-scale 1.0 --lr 1e-3
