# Solving The Checkerboard Artifacts In Image Deconvolution

This repository is implementation of the a final project for IDC's Image Understanding with Deep Learning course 2020.

## Requirements 
- PyTorch 1.1.0
- TensorBoard 1.14.0
- Numpy 1.15.4
- Pillow-SIMD 5.3.0.post1
- h5py 2.8.0
- tqdm 4.30.0

## Prepare dataset

To prepare dataset used in experiments, first download dataset files from this [link](https://data.vision.ee.ethz.ch/cvl/DIV2K) and organize it as shown below.

```bash
/YOUR_STORAGE_PATH/DIV2K
├── DIV2K_train_HR
├── DIV2K_train_LR_bicubic
│   └── X2
│   └── X3
│   └── X4
├── DIV2K_valid_HR
├── DIV2K_valid_LR_bicubic
│   └── X2
│   └── X3
│   └── X4
├── DIV2K_train_HR.zip
├── DIV2K_train_LR_bicubic_X2.zip
├── DIV2K_train_LR_bicubic_X3.zip
├── DIV2K_train_LR_bicubic_X4.zip
├── DIV2K_valid_HR.zip
├── DIV2K_valid_LR_bicubic_X2.zip
├── DIV2K_valid_LR_bicubic_X3.zip
└── DIV2K_valid_LR_bicubic_X4.zip
```

By default, we use "0001-0800.png" images to train the model and "0801-0900.png" images to validate the training.
All experiments also use images with BICUBIC degradation on RGB space.

## Training

### WDSR Baseline (=WDSR-A) Example

```bash
python train.py --dataset-dir "/YOUR_STORAGE_PATH/DIV2K" \
                --output-dir "/YOUR_STORAGE_PATH/output" \
                --model "WDSR-A" \
                --scale 2 \
                --n-feats 32 \
                --n-res-blocks 16 \
                --expansion-ratio 4 \
                --res-scale 1.0 \
                --lr 1e-3
```

### EDSR with norm deconv Example

```bash
python train.py --dataset-dir "/YOUR_STORAGE_PATH/DIV2K" \
                --output-dir "/YOUR_STORAGE_PATH/output" \
                --model "EDSR-Norm-Deconv" \
                --scale 2 \
                --n-feats 32 \
                --n-res-blocks 16 \
                --expansion-ratio 6 \
                --low-rank-ratio 0.8 \
                --res-scale 1.0 \
                --lr 1e-3
```

If you want to modify more options, see the `core/option.py` file.

## Evaluation

Trained model is evaluated on DIV2K validation 100 images. If you want to use self-ensemble for evaluation, add `--self-ensemble` option.

```bash
python eval.py --dataset-dir "/YOUR_STORAGE_PATH/DIV2K" \
               --checkpoint-file "/YOUR_STORAGE_PATH/output/WDSR-A-f32-b16-r4-x2-best.pth.tar"
```

| model |	upscale | Residual Blocksblock | Block depth |	parameters |	scale	 | PSNR  |
|-------|---------|----------------------|-------------|-------------|---------|-------|
|EDSR|	PixelShuffle|	32|	128|	10780675|	4	|29.04|
|EDSR|	Deconvolution|	32|	128|	9895171|	4	|29.04|
|EDSR|	NormDeconvolution|	32|	128|9895171|	4|	29.07|
|WDSR|	PixelShuffle|	16|	32|	1203312	|4|	28.77|
|WDSR|	Deconvolution|	16|	32|	1188327|	4|	28.86|
|WDSR|	NormDeconvolution|	16|	32|	1188327|	4|	28.9|
|WDSR|	PixelShuffle|	16|	32|	1195605	|3|	30.77|
|WDSR|	Deconvolution|	16|	32|	1186791|	3|	30.76|
|WDSR|	NormDeconvolution|	16|	32|	1186791|	3|	30.72|
|WDSR|	PixelShuffle|	16|	32|	1190100|	2|	34.67|
|WDSR|	Deconvolution|	16|	32|	1186791|	2|34.65|
|WDSR|	NormDeconvolution|	16|	32|	1186791|2|	34.63|



## Results - examples
<center><img src="./images/result_examples.png" /></center>

The differences are very subtle but can be observed. [a] theoriginal image with marked cropped area, [b] the original crop, [c] the crop from a downsampledimages - the input for the network, [d] result of Pixel-Shuffling, [e] result of Deconvolution and [f]result of Normalized Deconvolution


## pretrained models
models are located on a public S3:

https://deconv-pretrained.s3.amazonaws.com/EDSR-Deconv-f128-b32-r4-x4-best.pth.tar  
https://deconv-pretrained.s3.amazonaws.com/EDSR-Deconv-f32-b16-r4-x4-best.pth.tar  
https://deconv-pretrained.s3.amazonaws.com/EDSR-Norm-Deconv-f128-b32-r4-x4-new-best.pth.tar  
https://deconv-pretrained.s3.amazonaws.com/EDSR-Norm-Deconv-f32-b16-r4-x4-best.pth.tar  
https://deconv-pretrained.s3.amazonaws.com/EDSR-f128-b32-r4-x4-best.pth.tar  
https://deconv-pretrained.s3.amazonaws.com/EDSR-f32-b16-r4-x4-best.pth.tar  
https://deconv-pretrained.s3.amazonaws.com/WDSR-A-f32-b16-r4-x2-best.pth.tar  
https://deconv-pretrained.s3.amazonaws.com/WDSR-A-f32-b16-r4-x4-best.pth.tar  
https://deconv-pretrained.s3.amazonaws.com/WDSR-Deconv-f32-b16-r4-x2-best.pth.tar  
https://deconv-pretrained.s3.amazonaws.com/WDSR-Deconv-f32-b16-r4-x3-best.pth.tar  
https://deconv-pretrained.s3.amazonaws.com/WDSR-Deconv-f32-b16-r4-x3_MSE-best.pth.tar  
https://deconv-pretrained.s3.amazonaws.com/WDSR-Deconv-f32-b16-r4-x4-best.pth.tar  
https://deconv-pretrained.s3.amazonaws.com/WDSR-Deconv-f32-b8-r4-x2-best.pth.tar  
https://deconv-pretrained.s3.amazonaws.com/WDSR-Norm-Deconv-f32-b16-r4-x3_MSE-best.pth.tar  
https://deconv-pretrained.s3.amazonaws.com/WDSR-Norm-Deconv-f32-b16-r4-x4-best.pth.tar  