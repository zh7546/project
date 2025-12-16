## Training

1. To download GoPro training and testing data, run
```
python download_data.py --data train-test
```

2. Generate image patches from full-resolution training images of GoPro dataset
```
python generate_patches_gopro.py 
```

3. To train Restormer, run
```
cd Restormer
./train.sh Motion_Deblurring/Options/Deblurring_Restormer.yml
```

#### Testing on GoPro dataset

- Download GoPro testset, run
```
python download_data.py --data test --dataset GoPro
```

- Testing
```
python test.py --dataset GoPro
```

#### Testing on HIDE dataset

- Download HIDE testset, run
```
python download_data.py --data test --dataset HIDE
```

- Testing
```
python test.py --dataset HIDE
```

#### Testing on RealBlur-J dataset

- Download RealBlur-J testset, run
```
python download_data.py --data test --dataset RealBlur_J
```

- Testing
```
python test.py --dataset RealBlur_J
```

#### Testing on RealBlur-R dataset

- Download RealBlur-R testset, run
```
python download_data.py --data test --dataset RealBlur_R
```

- Testing
```
python test.py --dataset RealBlur_R
```

#### To reproduce PSNR/SSIM scores of the paper (Table 2) on GoPro and HIDE datasets, run this MATLAB script

```
evaluate_gopro_hide.m 
```

#### To reproduce PSNR/SSIM scores of the paper (Table 2) on RealBlur dataset, run

```
evaluate_realblur.py 
```
