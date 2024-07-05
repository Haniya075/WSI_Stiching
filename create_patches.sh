# %load /kaggle/working/WSI-finetuning/create_patches.sh
CUDA_VISIBLE_DEVICES=0 python create_patches_fp.py --source /kaggle/input/wsi-images/WSI --save_dir ./patches/Camelyon16_patch256_ostu --patch_level 1 --patch_size 256 --step_size 256 --seg --patch --stitch --use_ostu
