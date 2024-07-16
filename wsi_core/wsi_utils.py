import h5py
import numpy as np
import os
import pdb
from PIL import Image
import math
import cv2
from Image_Crop import crop_image

Image.MAX_IMAGE_PIXELS = None

def DrawGrid(img, coord, shape, thickness=2, color=(0,0,0,255)):
    cv2.rectangle(img, tuple(np.maximum([0, 0], coord-thickness//2)), tuple(coord - thickness//2 + np.array(shape)), (0, 0, 0, 255), thickness=thickness)
    return img

def DrawMapFromCoords(canvas, wsi_object, coords, patch_size, vis_level ,indices=None, verbose=1, draw_grid=True):
    downsamples = wsi_object.wsi.level_downsamples[vis_level]
    if indices is None:
        indices = np.arange(len(coords))
    total = len(indices)
    if verbose > 0:
        ten_percent_chunk = math.ceil(total * 0.1)
        
    patch_size = tuple(np.ceil((np.array(patch_size)/np.array(downsamples))).astype(np.int32))
    print('downscaled patch size: {}x{}'.format(patch_size[0], patch_size[1]))
    
    for idx in range(total):
        if verbose > 0:
            if idx % ten_percent_chunk == 0:
                print('progress: {}/{} stitched'.format(idx, total))
        
        patch_id = indices[idx]
        coord = coords[patch_id]
        patch = np.array(wsi_object.wsi.read_region(tuple(coord), vis_level, patch_size).convert("RGB"))
        coord = np.ceil(coord / downsamples).astype(np.int32)
        canvas_crop_shape = canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3].shape[:2]
        canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3] = patch[:canvas_crop_shape[0], :canvas_crop_shape[1], :]
        if draw_grid:
            DrawGrid(canvas, coord, patch_size)
            print("TERMINATEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
        
        #cropped_save_dir = r"./patches/cropped"
        #if not os.path.exists(cropped_save_dir):
            #os.makedirs(cropped_save_dir)
        #cropped_path = os.path.join(cropped_save_dir, count+'.jpg')
        #img=crop_image(Image.fromarray(canvas), cropped_path)
    return Image.fromarray(canvas)


def StitchCoords(hdf5_file_path, wsi_object,downscale=5,draw_grid=False, bg_color=(0,0,0), alpha=-1):
    wsi = wsi_object.getOpenSlide()
    
    while True:
        try:
            vis_level = wsi.get_best_level_for_downsample(downscale)
            file = h5py.File(hdf5_file_path, 'r')
            dset = file['coords']
            coords = dset[:]
            w, h = wsi.level_dimensions[0]

            print('start stitching {}'.format(dset.attrs['name']))
            print('original size: {} x {}'.format(w, h))

            w, h = wsi.level_dimensions[vis_level]

            print('downscaled size for stitching: {} x {}'.format(w, h))
            print('number of patches: {}'.format(len(coords)))

            patch_size = dset.attrs['patch_size']
            patch_level = dset.attrs['patch_level']
            print('patch size: {}x{} patch level: {}'.format(patch_size, patch_size, patch_level))
            patch_size = tuple((np.array((patch_size, patch_size)) * wsi.level_downsamples[patch_level]).astype(np.int32))
            print('ref patch size: {}x{}'.format(patch_size, patch_size))
            
            max_pixels = float('inf') if Image.MAX_IMAGE_PIXELS is None else Image.MAX_IMAGE_PIXELS

            if w * h >max_pixels: 
                raise Image.DecompressionBombError("Visualization Downscale %d is too large" % downscale)

            if alpha < 0 or alpha == -1:
                heatmap = Image.new(size=(w, h), mode="RGB", color=bg_color)
            else:
                heatmap = Image.new(size=(w, h), mode="RGBA", color=bg_color + (int(255 * alpha),))

            heatmap = np.array(heatmap)
            heatmap = DrawMapFromCoords(heatmap, wsi_object, coords, patch_size, vis_level ,indices=None, draw_grid=draw_grid)
            
            file.close()
            return heatmap
        
        except Image.DecompressionBombError as e:
            print(f"Error: {e}")
            downscale += 1
            print(f"Increasing downscale to {downscale} and retrying...")



def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path
