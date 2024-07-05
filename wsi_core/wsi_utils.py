import h5py
import numpy as np
import os
import pdb
from PIL import Image
import math
import cv2


def StitchCoords(hdf5_file_path, wsi_object, downscale=16, draw_grid=False, bg_color=(0,0,0), alpha=-1):
    wsi = wsi_object.getOpenSlide()
    vis_level = wsi.get_best_level_for_downsample(downscale)
    file = h5py.File(hdf5_file_path, 'r')
    dset = file['coords']
    coords = dset[:]
    w, h = wsi.level_dimensions[0]

    print('start stitching {}'.format(dset.attrs['name']))
    print('original size: {} x {}'.format(w, h))

    w, h = wsi.level_dimensions[vis_level]

    print('downscaled size for stiching: {} x {}'.format(w, h))
    print('number of patches: {}'.format(len(coords)))
    
    patch_size = dset.attrs['patch_size']
    patch_level = dset.attrs['patch_level']
    print('patch size: {}x{} patch level: {}'.format(patch_size, patch_size, patch_level))
    patch_size = tuple((np.array((patch_size, patch_size)) * wsi.level_downsamples[patch_level]).astype(np.int32))
    print('ref patch size: {}x{}'.format(patch_size, patch_size))

    if w*h > Image.MAX_IMAGE_PIXELS: 
        raise Image.DecompressionBombError("Visualization Downscale %d is too large" % downscale)
    
    if alpha < 0 or alpha == -1:
        heatmap = Image.new(size=(w,h), mode="RGB", color=bg_color)
    else:
        heatmap = Image.new(size=(w,h), mode="RGBA", color=bg_color + (int(255 * alpha),))
    
    heatmap = np.array(heatmap)
    heatmap = DrawMapFromCoords(heatmap, wsi_object, coords, patch_size, vis_level, indices=None, draw_grid=draw_grid)
    
    file.close()
    return heatmap

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
