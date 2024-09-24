import h5py
import os
import numpy as np
import matplotlib.pyplot as plt

def import_h5frm_to_nparr(src_frm_path: os.PathLike) -> np.ndarray:

    with h5py.File(src_frm_path) as f:
        all_frames_arr = np.array(f['Data3D']['Images'])   
    return all_frames_arr

def plot_single_frame(single_frame_arr: np.ndarray, 
                      out_png_path: os.PathLike, 
                      vmin: str = '100', 
                      vmax: str = '1000') -> None:

    plt.imshow(single_frame_arr, cmap='gray', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Intensity')
    plt.xlabel('X-axis on camera')
    plt.ylabel('Y-axis on camera')
    plt.savefig(out_png_path, dpi=4096)
    plt.close()


if __name__ == '__main__':
    
    #-----------------------------------
    #---------IMPORTING-----------------
    #-----------------------------------

    SOURCE_H5_FRM_FILE = r"C:\VS_CODE\outgoing\dark_01\dark_01__frm.h5"

    all_frames = import_h5frm_to_nparr(SOURCE_H5_FRM_FILE)

    # all_frames is now a 3D numpy array, containing all images. 
    # For example, if there were 180 images in the dataset, each measured on a 1920x1920 16-bit detector, 
    # it would be of shape (180,1920,1920) with integer vals ranging from 0 to 65535.

    #-----------------------------------
    #---------PLOTTING------------------
    #-----------------------------------

    SELECTED_FRAME = 4
    OUTPUT_PNG_IMG_FILE = f'{SELECTED_FRAME}_frame_test.png'

    selected_frame = all_frames[SELECTED_FRAME]
    plot_single_frame(single_frame_arr=selected_frame,
                      out_png_path=OUTPUT_PNG_IMG_FILE)

