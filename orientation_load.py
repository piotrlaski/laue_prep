import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# a bunch of useful spherical <-> carthesian conversions
def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return [r, theta, phi]
def arr_cartesian_to_spherical(input_array):
    vecs_sph = []
    for vec_cart in input_array:
        vec_sph = cartesian_to_spherical(*vec_cart)
        vecs_sph.append(vec_sph[1:])
    vecs_sph = np.array(vecs_sph)
    return vecs_sph
def spherical_to_cartesian(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return [x, y, z]
def arr_spherical_to_cartesian(input_array):
    vecs_cart = []
    for vec_sph in input_array:
        vec_cart = spherical_to_cartesian(*vec_sph)
        vecs_cart.append(vec_cart)
    vecs_cart = np.array(vecs_cart)
    return vecs_cart
def convert_distance_to_spherical(dist):
    dist = np.clip(dist, 0, 2)
    return 2*np.arcsin(dist/2)
def cart_dist(v,u):
    return np.sqrt(np.sum((v - u) ** 2))

#import functions
def import_h5mono_to_nparr(src_mono_path: os.PathLike) -> tuple[np.ndarray]:

    with h5py.File(src_mono_path, 'r') as mono_h5:
        vectors_mono = np.array(mono_h5['mono']['h_unit'])
        intensities_mono = np.array(mono_h5['mono']['F_squared_meas'])

    return(vectors_mono, intensities_mono)
def fill_mono_symmetry_equivs(vectors_mono: np.ndarray, intensities_mono: np.ndarray, hkl_equivs: tuple[tuple[int]], remove_dups: bool=True) -> tuple[np.ndarray]:

    #why we do this: http://pd.chem.ucl.ac.uk/pdnn/symm2/multj.htm

    #start with the known unique vectors and add all the symmetry equivalent vectors (multiply xyz coords by hkl_equivs contents)
    vectors_mono_full = vectors_mono
    for equiv in hkl_equivs:
        new_vectors = vectors_mono * equiv
        vectors_mono_full = np.concatenate((vectors_mono_full, new_vectors))

    #do the same thing with intensities, just by appending the unique intensities in the same order n times
    intensites_mono_full = intensities_mono
    for i in range (len(hkl_equivs)):
        intensites_mono_full = np.append(intensites_mono_full, intensities_mono)


    #finally, some of these transformations will give duplicate reflections, so we remove them (and the respective intensities)
    if remove_dups:
        vectors_mono_full_no_dups = []
        intensites_mono_full_no_dups = []

        for i, vector in enumerate(vectors_mono_full):
            if any(all(vector == vector_no_dups) for vector_no_dups in vectors_mono_full_no_dups):
                continue
            else:
                vectors_mono_full_no_dups.append(vector)
                intensites_mono_full_no_dups.append(intensites_mono_full[i])
        return(np.array(vectors_mono_full_no_dups), np.array(intensites_mono_full_no_dups))
    else:
        return(np.array(vectors_mono_full), np.array(intensites_mono_full))

def import_h5expt_to_nparr(src_expt_path: os.PathLike) -> tuple[np.ndarray]:

    with h5py.File(src_expt_path, 'r') as expt_h5:
            vectors_expt = np.array(expt_h5['expt']['h_unit_g'])   
            intensities_expt = np.array(expt_h5['expt']['I'])

    return(vectors_expt, intensities_expt)
def import_xxx_to_rotM(src_xxx_path: os.PathLike) -> R:

    with h5py.File(src_xxx_path, 'r') as xxx_file:
        rotM_arr = np.array(xxx_file['Global_Rotation_Matrix']['Direct_Rotation_Matrix'])
        rotM = R.from_matrix(rotM_arr)

    return rotM

#data manipulation functions
def rotate_sphere(data_cart, rot_matrix):
    vecs_rotated = []
    for vec_cart in data_cart:
        vec_rotated = np.matmul(rot_matrix, vec_cart)
        vecs_rotated.append(vec_rotated)
    vecs_rotated = np.array(vecs_rotated)
    return vecs_rotated

#visualization functions
def draw_sphere(data_cart, output_png):

    if np.shape(data_cart)[1] == 4:
        vecs_cart = data_cart[:,:3]
    else:
        vecs_cart = data_cart
    U, V, W = zip(*vecs_cart)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(U, V, W, s=2, marker = '.', lw = 0.1)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_aspect('equal')
    plt.axis('off')

    # Hide grid lines
    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    plt.savefig(output_png, dpi = 1200)
    plt.close()
    return

#testing functions
def import_xxx_vec_data(xxx_file_path):

    with h5py.File(xxx_file_path, 'r') as xxx:
        xxx_expt_vec = np.array(xxx['Global_Rotation_Matrix']['Clustering_Data']['Expt_Clusters'])
        xxx_mono_vec = np.array(xxx['Global_Rotation_Matrix']['Clustering_Data']['Mono_Rays'])

    return(xxx_expt_vec, xxx_mono_vec)

def main_export(SOURCE_H5_EXPT_FILE: os.PathLike,
                SOURCE_H5_MONO_FILE: os.PathLike,
                XXX_FILE: os.PathLike,
                HKLS_EQUIV: tuple,
                DATASET_NAME: str,
                OUTPUT_IMG_DIR: os.PathLike,
                NUMPY_SAVED_FILES_DIR: os.PathLike,
                REMOVE_DUPS = True) -> None:
    
    #----------------------------------------
    #-----------DATA IMPORT/EXPORT-----------
    #----------------------------------------

    #create empty dirs if they dont exist yet
    if os.path.exists(NUMPY_SAVED_FILES_DIR) and os.path.exists(OUTPUT_IMG_DIR):
        pass
    elif os.path.exists(NUMPY_SAVED_FILES_DIR):
        os.mkdir(OUTPUT_IMG_DIR)
    elif os.path.exists(OUTPUT_IMG_DIR):
        os.mkdir(NUMPY_SAVED_FILES_DIR)
    else:
        os.mkdir(NUMPY_SAVED_FILES_DIR)
        os.mkdir(OUTPUT_IMG_DIR)

    #check if the mono data has been exported to numpy binary already, and if so, load it from there to save time
    np_mono_vec_file = os.path.join(NUMPY_SAVED_FILES_DIR, 'mono_vecs.npy')
    np_mono_ints_file = os.path.join(NUMPY_SAVED_FILES_DIR, 'mono_ints.npy')
    if os.path.exists(np_mono_vec_file) and os.path.exists(np_mono_ints_file):
        vecs_mono = np.load(np_mono_vec_file)
        ints_mono = np.load(np_mono_ints_file)
    else:
        vecs_mono, ints_mono = import_h5mono_to_nparr(SOURCE_H5_MONO_FILE)
        vecs_mono, ints_mono = fill_mono_symmetry_equivs(vecs_mono, ints_mono, HKLS_EQUIV, remove_dups = REMOVE_DUPS)
        np.save(np_mono_vec_file, vecs_mono)
        np.save(np_mono_ints_file, ints_mono)

    #same as above, check if np exports already exists for expt files
    np_expt_vec_file = os.path.join(NUMPY_SAVED_FILES_DIR, 'expt_vecs.npy')
    np_expt_ints_file = os.path.join(NUMPY_SAVED_FILES_DIR, 'expt_ints.npy')
    if os.path.exists(np_expt_vec_file) and os.path.exists(np_expt_ints_file):
        vecs_expt = np.load(np_expt_vec_file)
        ints_expt = np.load(np_expt_ints_file)
    else:
        vecs_expt, ints_expt = import_h5expt_to_nparr(SOURCE_H5_EXPT_FILE)
        np.save(np_expt_vec_file, vecs_expt)
        np.save(np_expt_ints_file, ints_expt)
    
    #load OR import and save rotation matrix if it doesnt exist already
    np_rotM_file = os.path.join(NUMPY_SAVED_FILES_DIR, 'rotation_M_solution.npy')
    if os.path.exists(np_rotM_file):
        rotM_arr = np.load(np_rotM_file)
        rotM = R.from_matrix(rotM_arr)
    else:
        rotM = import_xxx_to_rotM(XXX_FILE)
        np.save(np_rotM_file, rotM.as_matrix())
    #rotate the existing vectors by the imported matrix
    vecs_mono_rotated = rotate_sphere(vecs_mono, rotM.as_matrix())

    #----------------------------------------
    #-----------VISULALIZATION---------------
    #----------------------------------------


    output_png_img_file_mono = os.path.join(OUTPUT_IMG_DIR, 'mono_' + DATASET_NAME + '.png')
    output_png_img_file_expt = os.path.join(OUTPUT_IMG_DIR, 'expt_' + DATASET_NAME + '.png')
    output_png_img_file_mono_solved = os.path.join(OUTPUT_IMG_DIR, '_mono_solved_' + DATASET_NAME + '.png')

    #draw all spheres
    draw_sphere(vecs_mono, output_png_img_file_mono)
    draw_sphere(vecs_expt, output_png_img_file_expt)
    draw_sphere(vecs_mono_rotated, output_png_img_file_mono_solved)

if __name__ == '__main__':

    SOURCE_H5_EXPT_FILE = r"C:\VS_CODE\outgoing\dark_01\dark_01__expt.h5"
    SOURCE_H5_MONO_FILE = r"C:\VS_CODE\outgoing\dark_01\dark_01__mono.h5"
    XXX_FILE = r"C:\VS_CODE\outgoing\dark_01\xxx.h5"

    # HKLS_EQUIV = ((-1, -1, -1), 
    #               (-1, 1, -1), 
    #               (1, -1, 1))

    HKLS_EQUIV = ((-1, -1, -1),
                (-1, 1, -1),
                (1, -1, 1),
                (-1, -1, 1),
                (1, -1, -1),
                (-1, 1, 1),
                (1, 1, -1))
    
    DATASET_NAME = 'snp'

    OUTPUT_IMG_DIR = r'graphs_' + DATASET_NAME
    NUMPY_SAVED_FILES_DIR = r'numpy_saves_' + DATASET_NAME

    REMOVE_DUPS = False

    main_export(SOURCE_H5_EXPT_FILE,
                SOURCE_H5_MONO_FILE,
                XXX_FILE,
                HKLS_EQUIV,
                DATASET_NAME,
                OUTPUT_IMG_DIR,
                NUMPY_SAVED_FILES_DIR,
                REMOVE_DUPS)