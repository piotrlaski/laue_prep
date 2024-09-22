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
def fill_mono_symmetry_equivs(vectors_mono: np.ndarray, intensities_mono: np.ndarray, hkl_equivs: tuple[tuple[int]]) -> tuple[np.ndarray]:

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
    vectors_mono_full_no_dups = []
    intensites_mono_full_no_dups = []

    for i, vector in enumerate(vectors_mono_full):
        if any(all(vector == vector_no_dups) for vector_no_dups in vectors_mono_full_no_dups):
            continue
        else:
            vectors_mono_full_no_dups.append(vector)
            intensites_mono_full_no_dups.append(intensites_mono_full[i])


    return(np.array(vectors_mono_full_no_dups), np.array(intensites_mono_full_no_dups))
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


if __name__ == '__main__':
    
    
    # SOURCE_H5_EXPT_FILE = r"C:\Users\piotr\Documents\VS_Code\working_dirs\outgoing\rh11_great_struct\dark_01__expt.h5"
    # SOURCE_H5_MONO_FILE = r"C:\Users\piotr\Documents\VS_Code\working_dirs\outgoing\rh11_great_struct\Rh1_mono.h5"
    # XXX_FILE = r"C:\Users\piotr\Documents\VS_Code\working_dirs\outgoing\rh11_great_struct\xxx.h5"

    # HKLS_EQUIV = ((-1, -1, -1), 
    #               (-1, 1, -1), 
    #               (1, -1, 1))
    
    SOURCE_H5_EXPT_FILE = r"C:\Users\piotr\Documents\VS_Code\working_dirs\outgoing\dark_01\dark_01__expt.h5"
    SOURCE_H5_MONO_FILE = r"C:\Users\piotr\Documents\VS_Code\working_dirs\outgoing\SNP_220K_mono.h5"
    XXX_FILE = r"C:\Users\piotr\Documents\VS_Code\working_dirs\outgoing\dark_01\xxx.h5"

    HKLS_EQUIV = ((-1, -1, -1),
                (-1, 1, -1),
                (1, -1, 1),
                (-1, -1, 1),
                (1, -1, -1),
                (-1, 1, 1),
                (1, 1, -1))
    
    name = 'snp'

    OUTPUT_PNG_IMG_FILE_MONO = r'C:\Users\piotr\Documents\VS_Code\laue_prep\mono_' + name + '.png'
    OUTPUT_PNG_IMG_FILE_EXPT = r'C:\Users\piotr\Documents\VS_Code\laue_prep\expt_' + name + '.png'
    OUTPUT_PNG_IMG_FILE_MONO_ROT = r'C:\Users\piotr\Documents\VS_Code\laue_prep\mono_rotated_' + name + '.png'
    NUMPY_SAVED_FILES_DIR = r'C:\Users\piotr\Documents\VS_Code\laue_prep\numpy_saves_' + name

    if os.path.exists(NUMPY_SAVED_FILES_DIR):
        pass
    else:
        os.mkdir(NUMPY_SAVED_FILES_DIR)

    #check if the data has been exported to numpy binary already, and if so, load it from there to save time
    np_mono_vec_file = os.path.join(NUMPY_SAVED_FILES_DIR, 'mono_vecs.npy')
    np_mono_ints_file = os.path.join(NUMPY_SAVED_FILES_DIR, 'mono_ints.npy')
    if os.path.exists(np_mono_vec_file) and os.path.exists(np_mono_ints_file):
        vecs_mono = np.load(np_mono_vec_file)
        ints_mono = np.load(np_mono_ints_file)
    else:
        vecs_mono, ints_mono = import_h5mono_to_nparr(SOURCE_H5_MONO_FILE)
        vecs_mono, ints_mono = fill_mono_symmetry_equivs(vecs_mono, ints_mono, HKLS_EQUIV)
        np.save(np_mono_vec_file, vecs_mono)
        np.save(np_mono_ints_file, ints_mono)

    #same for expt data
    np_expt_vec_file = os.path.join(NUMPY_SAVED_FILES_DIR, 'expt_vecs.npy')
    np_expt_ints_file = os.path.join(NUMPY_SAVED_FILES_DIR, 'expt_ints.npy')
    if os.path.exists(np_expt_vec_file) and os.path.exists(np_expt_ints_file):
        vecs_expt = np.load(np_expt_vec_file)
        ints_expt = np.load(np_expt_ints_file)
    else:
        vecs_expt, ints_expt = import_h5expt_to_nparr(SOURCE_H5_EXPT_FILE)
        np.save(np_expt_vec_file, vecs_expt)
        np.save(np_expt_ints_file, ints_expt)
    

    #import rotation matrix
    rotM = import_xxx_to_rotM(XXX_FILE)
    #rotate the existing vectors by the imported matrix
    vecs_mono_rotated = rotate_sphere(vecs_mono, rotM.as_matrix())

    #draw all spheres
    draw_sphere(vecs_mono, OUTPUT_PNG_IMG_FILE_MONO)
    draw_sphere(vecs_expt, OUTPUT_PNG_IMG_FILE_EXPT)
    draw_sphere(vecs_mono_rotated, OUTPUT_PNG_IMG_FILE_MONO_ROT)

    #====================================
    #TESTING
    # vecs_expt,vecs_mono = import_xxx_vec_data(XXX_FILE)

    # TEST_PATH_1 = r'C:\Users\piotr\Documents\VS_Code\laue_prep\xxx_expt.png'
    # TEST_PATH_2 = r'C:\Users\piotr\Documents\VS_Code\laue_prep\xxx_mono.png'
    # TEST_PATH_3 = r'C:\Users\piotr\Documents\VS_Code\laue_prep\xxx_expt_rotated.png'

    # draw_sphere(vecs_mono, TEST_PATH_1)
    # draw_sphere(vecs_expt, TEST_PATH_2)
    # vecs_expt_rotated = rotate_sphere(vecs_mono, rotM.as_matrix())
    # draw_sphere(vecs_expt_rotated, TEST_PATH_3)

    pass