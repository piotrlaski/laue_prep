from orientation_load import *

SOURCE_DIR = r"C:\VS_CODE\outgoing\orientation_run_cudppe"

HKLS_EQUIV = ((-1, -1, -1), 
                (-1, 1, -1), 
                (1, -1, 1))


for dataset in os.listdir(SOURCE_DIR):
    DATASET_NAME = dataset
    dataset_fullpath = os.path.join(SOURCE_DIR, dataset)

    for file in os.listdir(dataset_fullpath):
        if file[-7:] == 'expt.h5':
            SOURCE_H5_EXPT_FILE = os.path.join(dataset_fullpath, file)
        elif file[-7:] == 'mono.h5':
            SOURCE_H5_MONO_FILE = os.path.join(dataset_fullpath, file)
        elif file[-6:] == 'xxx.h5':
            XXX_FILE = os.path.join(dataset_fullpath, file)

    OUTPUT_IMG_DIR = os.path.join(dataset_fullpath, 'graphs')
    NUMPY_SAVED_FILES_DIR = os.path.join(dataset_fullpath, 'numpy_saves')

    REMOVE_DUPS = False

    main_export(SOURCE_H5_EXPT_FILE,
                SOURCE_H5_MONO_FILE,
                XXX_FILE,
                HKLS_EQUIV,
                DATASET_NAME,
                OUTPUT_IMG_DIR,
                NUMPY_SAVED_FILES_DIR,
                REMOVE_DUPS)