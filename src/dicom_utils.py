import os
import pydicom

DICOM_DATA_DIR = '../data/dicom'

def get_pixels(fname):
    """Returns a 2d numpy containing the pixel values of a dicom image"""
    dicom_dataset = pydicom.read_file(fname)
    return dicom_dataset.pixel_array

def pixels_iterator(data_dir=DICOM_DATA_DIR):
    for root, dirs, files in os.walk(data_dir):
        for fname in files:
            if fname.endswith('.md') or fname == '.DS_Store': # ugly but works
                continue
            yield get_pixels(os.path.join(root, fname))

def generate_jpgs(data_dir=DICOM_DATA_DIR):
    for root, dirs, files in os.walk(data_dir):
        for fname in files:
            yield
