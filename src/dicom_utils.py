import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import imread
import os
import png
from shutil import copy

DICOM_DATA_DIR = '../data/dicom'
DICOM_IMAGE_DIR = '../data/image'
TIF_DIR = '../data/horos-export'
GREEN_ARROWS_DIR = '../data/green-arrows/'

def data_iterator(func, data_dir=DICOM_DATA_DIR):
    """Iterator that walks a directory tree calling func on each file path"""
    for root, dirs, files in os.walk(data_dir):
        print("processing %s files" % str(len(files)))
        for fname in files:
            if fname.endswith('.md') or fname == '.DS_Store': # ugly but works
                continue
            file_path = os.path.join(root, fname)
            dataset = func(file_path)
            if dataset is None:
                continue
            yield (file_path, dataset)

def get_min_shape():
    """Get the minimum dimension of images in our dataset. Maybe useful for
    downsizing later."""
    min_width, min_height = None, None
    for _, dataset in data_iterator(get_dicom_dataset):
        try:
            pixels = dataset.pixel_array
            if min_height is None or pixels.shape[0] < min_height:
                min_height = pixels.shape[0]
            if min_width is None or pixels.shape[1] < min_width:
                min_width = pixels.shape[1]
        except Exception as e:
            continue
    return (min_height, min_width)

def generate_data():
    num_success, num_failures = 0, 0
    image_index = 1
    for path, dataset in data_iterator(get_dicom_dataset):
        try:
            pixels = dataset.pixel_array
        except Exception as e:
            print('exception: %s' % str(e))
            num_failures += 1
            if num_failures % 100 == 0:
                print("%s failures" % str(num_failures))
            continue
        pixels_scaled = np.uint((np.maximum(pixels, 0) / pixels.max()) * 255)
        w = png.Writer(pixels.shape[1], pixels.shape[0], greyscale=True)
        with open(os.path.join(DICOM_IMAGE_DIR, str(image_index) + '.png'), 'wb') as f:
            if len(pixels_scaled.shape) == 2:
                w.write(f, pixels_scaled)
                image_index += 1
                num_success += 1
    print('num success: %s' % str(num_success))
    print('num failures: %s' % str(num_failures))

def generate_hand_labeled_dataset():
    i = 0
    total = 0
    for path, image in data_iterator(imread, data_dir=TIF_DIR):
        if len(image.shape) > 2:
            copy(path, GREEN_ARROWS_DIR)
            i += 1
        total += 1
    print("found %s labeled images!" % str(i))
    print("found %s total images!" % str(total))

if __name__ == '__main__':
    generate_hand_labeled_dataset()
