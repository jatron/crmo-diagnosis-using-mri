import numpy as np
import os
import png
import pydicom


DICOM_DATA_DIR = '../data/dicom'
DICOM_IMAGE_DIR = '../data/image'

def get_dicom_dataset(fname):
    """Returns a pydiom.FileDataset object corresponding to a single dicom file"""
    try:
        return pydicom.read_file(fname, force=True)
    except Exception as e:
        print(e)
        return None

def data_iterator(data_dir=DICOM_DATA_DIR):
    """Iterator that walks a directory tree loading dicom files"""
    for root, dirs, files in os.walk(data_dir):
        print("processing %s files" % str(len(files)))
        for fname in files:
            if fname.endswith('.md') or fname == '.DS_Store': # ugly but works
                continue
            file_path = os.path.join(root, fname)
            dataset = get_dicom_dataset(file_path)
            if dataset is None:
                continue
            yield (file_path, dataset)

def get_min_shape():
    """Get the minimum dimension of images in our dataset. Maybe useful for
    downsizing later."""
    min_width, min_height = None, None
    for _, dataset in data_iterator():
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
    for path, dataset in data_iterator():
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

if __name__ == '__main__':
    generate_data()
