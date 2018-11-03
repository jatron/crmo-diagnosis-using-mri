Put directories containing dicom files here. The dicom_utils script will look in
here for files to convert/get pixel data from.

In order to run the dicom_utils script you might need to install the GDCM
library. (pydicom depends on this for extracting the pixel_data for *some* (but
not all) of the files.)

Instructions on how to do so are here: http://gdcm.sourceforge.net/wiki/index.php/Compilation

The documentation on that site ain't great so here's a step by step of what I
did on OS X:

1. Download the unix source: https://launchpad.net/ubuntu/+source/gdcm
2. Make sure you have cmake, make and g++ installed (if you have homebrew you
   can do `brew install cmake`)
3. cd to the gdcm source directory
4. `mkdir gdcm-build`
5. `cmake ../gdcm`
6. `make` (this took a few minutes to complete for me).
