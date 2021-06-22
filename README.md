# Maseg: segmentation of mammograms into breast, background, and pectoral muscle
Maseg is a U-Net segmentation method that segments mammograms into breast, background, and pectoral muscle. This model 
is designed to be suited for both raw and processed digital mammograms (DMs) and to generalize across models of 
different vendors, and across both CC and MLO views. It was developed with eight different datasets totalling 7,394
mammograms of 1,077 women.

Feel free to use our code and trained weights, but please cite our open access paper if you do so DOI: 10.1117/1.JMI.11.1.014001

## Dependencies
* pytorch lightning
* numpy
* pydicom

## Applying the model
The class RunModel can be used to apply the model to a specific mammograms. There are two ways in which you
can give a mammogram as input to the model:
1. The directory path of a dicom mammogram to the model. This will ensure the mammogram is read correctly and the 
pixel spacing is suitable for the model.
1. A numpy array of the mammogram. This gives more freedom of data types of mammograms, but does not ensure the correct
pixel spacing. A pixel spacing different form 0.4 mm can result in a lower segmentation performance.

A RunModel object has the following functions
* get_segmentation: will return a 2D numpy array with 0 (background), 1 (breast), and 2 (pectoral muscle) for each 
pixel.
* get_segmentation_probabilities: will return a 3D numpy with the probabilities for each class in dimension 2.
* get_image: will return the original image when given a path to a dicom mammogram.

