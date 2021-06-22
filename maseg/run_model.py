import os
import torch
import cv2
import warnings
import numpy as np
from maseg.unet import Unet
from maseg.utils.readers import read_mammogram


class RunModel:
    """
        The RunModel object can be used to apply the segmentation model with specific weights to different mammograms
        to retrieve a segmentation or its performance when a ground truth mask is given.
    """
    def __init__(self, model_weights_path):
        """
            Initialize a RunModel object with specific model weights. The RunModel object can then be used to apply the
            model to different mammograms to retrieve a segmentation.
            model_weights_path: path to a .ckpt file with desired model weights
        """
        self.model_weights_path = model_weights_path
        self.model = Unet.load_from_checkpoint(
            model_weights_path,
            num_classes=3,
            input_channels=1,
            depth=5,
            features_start=64,
            bilinear=True,
            strict=False,
        )
        self.model_input_spacing = [0.4, 0.4, 1.0]

    def get_segmentation_probabilities(self, input_image, output_size=None):
        """
            Gives a segmentation probabilities back for an input image
            input_image: can be a path to dicom mammogram or a numpy array (ideally with pixel spacing 0.4 mm)
            output_size: if specified, the model will return the segmentation in output size, if None, the output will
            match the original size of the mammogram. Default is None
        """
        image, original_size = self._check_and_get_correct_image_input(input_image)
        image = image.to(self.model.device)  # put image to same device as model
        # run_model
        output = self.model.eval().forward(image.unsqueeze(0).unsqueeze(0), pad_tensor=True)
        # upsample to original size
        if output_size:
            output = torch.nn.functional.interpolate(output, size=output_size, mode='bilinear', align_corners=False)
        else:
            output = torch.nn.functional.interpolate(output, size=original_size, mode='bilinear', align_corners=False)
        return output

    def get_segmentation(self, input_image, fill_holes_in_breast=False, output_size=None):
        """
            Gives a segmentation back for an input image
            input_image: can be a tensor or a path to a path to dicom image
            fill_holes_in_breast: if True, this function will only keep the largest segmented breast area and fill any
            existing holes
            output_size: if specified, the model will return the segmentation in output size, if None, the output will
            match the original size of the mammogram. Default is None
            :return:
        """
        output = self.get_segmentation_probabilities(input_image, output_size)
        # get segmentation
        _, segmentation = output.max(dim=1)
        segmentation = segmentation.squeeze().to('cpu').numpy()
        if fill_holes_in_breast:
            segmentation = self._fill_holes_in_breast(segmentation)
        return segmentation

    def _check_and_get_correct_image_input(self, image):
        if isinstance(image, str):
            if os.path.isfile(image):
                mammogram = read_mammogram(image, spacing=self.model_input_spacing)
                image = mammogram.image
                if image.max() > 1:
                    image = image - image.min()
                    image = image / image.max()
                image = torch.from_numpy(image.copy()).type(torch.FloatTensor)
                original_size = mammogram.header['orig_shape'][-2:]
            else:
                raise ValueError(f"Input image '{image}' is a string, but not an existing file")
        elif isinstance(image, np.ndarray):
            warnings.warn(f"Input image is an numpy array with unknown pixel spacing. "
                          f"If pixel spacing is not 0.4 mm, performance can be lower")
            image = image.squeeze()
            original_size = list(image.shape)
            if np.max(image) > 1:
                image = image / np.max(image)
            image = torch.from_numpy(image.copy()).type(torch.FloatTensor)
        else:
            raise ValueError(f"Input image is not a path to a file or a numpy array")
        return image, original_size

    @staticmethod
    def _fill_holes_in_breast(segmentation):
        try:
            _, contours, _ = cv2.findContours((segmentation == 1).astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except:
            contours, _ = cv2.findContours((segmentation == 1).astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = []
        for contour in contours:
            areas.append(cv2.contourArea(contour))
        segmentation_filled = cv2.drawContours(segmentation.astype("uint8"), contours, np.argmax(areas), 1, -1)
        return segmentation_filled

    @staticmethod
    def get_image(path, spacing=None):
        mammogram = read_mammogram(path, spacing)
        image = mammogram.image
        return image
