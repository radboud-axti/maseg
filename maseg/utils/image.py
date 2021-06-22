"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import warnings
from typing import Union

import numpy as np
import pydicom

from maseg.utils.dicom_utils import (
    DICOM_MANUFACTURER,
    DICOM_MODEL_NAME,
    DICOM_PHOTOMETRIC_INTERPRETATION,
    DICOM_WINDOW_CENTER,
    DICOM_WINDOW_CENTER_WIDTH_EXPLANATION,
    DICOM_WINDOW_WIDTH,
    DICOM_PRESENTATION_INTENT_TYPE,
    build_dicom_lut,
)


def clip_and_scale(
    arr: np.ndarray,
    clip_range: Union[bool, tuple, list] = False,
    source_interval: Union[bool, tuple, list] = False,
    target_interval: Union[bool, tuple, list] = False,
):
    """
    Clips image to specified range, and then linearly scales to the specified range (if given).
    In particular, the range in source interval is mapped to the target interval linearly,
    after clipping has been applied.
    - If clip_range is not set, the image is not clipped.
    - If target_interval is not set, only clipping is applied.
    - If source_interval is not set, the minimum and maximum values will be picked.
    Parameters
    ----------
    arr : array_like
    clip_range : tuple
        Range to clip input array to.
    source_interval : tuple
       If given, this denote the original minimal and maximal values.
    target_interval : tuple
        Interval to map input values to.
    Returns
    -------
    ndarray
        Clipped and scaled array.
    """
    arr = np.asarray(arr)
    if clip_range and tuple(clip_range) != (0, 0):
        if not len(clip_range) == 2:
            raise ValueError("Clip range must be two a tuple of length 2.")
        arr = np.clip(arr, clip_range[0], clip_range[1])
    if target_interval and tuple(target_interval) != (0, 0):
        if not len(target_interval) == 2:
            raise ValueError("Scale range must be two a tuple of length 2.")
        if source_interval:
            arr_min, arr_max = source_interval
        else:
            arr_min = arr.min()
            arr_max = arr.max()
        if arr_min == arr_max:
            if not arr_max == 0:
                arr = target_interval[1] * arr / arr_max
        else:
            size = target_interval[1] - target_interval[0]
            arr -= arr_min
            arr = arr / (arr_max - arr_min)
            arr *= size
            arr += target_interval[0]
    return arr


class Image:
    """
    Rudimentary object to allow for storing image properties and ability to write to file.

    Do not trust on this! API can change.
    """

    def __init__(self, data, data_origin=None, header=None, *args, **kwargs):
        self.data = data
        self.data_origin = data_origin
        self.header = header
        self.spacing = None if not header else header.get("spacing", None)

    @property
    def shape(self):
        return self.data.shape


class MammogramImage(Image):
    def __init__(self, data, image_fn, header, voi_lut_function=None, view=None, laterality=None):
        super().__init__(data, image_fn, header)

        self.raw_image = data
        self.header = header

        # If VOILutFunction is not defined for GE, it implies that the voi_lut_function is actually SIGMOID.
        if voi_lut_function == "":
            manufacturer = self.header["dicom_tags"][DICOM_MANUFACTURER] if self.header["dicom_tags"][DICOM_MANUFACTURER] else ''
            if manufacturer.lower() == "ge medical systems":
                voi_lut_function = "SIGMOID"
            else:
                voi_lut_function = "LINEAR"

        self.voi_lut_function = voi_lut_function
        if self.voi_lut_function not in ["LINEAR", "LINEAR_EXACT", "SIGMOID"]:
            raise ValueError(f"VOI LUT Function {self.voi_lut_function} is not supported by the DICOM standard.")

        # Photometric Interpretation determines how to read the pixel values and if they should be inverted.
        self.photometric_interpretation = self.header["dicom_tags"][DICOM_PHOTOMETRIC_INTERPRETATION]

        self.view = view
        self.laterality = laterality
        self.model = self.header["dicom_tags"][DICOM_MODEL_NAME] if self.header["dicom_tags"][DICOM_MODEL_NAME] else ''

        self._image = None

        # Window leveling
        self._output_range = (0.0, 1.0)
        self._current_set_center_width = [None, None]
        self.num_dicom_center_widths = 0
        self.dicom_window_center = []
        self.dicom_window_width = []

        self._parse_window_level()

        # LUTs
        self._uniques = None
        self._current_set_lut = None
        self.dicom_luts = []
        self.num_dicom_luts = 0
        self._parse_luts()

    def _parse_window_level(self):
        window_center = self.header["dicom_tags"][DICOM_WINDOW_CENTER]
        window_width = self.header["dicom_tags"][DICOM_WINDOW_WIDTH]
        explanation = self.header["dicom_tags"][DICOM_WINDOW_CENTER_WIDTH_EXPLANATION]

        if window_center and window_width:
            self.dicom_window_center = [float(_) for _ in window_center.split("\\")]
            self.dicom_window_width = [float(_) for _ in window_width.split("\\")]

            if self.voi_lut_function == "SIGMOID" and (
                len(self.dicom_window_center) == 0 or len(self.dicom_window_width) == 0
            ):
                raise ValueError(
                    f"{self.data_origin}: " f"SIGMOID as VOILutFunction requires a window center and width value."
                )

            if not len(self.dicom_window_width) == len(self.dicom_window_center):
                raise ValueError(f"Number of widths and center mismatch.")
            self.num_dicom_center_widths = len(self.dicom_window_width)

            if self.num_dicom_center_widths >= 1:
                self._current_set_center_width = [self.dicom_window_center[0], self.dicom_window_width[0]]
            if self.header["dicom_tags"][DICOM_PRESENTATION_INTENT_TYPE] == 'FOR PROCESSING':
                self._current_set_center_width = [None, None]
                self.num_dicom_center_widths = 0

        if explanation:
            self.dicom_center_width_explanation = [_.strip() for _ in explanation.split("\\")]

    def _parse_luts(self):
        # SimpleITK does not yet support sequence tags, therefore read with pydicom.
        dcm = pydicom.read_file(str(self.data_origin), stop_before_pixels=True, force=True)
        if not self._uniques:
            self._uniques = np.unique(self.raw_image)
        voi_lut_sequence = getattr(dcm, "VOILUTSequence", [])

        for voi_lut in voi_lut_sequence:
            self.num_dicom_luts += 1
            lut_descriptor = list(voi_lut.LUTDescriptor)
            lut_explanation = getattr(voi_lut, "LUTExplanation", "")  # Sometimes missing
            lut_data = list(voi_lut.LUTData)
            len_lut = lut_descriptor[0] if not lut_descriptor[0] == 0 else 2 ** 16
            first_value = lut_descriptor[1]  # TODO: This assumes that mammograms are always unsigned integers.
            # number_of_bits_lut_data = lut_descriptor[2]

            self.dicom_luts.append((lut_explanation, lut_data, len_lut, first_value))

    def set_lut(self, idx):
        if idx is not None and (idx < 0 or idx >= len(self.dicom_luts)):
            raise ValueError(f"Incorrect LUT index. Got {idx}.")
        self._current_set_lut = idx

    def set_center_width(self, window_center, window_width):
        if window_width <= 0:
            raise ValueError(f"window width should be larger than 0. Got {window_width}.")
        if not window_center or not window_width:
            raise ValueError(f"center and width should both be set.")

        self._current_set_center_width = [window_center, window_width]

    def _apply_sigmoid(self, image, window_center, window_width):
        # https://dicom.innolitics.com/ciods/nm-image/voi-lut/00281056
        image_min = image.min()
        image_max = image.max()
        output = (image_max - image_min) / (1 + np.exp(-4 * (image - window_center) / window_width)) + image_min
        return output.astype(image.dtype)

    def _apply_linear_exact(self, image, window_center, window_width):
        output = np.zeros(image.shape, dtype=np.float)

        lower_mask = image <= window_center - window_width / 2
        upper_mask = image > window_center + window_width / 2

        output[lower_mask] = self._output_range[0]
        output[upper_mask] = self._output_range[1]

        output[~lower_mask & ~upper_mask] = (image[~lower_mask & ~upper_mask] - window_center) / window_width + 0.5

        output = clip_and_scale(output, None, None, self._output_range)
        return output

    def _apply_linear(self, image, window_center, window_width):
        output = np.zeros(image.shape, dtype=float)

        lower_mask = image <= window_center - 0.5 - (window_width - 1) / 2
        upper_mask = image > window_center - 0.5 + (window_width - 1) / 2

        output[lower_mask] = self._output_range[0]
        output[upper_mask] = self._output_range[1]

        output[~lower_mask & ~upper_mask] = (image[~lower_mask & ~upper_mask] - (window_center - 0.5)) / (
            window_width - 1
        ) + 0.5

        output = clip_and_scale(output, None, None, self._output_range)
        return output

    @property
    def image(self):
        # MONOCHROME1 handling
        if self.photometric_interpretation == "MONOCHROME1":
            image_max = self.raw_image.max()
            self._image = image_max - self.raw_image
            if self.header["dicom_tags"][DICOM_MANUFACTURER] in ["Agfa-Gevaert", "LORAD", "GE MEDICAL SYSTEMS", "Planmed", "FUJIFILM Corporation", "IMS GIOTTO S.p.A.", "SIEMENS", "HOLOGIC, Inc."]:
                if (
                    self.num_dicom_center_widths == 0
                    and self.header["dicom_tags"][DICOM_MANUFACTURER] == "Agfa-Gevaert"
                ):
                    self.dicom_window_center = [image_max / 2]
                    self.dicom_window_width = [image_max]

            else:
                raise NotImplementedError(f"{self.data_origin}: {self.header['dicom_tags'][DICOM_MANUFACTURER]}")

        if self._current_set_lut is not None and any([_ is not None for _ in self._current_set_center_width]):
            warnings.warn(
                f"Both LUT and center width are set, only LUT will be applied. "
                f"Got {self._current_set_lut} and {self._current_set_center_width} for {self.data_origin}."
            )

        if self._current_set_lut is not None:
            _, lut_data, len_lut, first_value = self.dicom_luts[self._current_set_lut]
            LUT = build_dicom_lut(self._uniques, lut_data, len_lut, first_value)
            self._image = clip_and_scale(LUT[self.raw_image], None, None, self._output_range)

        elif all(self._current_set_center_width):
            if self.voi_lut_function == "LINEAR":
                self._image = self._apply_linear(self.raw_image, *self._current_set_center_width)
            elif self.voi_lut_function == "LINEAR_EXACT":
                self._image = self._apply_linear_exact(self.raw_image, *self._current_set_center_width)
            elif self.voi_lut_function == "SIGMOID":
                self._image = clip_and_scale(
                    self._apply_sigmoid(self.raw_image, *self._current_set_center_width),
                    None,
                    None,
                    self._output_range,
                )
            else:
                raise ValueError(f"VOI LUT Function {self.voi_lut_function} is not supported by the DICOM standard.")
        else:
            return self.raw_image

        return self._image

    def to_filename(self, *args, **kwargs):
        raise NotImplementedError(f"API unstable. Saving the raw image will create difficulties when parsing LUTs.")
