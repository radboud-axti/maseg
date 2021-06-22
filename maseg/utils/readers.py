import logging
import pathlib
import pydicom

import numpy as np
import SimpleITK as sitk
from maseg.utils.image import MammogramImage
from maseg.utils.dicom_utils import (
    DICOM_MODALITY_TAG,
    DICOM_VOI_LUT_FUNCTION,
    DICOM_VOI_LUT_SEQUENCE,
    DICOM_WINDOW_CENTER,
    DICOM_WINDOW_WIDTH,
    DICOM_WINDOW_CENTER_WIDTH_EXPLANATION,
    DICOM_FIELD_OF_VIEW_HORIZONTAL_FLIP,
    DICOM_PATIENT_ORIENTATION,
    DICOM_LATERALITY,
    DICOM_IMAGE_LATERALITY,
    DICOM_VIEW_POSITION,
    DICOM_PHOTOMETRIC_INTERPRETATION,
    DICOM_MANUFACTURER,
    DICOM_MODEL_NAME,
    DICOM_PRESENTATION_INTENT_TYPE,
)

logger = logging.getLogger(__name__)


_SITK_INTERPOLATOR_DICT = {
    "nearest": sitk.sitkNearestNeighbor,
    "linear": sitk.sitkLinear,
    "gaussian": sitk.sitkGaussian,
    "label_gaussian": sitk.sitkLabelGaussian,
    "bspline": sitk.sitkBSpline,
    "hamming_sinc": sitk.sitkHammingWindowedSinc,
    "cosine_windowed_sinc": sitk.sitkCosineWindowedSinc,
    "welch_windowed_sinc": sitk.sitkWelchWindowedSinc,
    "lanczos_windowed_sinc": sitk.sitkLanczosWindowedSinc,
}


def read_image_as_sitk_image(filename):
    """
    Read file as a SimpleITK image trying to parse the error.

    Parameters
    ----------
    filename : pathlib.Path or str

    Returns
    -------
    SimpleITK image.
    """
    try:
        sitk.ProcessObject_SetGlobalWarningDisplay(False)
        sitk_image = sitk.ReadImage(str(filename))  # gives monochrome1 warning
        dcm_image = pydicom.dcmread(filename, stop_before_pixels=True)
        spacing = np.array(dcm_image.ImagerPixelSpacing)
        spacing = tuple(np.append(spacing, 1))
        if spacing != sitk_image.GetSpacing():
            sitk_image.SetSpacing(spacing)
    except RuntimeError as error:
        if "itk::ERROR" in str(error):
            error = str(error).split("itk::ERROR")[-1]

        raise RuntimeError(error)

    return sitk_image


def read_image(filename, dtype=None, no_metadata=False, force_2d=False, spacing=None, **kwargs):
    """Read medical image

    Parameters
    ----------
    filename : Path, str
        Path to image, can be any SimpleITK supported filename
    dtype : dtype
        The requested dtype the output should be cast.
    no_metadata : bool
        Do not output metadata
    force_2d : bool
        If this is set to true, first slice in first axis will be taken, if the size[0] == 1.

    Returns
    -------
    Image as ndarray and dictionary with metadata.
    """
    filename = pathlib.Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"{filename} does not exist.")

    new_spacing = spacing if spacing else False
    if new_spacing and np.all(np.asarray(new_spacing) <= 0):
        new_spacing = False

    metadata = {}
    sitk_image = read_image_as_sitk_image(filename)

    # TODO: A more elaborate check for dicom can be needed, not necessarly all dicom files have .dcm as extension.
    if filename.suffix.lower() == ".dcm" and kwargs.get("dicom_keys", None):
        dicom_data = {}
        metadata_keys = sitk_image.GetMetaDataKeys()
        for v in kwargs["dicom_keys"]:
            dicom_data[v] = None if v not in metadata_keys else sitk_image.GetMetaData(v).strip()
        metadata["dicom_tags"] = dicom_data

    orig_shape = sitk.GetArrayFromImage(sitk_image).shape
    if new_spacing:
        sitk_image, orig_spacing = resample_sitk_image(
            sitk_image, spacing=new_spacing, interpolator=kwargs.get("interpolator", None), fill_value=0
        )
        metadata.update({"orig_spacing": tuple(orig_spacing), "orig_shape": orig_shape})

    image = sitk.GetArrayFromImage(sitk_image)

    metadata.update(
        {
            "filename": filename.resolve(),
            "depth": sitk_image.GetDepth(),
            "spacing": sitk_image.GetSpacing(),
            "origin": sitk_image.GetOrigin(),
            "direction": sitk_image.GetDirection(),
        }
    )

    if force_2d:
        if not image.shape[0] == 1:
            raise ValueError(f"Forcing to 2D while the first dimension is not 1.")
        image = image[0]

    if dtype:
        image = image.astype(dtype)

    if no_metadata:
        return image

    return image, metadata


def resample_sitk_image(sitk_image, spacing=None, interpolator=None, fill_value=0):
    """Resamples an ITK image to a new grid. If no spacing is given,
    the resampling is done isotropically to the smallest value in the current
    spacing. This is usually the in-plane resolution. If not given, the
    interpolation is derived from the input data type. Binary input
    (e.g., masks) are resampled with nearest neighbors, otherwise linear
    interpolation is chosen.

    Parameters
    ----------
    sitk_image : SimpleITK image or str
      Either a SimpleITK image or a path to a SimpleITK readable file.
    spacing : tuple
      Tuple of integers
    interpolator : str
      Either `nearest`, `linear` or None.
    fill_value : int

    Returns
    -------
    SimpleITK image.
    """
    if isinstance(sitk_image, (str, pathlib.Path)):
        sitk_image = read_image_as_sitk_image(sitk_image)
    num_dim = sitk_image.GetDimension()
    if not interpolator:
        interpolator = "linear"
        pixelid = sitk_image.GetPixelIDValue()

        if pixelid not in [1, 2, 4]:
            raise NotImplementedError(
                "Set `interpolator` manually, " "can only infer for 8-bit unsigned or 16, 32-bit signed integers"
            )
        if pixelid == 1:  #  8-bit unsigned int
            interpolator = "nearest"

    orig_pixelid = sitk_image.GetPixelIDValue()
    orig_origin = sitk_image.GetOrigin()
    orig_direction = sitk_image.GetDirection()
    orig_spacing = np.array(sitk_image.GetSpacing())
    orig_size = np.array(sitk_image.GetSize(), dtype=int)

    if not spacing:
        min_spacing = orig_spacing.min()
        new_spacing = [min_spacing] * num_dim
    else:
        # new_spacing = [float(s) if s else orig_spacing[idx] for idx, s in enumerate(spacing)]
        new_spacing = [float(spacing[idx]) if spacing[idx] else s for idx, s in enumerate(orig_spacing)]

    assert interpolator in _SITK_INTERPOLATOR_DICT.keys(), "`interpolator` should be one of {}".format(
        _SITK_INTERPOLATOR_DICT.keys()
    )

    sitk_interpolator = _SITK_INTERPOLATOR_DICT[interpolator]

    new_size = orig_size * (orig_spacing / new_spacing)
    new_size = np.ceil(new_size).astype(int)  # Image dimensions are in integers
    # SimpleITK expects lists
    new_size = [int(s) if spacing[idx] else int(orig_size[idx]) for idx, s in enumerate(new_size)]

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetSize(new_size)
    resample_filter.SetTransform(sitk.Transform())
    resample_filter.SetInterpolator(sitk_interpolator)
    resample_filter.SetOutputSpacing(new_spacing)
    resample_filter.SetOutputOrigin(orig_origin)
    resample_filter.SetOutputDirection(orig_direction)
    resample_filter.SetOutputPixelType(orig_pixelid)
    resample_filter.SetSize(new_size)
    resampled_sitk_image = resample_filter.Execute(sitk_image)

    return resampled_sitk_image, orig_spacing


def read_mammogram(filename, spacing=None, interpolator="linear"):
    """
    Read mammograms in dicom format. Attempts to read correct DICOM LUTs.

    Parameters
    ----------
    filename : pathlib.Path or str

    Returns
    -------
    maseg.image.MammogramImage
    """
    extra_tags = [
        DICOM_MODALITY_TAG,
        DICOM_VOI_LUT_FUNCTION,
        DICOM_VOI_LUT_SEQUENCE,
        DICOM_LATERALITY,
        DICOM_IMAGE_LATERALITY,
        DICOM_VIEW_POSITION,
        DICOM_WINDOW_WIDTH,
        DICOM_WINDOW_CENTER,
        DICOM_WINDOW_CENTER_WIDTH_EXPLANATION,
        DICOM_FIELD_OF_VIEW_HORIZONTAL_FLIP,
        DICOM_PATIENT_ORIENTATION,
        DICOM_PHOTOMETRIC_INTERPRETATION,
        DICOM_MANUFACTURER,
        DICOM_MODEL_NAME,
        DICOM_PRESENTATION_INTENT_TYPE,
    ]

    image, metadata = read_image(filename, dicom_keys=extra_tags, spacing=spacing, interpolator=interpolator)
    if "dicom_tags" in metadata.keys():
        dicom_tags = metadata["dicom_tags"]
    else:
        dicom_tags = {DICOM_LATERALITY: '',
                      DICOM_IMAGE_LATERALITY: '',
                      DICOM_FIELD_OF_VIEW_HORIZONTAL_FLIP: 'NO',
                      DICOM_PATIENT_ORIENTATION: '',
                      DICOM_VIEW_POSITION: '',
                      DICOM_VOI_LUT_FUNCTION: None,
                      DICOM_MANUFACTURER: '',
                      DICOM_MODEL_NAME: '',
                      DICOM_PHOTOMETRIC_INTERPRETATION: '',
                      DICOM_WINDOW_CENTER: str(image.max() // 2),
                      DICOM_WINDOW_WIDTH: str(image.max()),
                      DICOM_WINDOW_CENTER_WIDTH_EXPLANATION: 'NORMAL',
                      DICOM_PRESENTATION_INTENT_TYPE: 'FOR PRESENTATION'}
        metadata["dicom_tags"] = dicom_tags

    # modality = dicom_tags[DICOM_MODALITY_TAG]
    # if not modality == "MG":
    #     raise ValueError(f"{filename} is not a mammogram. Wrong Modality in DICOM header.")
    # if not metadata["depth"] == 1:
    #     raise ValueError(f"First dimension of mammogram should be one.")

    # Remove the depth dimension
    if len(image.shape) > 2:
        image = image.reshape(list(image.shape)[1:])

    # Read laterality
    laterality = dicom_tags[DICOM_LATERALITY] or dicom_tags[DICOM_IMAGE_LATERALITY]
    metadata["laterality"] = laterality

    # Sometimes a horizontal flip is required:
    # https://groups.google.com/forum/#!msg/comp.protocols.dicom/X4ddGYiQOzs/g04EDChOQBwJ
    needs_horizontal_flip = dicom_tags[DICOM_FIELD_OF_VIEW_HORIZONTAL_FLIP] == "YES"
    if laterality:
        # Check patient position
        orientation = dicom_tags[DICOM_PATIENT_ORIENTATION].split("\\")[0]
        if (laterality == "L" and orientation == "P") or (laterality == "R" and orientation == "A"):
            needs_horizontal_flip = True

    if needs_horizontal_flip:
        image = np.ascontiguousarray(np.fliplr(image))

    # TODO: These metadata tags are not necessarily required when reading mammograms, add extra flag to read_image.
    # TODO: This needs to be done upstream in fexp.
    del metadata["depth"]
    del metadata["direction"]
    del metadata["origin"]

    # TODO: Move to MammogramImage
    voi_lut_function = dicom_tags[DICOM_VOI_LUT_FUNCTION] if dicom_tags[DICOM_VOI_LUT_FUNCTION] else ""

    return MammogramImage(
        image,
        filename,
        metadata,
        voi_lut_function=voi_lut_function,
        view=dicom_tags[DICOM_VIEW_POSITION],
        laterality=laterality,
    )


if __name__ == "__main__":
    filename = "mammogram_dicom_file.dcm"
    z = read_mammogram(
        filename,
        spacing=(0.5, 0.5, 1.0),
        interpolator="nearest",
    )
    print("ok")
