import numpy as np

DICOM_MODALITY_TAG = "0008|0060"
DICOM_VOI_LUT_FUNCTION = "0028|1056"
DICOM_VOI_LUT_SEQUENCE = "0028|3010"
DICOM_WINDOW_CENTER = "0028|1050"
DICOM_WINDOW_WIDTH = "0028|1051"
DICOM_WINDOW_CENTER_WIDTH_EXPLANATION = "0028|1055"
DICOM_FIELD_OF_VIEW_HORIZONTAL_FLIP = "0018|7034"
DICOM_PATIENT_ORIENTATION = "0020|0020"
DICOM_LATERALITY = "0020|0060"
DICOM_IMAGE_LATERALITY = "0020|0062"
DICOM_VIEW_POSITION = "0018|5101"
DICOM_PHOTOMETRIC_INTERPRETATION = "0028|0004"
DICOM_MANUFACTURER = "0008|0070"
DICOM_MODEL_NAME = "0008|1090"
DICOM_PRESENTATION_INTENT_TYPE = "0008|0068"


def build_dicom_lut(uniques, dicom_lut, len_lut, first_input_value):
    """Builds a lookup table from the dicom VOILUTSequence data, described in:
    http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.11.2.html

    VOI LUT Sequence    (0028,3010)     1C          Defines a Sequence of VOI LUTs.
                                                    One or more Items shall be included in this Sequence.
                                                    Required if Window Center (0028,1050) is not present.
                                                    May be present otherwise.

    >LUT Descriptor     (0028,3002)     1           Specifies the format of the LUT Data in this Sequence.
    >LUT Data           (0028,3006)     1           LUT Data in this Sequence.

    The LUT data can be used as dicom_lut, whereas the LUT Descriptor contains
    (len_lut, first_input_value, number_of_bits)


    Parameters
    ----------
    uniques : np.ndarray
        Unique values in data array sorted ascending.
    dicom_lut : list
    len_lut : int
    first_input_value : int

    Returns
    -------
    np.ndarray
    """
    LUT = np.ones(uniques[-1] + 1) * dicom_lut[-1]

    idx = 0
    for unique_value in uniques:
        if unique_value <= first_input_value:
            LUT[unique_value] = first_input_value
        elif unique_value >= len_lut:
            break
        else:
            LUT[unique_value] = dicom_lut[idx]
        idx += 1

    return LUT
