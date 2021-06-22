import matplotlib.pyplot as plt
from maseg.run_model import RunModel


if __name__ == '__main__':
    model_weights_path = 'segmentation_weights.ckpt'
    model = RunModel(model_weights_path)

    input_image = 'mammogram_dicom_file.dcm'
    segmentation = model.get_segmentation(input_image, fill_holes_in_breast=False, output_size=None)

    plt.imshow(segmentation)
    plt.show()

    print('done')
