from data import *
from segmentation import *
import os

if __name__ == "__main__":
    input_nii_file = r"C:\projects\NeurophetSegBaseEngine\data\mask\irb82_0001_20230721_swi_synthseg.nii.gz"
    input_nii_file = r"C:\projects\SynthSeg-master\output\irb82_0001_20230721_swi_synthseg.nii.gz"
    input_nii_file = r"C:\database\dataset\mask_robust\irb82_0002_20230720_swi_mag_synthseg.nii.gz"
    output_nii_file = r'C:\projects\NeurophetSegBaseEngine\jung\test.nii'

    # merge_mask(input_nii_file, output_nii_file)
    img, data = load_nii(input_nii_file)

    type = 1
    if type == 1:
        data = merge_labels(data, labels=[3, 8, 42, 47], target_label=(1, 0))
        data = add_padding(data, labels= [1], padding_size = 1)

    elif type == 2:
        data = merge_labels(data, labels=[3, 8, 42, 47], target_label=(1, 0))
        data = get_borders(data, labels = [1])
        data = add_padding(data, labels = [1])

    elif type == 3:
        data = get_borders(data, labels = [3, 8, 42, 47], target_label= 1)

    else:
        data = get_borders(data, labels=[24, 3, 8, 42, 47])
        data = add_padding(data, labels=[1])

    save_nii(output_nii_file, data, img.affine)


    import sys
    #sys.exit()

    input_directory = r"C:\database\dataset\mask_robust"
    output_directory = r'C:\projects\NeurophetSegBaseEngine\jung\temp'

    for filename in os.listdir(input_directory):
        if filename.endswith(".nii.gz") or filename.endswith(".nii"):
            # Construct full paths
            input_file_path = os.path.join(input_directory, filename)
            output_file_path = os.path.join(output_directory, filename)

            # Merge and save mask
            img, data = load_nii(input_file_path)
            data = merge_labels(data, labels=[3, 8, 42, 47], target_label=(1, 0))
            data = add_padding(data, labels=[1], padding_size=1)
            save_nii(output_file_path, data, img.affine)
