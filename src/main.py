import numpy

from data import *
from segmentation import *
from augmentation import *
from visualization import *
import experimental
import os


def resize_and_merge(input_nii_file, output_nii_file, target_nii_file):
    img = load_nii(input_nii_file)
    target_img = load_nii(target_nii_file)
    target_img_info = extract_nifti_info(target_img)

    new_img = resize(img, target_img_info['shape'])

    data = nifti_to_numpy(new_img)
    data = change_labels(data, labels = [3, 8, 42, 47], target_label=(1, 0))
    data = add_padding(data, labels=[1], padding_size=2)

    save_nii(output_nii_file, data, new_img.affine)

def new_mask(input_nii_file, output_nii_file, type = 1):
    img = load_nii(input_nii_file)
    data = nifti_to_numpy(img)

    type = type
    specific_labels = [3, 8, 42, 47]
    if type == 1:
        data = change_labels(data, labels=specific_labels, target_label=(1, 0))
        data = add_padding(data, labels=[1], padding_size=2)

    elif type == 2:
        data = change_labels(data, labels=specific_labels, target_label=(1, 0))
        data = get_borders(data, labels=[1])
        data = add_padding(data, labels=[1])

    elif type == 3:
        data = get_borders(data, labels=specific_labels, target_label=1)

    else:
        data = get_borders(data, labels=specific_labels)
        data = add_padding(data, labels=[1])

    save_nii(output_nii_file, data, img.affine)

def reorient_datas(input_file_path, output_file_path):
    img = load_nii(input_file_path)
    info = extract_nifti_info(img)

    img3 = reorient(img)
    info3 = extract_nifti_info(img3)

    print(filename)
    print("Orientation of input: ", info['orientation'])
    print("Change of Orientation: ", info3['orientation'])
    print()

    nib.save(img3, output_file_path)

######################
input_nii_file = r"C:\projects\NeurophetSegBaseEngine\data\mask\irb82_0001_20230721_swi_synthseg.nii.gz"
input_nii_file = r"C:\projects\SynthSeg-master\output\irb82_0001_20230721_swi_synthseg.nii.gz"
input_nii_file = r"C:\database\dataset\seg10\mask_robust\irb82_0002_20230720_swi_mag_synthseg.nii.gz"
input_nii_file = r"C:\database\dataset\seg10\mask_robust\irb82_0041_20220920_swi_mag_synthseg.nii.gz"

output_nii_file = './test.nii'


input_directory = r"C:\database\dataset\seg10\mask_robust"
input_directory = r"C:\database\dataset\seg10\input"
resampled_input_directory = r"C:\database\dataset\seg10\input_resampled"

output_directory = r'C:\projects\NeurophetSegBaseEngine\jung\temp'



if __name__ == "__main__":

    input_file = r"E:\dataset\seg10\ss_input_reorient\irb82_0048.nii.gz"
    mask_file = r"E:\dataset\seg10\ss_mask_new\irb82_0048.nii.gz"

    img = load_nii(input_file)
    mask_img = load_nii(mask_file)
    mask_data = nifti_to_numpy(mask_img)
    # clustered_data = cluster_labels(mask_data, num_clusters= 4)
    clustered_data, num_objects = get_connected_components(mask_data)
    bounding_box = create_bounding_box(clustered_data)
    mask_with_bounding_box = experimental.draw_bounding_boxes(clustered_data,bounding_box)
    save_nii('test.nii', mask_with_bounding_box)


    import sys
    sys.exit()


    resampled_img = load_nii(resampled_file)

    img_info = extract_nifti_info(img)
    resampled_img_info = extract_nifti_info(resampled_img)

    new_img = resize(resampled_img, img_info['shape'])
    new_img_info = extract_nifti_info(new_img)
    print("Original shape : ", resampled_img_info['shape'])
    print("Target shape : ", img_info['shape'])
    print("Changed shape : ", new_img_info['shape'])



    input_directory = r'C:\database\dataset\seg10\ss_mask'
    save_directory = r'C:\database\dataset\seg10\ss_mask_new'

    for filename in os.listdir(input_directory):
        if filename.endswith(".nii.gz") or filename.endswith(".nii"):
            # Construct full paths
            input_file_path = os.path.join(input_directory, filename)
            img = load_nii(input_file_path)
            data = nifti_to_numpy(img)
            data = change_labels(data, [3, 8, 42, 47], (1,0))

            output_file_path = os.path.join(save_directory, filename)
            save_nii(output_file_path, data, img.affine)





