import numpy as np
import nibabel as nib
from nibabel.nifti1 import Nifti1Image

class nifty():
    def __init__(self, file_path):
        self.image = nib.load(file_path)
        self.np_image = self.image.get_fdata()