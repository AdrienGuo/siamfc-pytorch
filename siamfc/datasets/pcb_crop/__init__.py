from .crop_tri_127_origin import CropTri127Origin
from .crop_tri_origin import CropTriOrigin
from .pcb_crop_official_origin import PCBCropOfficialOrigin
from .pcb_crop_origin import PCBCropOrigin
from .pcb_crop_siamcar import PCBCropOfficial
from .pcb_crop_siamfc import PCBCropSiamFC

PCBCrop = {
    'siamfc': PCBCropSiamFC,
    'official': PCBCropOfficial,
    'origin': PCBCropOrigin,
    'official_origin': PCBCropOfficialOrigin,
    'tri_origin': CropTriOrigin,
    'tri_127_origin': CropTri127Origin,
}

def get_pcb_crop(method):
    return PCBCrop[method]
