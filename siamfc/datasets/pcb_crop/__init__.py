from .pcb_crop_siamcar import PCBCropOfficial
from .pcb_crop_official_origin import PCBCropOfficialOrigin
from .pcb_crop_origin import PCBCropOrigin

PCBCrop = {
    'official': PCBCropOfficial,
    'origin': PCBCropOrigin,
    'official_origin': PCBCropOfficialOrigin,
}

def get_pcb_crop(method):
    return PCBCrop[method]
