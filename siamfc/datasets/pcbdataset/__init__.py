from .pcbdataset import PCBDataset
from .pcbdataset_tri import PCBDatasetTri

# Define pcbdataset dictionary
pcbdataset_ = {
    'siamfc': PCBDataset,
    'siamcar': PCBDataset,
    'origin': PCBDataset,
    'official_origin': PCBDataset,
    'tri_origin': PCBDatasetTri,
    'tri_127_origin': PCBDatasetTri,
}


def get_pcbdataset(method):
    return pcbdataset_[method]
