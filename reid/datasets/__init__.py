from __future__ import absolute_import

from .market1501 import Market1501
from .msmt17_copy import MSMT17
from .cuhk03_np import CUHK03
from .cuhk02 import CUHK02
from .dukemtmc import DukeMTMC
from .cuhk01 import CUHK01
from .threedpes import ThreeDPES
from .ilids import ILIDS
from .prid import PRID
from .viper import VIPeR
from PIL import Image
from torch.utils.data import Dataset
泛化到未知域设置
__factory = {
    'market1501': Market1501, 'msmt17': MSMT17,
    'cuhk03-np': CUHK03, 'cuhk02': CUHK02, 
}

def names():
    set_names = sorted(__factory.keys())
    return set_names


def create(name, root, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)


class BaseDataset(Dataset):
    def __init__(self, dataset, trans):
        self.dataset = dataset
        self.trans = trans

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        fname, label, cam = self.dataset[index]
        image = Image.open(fname).convert('RGB')
        return self.trans(image), fname, label, index, cam
