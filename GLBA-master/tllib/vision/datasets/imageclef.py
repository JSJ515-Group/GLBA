"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import Optional
import os
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class ImageCLEF(ImageList):
    """ImageCLEF Dataset.

    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'C'``: Caltech, \
            ``'I'``: ImageNet and ``'P'``: Pascal.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            c/
                0/
                    *.jpg
                    ...
            i/
            p/
            list/
                cList.txt
                iList.txt
                pList.txt
    """
    download_list = [
        ("list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/d9bca681c71249f19da2/?dl=1"),
        ("c", "c.tgz", "https://cloud.tsinghua.edu.cn/f/edc8d1bba1c740dc821c/?dl=1"),
        ("i", "i.tgz", "https://cloud.tsinghua.edu.cn/f/ca6df562b7e64850ad7f/?dl=1"),
        ("p", "p.tgz", "https://cloud.tsinghua.edu.cn/f/82b24ed2e08f4a3c8888/?dl=1"),
    ]
    image_list = {
        "C": "list/cList.txt",
        "I": "list/iList.txt",
        "P": "list/pList.txt"
    }
    CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']

    def __init__(self, root: str, task: str, download: Optional[bool] = True, **kwargs):
        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        super(ImageCLEF, self).__init__(root, ImageCLEF.CLASSES, data_list_file=data_list_file, **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())