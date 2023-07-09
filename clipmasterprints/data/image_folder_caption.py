import torchvision
from collections import OrderedDict

class ImageFolderCaption(torchvision.datasets.ImageFolder):

    def __init__(self, root, mapping_path, transform):
        super().__init__(root, transform)
        mapping_lst = open(mapping_path, 'r').read().split('\n')
        self.folder_to_class = dict([(string_pair[:9], string_pair[9:].strip()) for string_pair in mapping_lst if string_pair])
        self.idx_to_class = OrderedDict([(value, self.folder_to_class[key]) for key, value in self.class_to_idx.items()])

    def __getitem__(self, idx):
        (image,idx) = super().__getitem__(idx)
        return (image,self.idx_to_class[idx])