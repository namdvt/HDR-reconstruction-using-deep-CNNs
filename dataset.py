from torch.utils.data import Dataset, DataLoader, random_split
import cv2


class HDRDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        self.folder = root
        self.indexes = open(root + '/annotations.txt').read().splitlines()

    def __getitem__(self, index):
        ldr_image, hdr_image = self.indexes[index].split('\t')
        ldr_image = cv2.imread(ldr_image)
        ldr_image = cv2.cvtColor(ldr_image, cv2.COLOR_BGR2RGB)
        ldr_image = ldr_image / 255

        hdr_image = cv2.imread(hdr_image, cv2.IMREAD_ANYDEPTH)
        hdr_image = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2RGB)

        return ldr_image.transpose(2, 0, 1), hdr_image.transpose(2, 0, 1)

    def __len__(self):
        return len(self.indexes)


def get_loader(root, batch_size, shuffle=True):
    dataset = HDRDataset(root=root)

    num_train = int(len(dataset) * 0.8)
    num_val = len(dataset) - num_train
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              drop_last=True)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=True)

    return train_loader, val_loader
