import h5py
from torch.utils.data import Dataset, DataLoader


BATCH_SIZE = 256


class PCamDataset(Dataset):

    def __init__(self, path_images, path_labels, transform=None, filter=[32889, 121632, 250861, 260375]):
        self.path_images = path_images
        self.path_labels = path_labels
        self.bad_indices = filter
        self.transform = transform

        with h5py.File(self.path_images, 'r') as f:
            n = f['x'].shape[0]

        self.valid_indices = [i for i in range(n) if i not in self.bad_indices]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):

        idx = self.valid_indices[idx]

        # get the image
        with h5py.File(self.path_images, 'r') as f:
            image = f['x'][idx]

        # get the label
        with h5py.File(self.path_labels, 'r') as f:
            label = f['y'][idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    

def load_data(train_transform, test_transform):

    ### load data

    # load the training data
    train_data = PCamDataset('pcam/training_split.h5', 'Labels/Labels/camelyonpatch_level_2_split_train_y.h5', transform=train_transform)
    # load test_data
    test_data = PCamDataset('pcam/test_split.h5', 'Labels/Labels/camelyonpatch_level_2_split_test_y.h5', transform=test_transform)
    # load validation data
    val_data = PCamDataset('pcam/validation_split.h5', 'Labels/Labels/camelyonpatch_level_2_split_valid_y.h5', transform=test_transform)

    ### get the dataloaders
    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )

    return train_loader, test_loader, val_loader