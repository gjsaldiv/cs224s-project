from torch.utils.data import Dataset
import numpy as np
import torch 

class CREMADataset(Dataset):
    """
    CREMA-D dataset to load and use for transformer training/testing
    """
    def __init__(self, X, Y, num_examples, split='train', train_ratio = 0.8, val_ratio = 0.1):
        super().__init__()
        # X are the features, Y are the labels
        self.Y = Y
        self.X = X
        self.num_examples = num_examples
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        # Decides which indices belong to which split.
        train_indices, val_indices, test_indices = self.split_data(num_examples, train_ratio=train_ratio, val_ratio=val_ratio)

        if split == 'train':
            indices = train_indices
        elif split == 'val':
            indices = val_indices
        elif split == 'test':
            indices = test_indices
        else:
            raise Exception(f'Split {split} not supported.')

        self.indices = indices

    def split_data(self, num_examples, train_ratio = 0.8, val_ratio = 0.1):
        """Splits data into train, val, and test sets based on speaker. When
        evaluating methods on the test split, we measure how well they generalize
        to new (unseen) speakers.

        Concretely, this stores and returns indices belonging to each split.
        """
        # Fix seed so everyone reproduces the same splits.
        rs = np.random.RandomState(42)

        indices = np.arange(0, num_examples)
        train_idx = int(num_examples * train_ratio)
        num_remaining = num_examples - train_idx
        val_idx = train_idx + int(num_remaining/2)

        print(f'train idx: {train_idx}')
        print(f'val idx: {val_idx}')

        train_indices = indices[:train_idx]
        val_indices = indices[train_idx:val_idx]
        test_indices = indices[val_idx:]
        return train_indices, val_indices, test_indices
    
    def __getitem__(self, index):
        index = self.indices[index]
        return torch.FloatTensor(self.X[index,:]), torch.LongTensor([self.Y[index]])

    def __len__(self):
        """Returns total number of utterances in the dataset."""
        return len(self.indices)
