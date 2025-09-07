import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class WaterbirdsDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Path to root directory (contains metadata.csv and folders per species)
            split (str): One of 'train', 'val', 'test'
            transform (callable, optional): Image transform function
        """
        self.root_dir = root_dir
        self.transform = transform

        metadata_path = os.path.join(root_dir, 'metadata.csv')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"metadata.csv not found at {metadata_path}")

        self.metadata = pd.read_csv(metadata_path)

        split_map = {'train': 0, 'val': 1, 'test': 2}
        if split not in split_map:
            raise ValueError(f"Split must be one of {list(split_map.keys())}")

        # Filter by split
        self.metadata = self.metadata[self.metadata['split'] == split_map[split]].reset_index(drop=True)

        # Build a mapping from filename to full path by scanning all folders
        self.filename_to_path = self._build_filepath_index()

    def _build_filepath_index(self):
        """
        Traverse all subfolders and build a full relative path → full absolute path mapping
        """
        filename_to_path = {}
        for dirpath, _, filenames in os.walk(self.root_dir):
            for fname in filenames:
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(dirpath, fname)
                    rel_path = os.path.relpath(full_path, start=self.root_dir).replace('\\', '/')
                    filename_to_path[rel_path] = full_path
        return filename_to_path


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        fname = row['img_filename']
        label = int(row['y'])

        if fname not in self.filename_to_path:
            raise FileNotFoundError(f"Could not find image file: {fname}")

        image_path = self.filename_to_path[fname]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
