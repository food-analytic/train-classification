import os
import shutil
import argparse
import random
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision.datasets import ImageFolder


class CustomImageFolder(ImageFolder):
    def __init__(self, excluded_classes=None, **kwargs):
        self.excluded_classes = (
            set(excluded_classes) if excluded_classes is not None else set()
        )
        super().__init__(**kwargs)

    # Override the find_classes method to allow excluding some classes that the size is too small
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(
            entry.name
            for entry in os.scandir(directory)
            if entry.is_dir() and entry.name not in self.excluded_classes
        )
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


def find_low_size_classes(imgs, threshold):
    df_dataset = pd.DataFrame(
        imgs,
        columns=["image_path", "class_name"],
    )
    df_class_count = (
        df_dataset.groupby("class_name")
        .count()
        .rename(columns={"image_path": "sample_size"})
    )
    low_size_classes = set(df_class_count[df_class_count.sample_size < threshold].index)
    return low_size_classes


def get_train_val_test_indices(imgs, val_length, test_length, random_state=None):
    df_dataset = pd.DataFrame(
        imgs,
        columns=["image_path", "class_name"],
    )
    val_test_length = val_length + test_length
    val_test_indices = (
        df_dataset.groupby("class_name")
        .sample(n=val_test_length, random_state=random_state)
        .index
    )
    df_train = df_dataset.drop(index=val_test_indices)
    df_val_test = df_dataset.loc[val_test_indices]
    test_indices = (
        df_dataset.groupby("class_name")
        .sample(n=test_length, random_state=random_state)
        .index
    )
    df_val = df_val_test.drop(index=test_indices)
    df_test = df_val_test.loc[test_indices]
    train_indices = df_train.index
    val_indices = df_val.index
    test_indices = df_test.index
    return train_indices, val_indices, test_indices


def create_split(dataset, indices, directory):
    for index in tqdm(indices, desc=f"Creating {os.path.basename(directory)}"):
        image_path, class_idx = dataset.imgs[index]
        image_path = os.path.abspath(image_path)
        class_name = dataset.classes[class_idx]
        os.makedirs(os.path.join(directory, class_name), exist_ok=True)
        shutil.copy(image_path, os.path.join(directory, class_name))


def main():
    args = argparse.ArgumentParser()
    args.add_argument(
        "-d", "--image_dir", type=str, default="data/full", help="image directory"
    )
    args.add_argument(
        "-trd",
        "--train_dir",
        type=str,
        default="data/splits/train",
        help="train directory",
    )
    args.add_argument(
        "-vd", "--val_dir", type=str, default="data/splits/val", help="val directory"
    )
    args.add_argument(
        "-ted",
        "--test_dir",
        type=str,
        default="data/splits/test",
        help="test directory",
    )
    args.add_argument("-s", "--seed", type=int, default=42, help="random seed")
    args.add_argument(
        "-m",
        "--min_samples",
        type=int,
        default=60,
        help="minimum number of samples per class",
    )
    args.add_argument(
        "-v", "--val_length", type=int, default=15, help="number of validation samples"
    )
    args.add_argument(
        "-t", "--test_length", type=int, default=15, help="number of test samples"
    )
    args = args.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset = CustomImageFolder(root=args.image_dir)
    low_size_classes = find_low_size_classes(dataset.imgs, args.min_samples)
    low_size_classes = {dataset.classes[class_idx] for class_idx in low_size_classes}

    dataset = CustomImageFolder(root=args.image_dir, excluded_classes=low_size_classes)
    train_indices, val_indices, test_indices = get_train_val_test_indices(
        dataset.imgs, args.val_length, args.test_length, random_state=args.seed
    )

    create_split(dataset, train_indices, args.train_dir)
    create_split(dataset, val_indices, args.val_dir)
    create_split(dataset, test_indices, args.test_dir)
    print("Done")


if __name__ == "__main__":
    main()
