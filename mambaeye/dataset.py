import os
import logging
import random
import torch
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Any
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import v2

from mambaeye.positional_encoding import sinusoidal_position_encoding_2d

def find_coeffs(target_pts, source_pts):
    """
    Compute the perspective transformation coefficients.
    target_pts: destination points (output plane)
    source_pts: source points (input plane)
    """
    matrix = []
    for p1, p2 in zip(target_pts, source_pts):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = np.matrix(matrix, dtype=float)
    B = np.array(source_pts).flatten()
    res = np.linalg.solve(A, B)
    return np.asarray(res).squeeze()


def apply_perspective(image, distortion_scale=0.3):
    """
    Apply random perspective transformation to the image without radical size scaling.
    - Computes output size based on distorted points' bounding box.
    - Uses transparent fillcolor for areas outside the transform.
    - Returns the transformed image and its bounding box (left, top, right, bottom).

    Args:
    - image: PIL Image object.
    - distortion_scale: Float, controls the maximum distortion (0.0 to 1.0).

    Returns:
    - transformed_image: PIL Image after perspective transform.
    - bbox: Tuple (left, top, right, bottom) of the non-transparent content.
    """
    width, height = image.size
    # Source points: original corners
    source_pts = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=float
    )

    # Generate random offsets for target points
    delta = distortion_scale * min(width, height) / 2
    offsets = np.random.uniform(-delta, delta, (4, 2))
    target_pts = source_pts + offsets

    # Compute bounding box of target points to set output size
    min_x, min_y = target_pts.min(axis=0)
    max_x, max_y = target_pts.max(axis=0)
    out_width = int(np.ceil(max_x - min_x))
    out_height = int(np.ceil(max_y - min_y))

    # Shift target points to start at (0, 0)
    target_pts -= np.array([min_x, min_y])

    # Compute perspective coefficients
    coeffs = find_coeffs(target_pts, source_pts)

    # Convert to RGBA for transparent fill
    img_rgba = image.convert("RGB")

    # Apply transformation with transparent fillcolor
    transformed = img_rgba.transform(
        (out_width, out_height),
        Image.PERSPECTIVE,
        coeffs,
        resample=Image.BICUBIC,
        fillcolor=(0, 0, 0),
    )

    # Get bounding box of non-transparent regions
    bbox = transformed.getbbox()

    return transformed, bbox


class ImagenetDatasetSinusoidal(Dataset):
    """
    Dataset class for training with Sinusoidal Positional Embeddings and perspective augmentations.
    
    Extracts variable-length sequences of patches from images after applying random
    scaling, cropping, and perspective transformations to mimic real-world continuous sampling.
    """
    def __init__(
        self,
        img_dir: str,
        label_path: Optional[str] = None,
        patch_size: int = 16,
        move_embedding_dim: int = 256,
        min_canvas_size: int = 256,
        max_canvas_size: int = 512,
        sequence_length: int = 512,
        validate: bool = False,
        padding_value: int = 0,
        perspective_prob: float = 0.3,
        perspective_distortion: float = 0.5,
        crop_prob: float = 0.8,
        random_erase_prob: float = 0.4,
        localized_seq_prob: float = 0.5,
    ):
        """
        Args:
            img_dir: Path to the directory containing image class folders.
            label_path: Optional path to label mapping file.
            patch_size: Size of the patches to extract.
            move_embedding_dim: Dimension for sinusoidal positioning embedding.
            min_canvas_size: Minimum canvas size for random scaling.
            max_canvas_size: Maximum canvas size or canvas size for validation.
            sequence_length: Number of patches to extract per sequence.
            validate: If True, applies deterministic transformations and skips augmentations.
            padding_value: Pixel value to use for padding areas outside the image.
        """
        super().__init__()
        self.img_dir = img_dir
        self.label_path = label_path
        self.patch_size = patch_size
        self.move_embedding_dim = move_embedding_dim
        self.max_canvas_size = max_canvas_size
        self.min_canvas_size = min_canvas_size
        self.sequence_length = sequence_length
        self.validate = validate
        self.padding_value = padding_value
        self.perspective_prob = perspective_prob
        self.perspective_distortion = perspective_distortion
        self.crop_prob = crop_prob
        self.random_erase_prob = random_erase_prob
        self.localized_seq_prob = localized_seq_prob

        self.img_data = []

        self.totensor = torchvision.transforms.ToTensor()
        self.random_erase = torchvision.transforms.RandomErasing(
            p=self.random_erase_prob,
            value=padding_value,
        )
        self.pre_transforms = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply(
                    [
                        T.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                        )
                    ],
                    p=0.4,
                ),
                T.RandomGrayscale(p=0.2),
                T.RandomApply(
                    [v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0))], p=0.2
                ),
            ]
        )

        image_class_folder = os.listdir(img_dir)
        image_class_folder.sort()
        # only for folders
        image_class_folder = [f for f in image_class_folder if os.path.isdir(os.path.join(img_dir, f))]
        for class_num, folder_name in enumerate(image_class_folder):
            folder_path = os.path.join(img_dir, folder_name)
            image_path_list = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            image_path_list.sort()
            for image_name in image_path_list:
                label = int(class_num)
                image_path = os.path.join(folder_path, image_name)
                self.img_data.append((image_path, label))

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]]:
        """
        Get a single transformed image sequence and its metadata.

        Args:
            idx: Index of the image in the dataset.

        Returns:
            A tuple of (img_sequence, move_embedding, information_ratio, label, absolute_location).
            Returns None if image transformation or reading fails.
        """
        img_path, label = self.img_data[idx]
        img = Image.open(img_path).convert("RGB")
        if not self.validate:
            canvas_size = random.randint(self.min_canvas_size, self.max_canvas_size)
        else:
            canvas_size = self.max_canvas_size

        width, height = img.size  # PIL: width=horizontal (W), height=vertical (H)
        ratio = min(canvas_size / width, canvas_size / height)
        width = int(width * ratio)  # Still horizontal
        height = int(height * ratio)  # Still vertical
        if width > canvas_size or height > canvas_size:
            img = img.resize((width, height), Image.Resampling.LANCZOS)
        else:
            img = img.resize((width, height), Image.Resampling.BICUBIC)

        if not self.validate:
            img = self.pre_transforms(img)

            # random perspective transformation
            if random.random() < self.perspective_prob:
                try:
                    img, _ = apply_perspective(img, distortion_scale=self.perspective_distortion)
                    width, height = (
                        img.size
                    )  # Update after perspective (still W horiz, H vert)
                except (ValueError, OSError, RuntimeError) as e:
                    logging.warning(f"Perspective transform failed for {img_path}: {e}")

            # random crop & resize
            try:
                min_side = max(self.patch_size, int(0.3 * min(width, height)))
                x_random_width = random.randint(
                    min_side, width - min_side
                )  # horizontal crop width
                y_random_height = random.randint(
                    min_side, height - min_side
                )  # vertical crop height (note: renamed for clarity, but y=vert here)
                x_offset = random.randint(0, width - x_random_width)  # horiz offset
                y_offset = random.randint(0, height - y_random_height)  # vert offset

                if random.random() < self.crop_prob:
                    img = img.crop(
                        (
                            x_offset,
                            y_offset,
                            x_offset + x_random_width,
                            y_offset + y_random_height,
                        )
                    )
                else:
                    img = img.crop((0, 0, width, height))

                resize_width = random.randint(int(min_side), canvas_size)  # horiz
                resize_height = random.randint(int(min_side), canvas_size)  # vert

                img = img.resize(
                    (resize_width, resize_height), Image.Resampling.LANCZOS
                )
            except (ValueError, OSError, RuntimeError) as e:
                logging.warning(f"Crop/resize failed for {img_path}: {e}")
                return None  # Consider improving error handling as previously noted

            img = self.totensor(img)
            img = self.random_erase(img)
            width, height = (
                img.shape[2],
                img.shape[1],
            )  # FIXED: Now width=horizontal (img.shape[2]=W), height=vertical (img.shape[1]=H) to match PIL

            canvas = (
                torch.ones(3, canvas_size, canvas_size, dtype=torch.float32)
                * self.padding_value
            )
            row_offset = (
                canvas_size - height
            ) // 2  # FIXED: Renamed to row_offset (vertical/rows/H)
            col_offset = (
                canvas_size - width
            ) // 2  # FIXED: col_offset (horizontal/cols/W)
            canvas[
                :,
                row_offset : row_offset + height,
                col_offset : col_offset + width,
            ] = img
            img = canvas

        else:
            img = self.totensor(img)
            width, height = (
                img.shape[2],
                img.shape[1],
            )  # FIXED: Same as above, width=horiz/W, height=vert/H
            canvas = (
                torch.ones(3, canvas_size, canvas_size, dtype=torch.float32)
                * self.padding_value
            )
            row_offset = (canvas_size - height) // 2  # FIXED: row_offset
            col_offset = (canvas_size - width) // 2  # FIXED: col_offset
            canvas[
                :,
                row_offset : row_offset + height,
                col_offset : col_offset + width,
            ] = img
            img = canvas

        row_end = max(row_offset + 1, row_offset + height)
        row_end = min(row_end, canvas_size - self.patch_size + 1)
        col_end = max(col_offset + 1, col_offset + width)
        col_end = min(col_end, canvas_size - self.patch_size + 1)

        if self.validate or random.random() < self.localized_seq_prob:
            row_coords = torch.randint(
                row_offset,
                row_end,
                (self.sequence_length,),
                dtype=torch.long,
            )
            col_coords = torch.randint(
                col_offset,
                col_end,
                (self.sequence_length,),
                dtype=torch.long,
            )
        else:
            row_coords = torch.randint(
                0,
                canvas_size - self.patch_size + 1,
                (self.sequence_length,),
                dtype=torch.long,
            )
            col_coords = torch.randint(
                0,
                canvas_size - self.patch_size + 1,
                (self.sequence_length,),
                dtype=torch.long,
            )
        all_patches_unfolded = img.unfold(1, self.patch_size, 1).unfold(
            2, self.patch_size, 1
        )
        # Permute to (num_patches_H, num_patches_W, C, patch_H, patch_W)
        all_patches_permuted = all_patches_unfolded.permute(1, 2, 0, 3, 4)
        del all_patches_unfolded

        # Select the required patches using the random coordinates
        img_sequence_patches = all_patches_permuted[
            row_coords, col_coords
        ]  # FIXED: row_coords for dim0 (H/rows), col_coords for dim1 (W/cols)
        del all_patches_permuted

        # First sequence (original order)
        img_sequence1 = img_sequence_patches.flatten(start_dim=1)
        absolute_location1 = torch.stack(
            [row_coords.float(), col_coords.float()], dim=1
        )  # (L, 2)
        initial_row1 = torch.tensor([row_coords[0].item()])
        initial_col1 = torch.tensor([col_coords[0].item()])
        full_row_coords1 = torch.cat((initial_row1, row_coords))
        full_col_coords1 = torch.cat((initial_col1, col_coords))
        move_row1 = (full_row_coords1[1:] - full_row_coords1[:-1]).float()
        move_col1 = (full_col_coords1[1:] - full_col_coords1[:-1]).float()
        move_sequence1 = torch.stack([move_row1, move_col1], dim=1)  # (L, 2)
        move_embedding1 = sinusoidal_position_encoding_2d(
            move_sequence1, self.move_embedding_dim
        )

        mask1 = torch.zeros(height, width, dtype=torch.bool, device=row_coords.device)
        information_ratio1 = torch.empty(
            len(row_coords), dtype=torch.float32, device=row_coords.device
        )

        covered = torch.tensor(0, dtype=torch.int64, device=row_coords.device)
        total = height * width

        row_coords = row_coords - row_offset
        col_coords = col_coords - col_offset

        for i in range(len(row_coords)):
            # Use r0/c0 for row/col to maintain consistency
            # row_coords indexes height (vertical), col_coords indexes width (horizontal)
            r0 = int(row_coords[i])
            c0 = int(col_coords[i])

            # clip to canvas in case a patch goes past the edge
            r1 = min(r0 + self.patch_size, height)
            c1 = min(c0 + self.patch_size, width)
            r1 = max(r1, 0)
            c1 = max(c1, 0)

            r0 = min(r0, height)
            c0 = min(c0, width)
            r0 = max(r0, 0)
            c0 = max(c0, 0)

            if r1 - r0 == 0 or c1 - c0 == 0:
                newly_covered = 0
            else:
                region = mask1[r0:r1, c0:c1]  # Correct: mask[row, col]
                newly_covered = region.numel() - region.sum(dtype=torch.int64)
                region.fill_(True)  # in-place set to 1/True

            covered += newly_covered
            information_ratio1[i] = covered.float() / total

        del img_sequence_patches

        return (
            img_sequence1,
            move_embedding1,
            information_ratio1,
            label,
            absolute_location1,
        )

    def __len__(self):
        return len(self.img_data)


class InferenceDataset(Dataset):
    """
    Dataset class for inference, formatting images into a uniform canvas grid format 
    while optionally resampling based on a custom `resize_mode`.
    """
    def __init__(
        self,
        img_dir: str,
        patch_size: int = 16,
        canvas_size: int = 512,
        padding_value: int = 0,
        resize_mode: str = "none",
    ):
        """
        Args:
            img_dir: Directory containing image class folders.
            patch_size: Patch resolution used by the model.
            canvas_size: Standardized background canvas size.
            padding_value: Pixel value to pad non-image areas.
            resize_mode: Method to resize the image ('none', 'fit', 'full').
        """
        super().__init__()
        self.img_dir = img_dir
        self.patch_size = patch_size
        self.canvas_size = canvas_size
        self.padding_value = padding_value
        self.resize_mode = resize_mode
        self.img_data = []
        image_class_folder = os.listdir(img_dir)
        # only for folders
        image_class_folder = [f for f in image_class_folder if os.path.isdir(os.path.join(img_dir, f))]
        image_class_folder.sort()
        for class_num, folder_name in enumerate(image_class_folder):
            folder_path = os.path.join(img_dir, folder_name)
            image_path_list = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            image_path_list.sort()
            for image_name in image_path_list:
                label = int(class_num)
                image_path = os.path.join(folder_path, image_name)
                self.img_data.append((image_path, label))
        self.totensor = torchvision.transforms.ToTensor()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single inference image with metadata boundaries.

        Args:
            idx: Index of the image.

        Returns:
            Tuple of (canvas, label, original_size, top_left_offset, bottom_right_end_bounds).
        """
        img_path, label = self.img_data[idx]
        img = Image.open(img_path).convert("RGB")

        width, height = img.size
        if width > self.canvas_size or height > self.canvas_size:
            # resize with the same ratio
            ratio = min(self.canvas_size / width, self.canvas_size / height)
            width = int(width * ratio)
            height = int(height * ratio)
            img = img.resize((width, height), Image.Resampling.LANCZOS)

        if self.resize_mode == "none":
            pass
        elif self.resize_mode == "fit":
            ratio = min(self.canvas_size / width, self.canvas_size / height)
            width = int(width * ratio)
            height = int(height * ratio)
            img = img.resize((width, height), Image.Resampling.LANCZOS)
        elif self.resize_mode == "full":
            img = img.resize(
                (self.canvas_size, self.canvas_size), Image.Resampling.LANCZOS
            )

        img_tensor = self.totensor(img)
        canvas = (
            torch.ones(3, self.canvas_size, self.canvas_size, dtype=torch.float32)
            * self.padding_value
        )
        x_offset = (self.canvas_size - img_tensor.shape[1]) // 2
        y_offset = (self.canvas_size - img_tensor.shape[2]) // 2
        canvas[
            :,
            x_offset : x_offset + img_tensor.shape[1],
            y_offset : y_offset + img_tensor.shape[2],
        ] = img_tensor

        # Calculate bounds for valid patch sampling (consistent with ImagenetDataset validation mode)
        # Use original image dimensions, not canvas dimensions!
        img_height = img_tensor.shape[1]  # Original image height
        img_width = img_tensor.shape[2]  # Original image width

        x_end = max(x_offset + 1, x_offset + img_height)
        x_end = min(x_end, self.canvas_size - self.patch_size + 1)
        y_end = max(y_offset + 1, y_offset + img_width)
        y_end = min(y_end, self.canvas_size - self.patch_size + 1)

        return (
            canvas,
            label,
            torch.tensor([width, height]),
            torch.tensor([x_offset, y_offset]),
            torch.tensor([x_end, y_end]),
        )

    def __len__(self):
        return len(self.img_data)


def collate_fn_keep_batch_size(batch):
    """Top-level, picklable collate_fn that removes None and keeps batch size by duplication.

    If some samples are None, duplicates valid samples to maintain the original
    batch size. Returns None if all items are None so the caller can skip.
    """
    target_size = len(batch)
    valid_items = [sample for sample in batch if sample is not None]

    if len(valid_items) == 0:
        return None

    # Duplicate valid items to reach the target batch size
    while len(valid_items) < target_size:
        valid_items.append(random.choice(valid_items))

    first_item = valid_items[0]

    if torch.is_tensor(first_item):
        return torch.stack(valid_items, dim=0)
    if isinstance(first_item, tuple):
        transposed_fields = list(zip(*valid_items))
        collated_fields = []
        for field_items in transposed_fields:
            exemplar = field_items[0]
            if torch.is_tensor(exemplar):
                collated_fields.append(torch.stack(field_items, dim=0))
            elif isinstance(exemplar, np.ndarray):
                tensors = [torch.from_numpy(array) for array in field_items]
                collated_fields.append(torch.stack(tensors, dim=0))
            elif isinstance(exemplar, (int, np.integer)):
                collated_fields.append(torch.tensor(field_items, dtype=torch.long))
            elif isinstance(exemplar, (float, np.floating)):
                collated_fields.append(torch.tensor(field_items, dtype=torch.float32))
            else:
                collated_fields.append(list(field_items))
        return tuple(collated_fields)
    return valid_items