import random
from typing import Optional

import hydra
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.models import ResNet50_Weights

from mambaeye.dataset import InferenceDataset
from mambaeye.mambaeye_pl import MambaEyePL
from mambaeye.model import MambaEye
from mambaeye.positional_encoding import sinusoidal_position_encoding_2d
from mambaeye.scan import generate_scan_positions

# Fix the random seed for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")

MODEL_ALIASES = {
    "tiny": ("usingcolor/MambaEye-tiny", "mambaeye_tiny.pt"),
    "tiny-ft": ("usingcolor/MambaEye-tiny", "mambaeye_tiny_ft.pt"),
    "small": ("usingcolor/MambaEye-small", "mambaeye_small.pt"),
    "small-ft": ("usingcolor/MambaEye-small", "mambaeye_small_ft.pt"),
    "base": ("usingcolor/MambaEye-base", "mambaeye_base.pt"),
    "base-ft": ("usingcolor/MambaEye-base", "mambaeye_base_ft.pt"),
}

# Mapping from model_name alias to the corresponding Hydra model config
MODEL_CONFIGS = {
    "tiny": "tiny_12layers",
    "tiny-ft": "tiny_12layers",
    "small": "small_24layers",
    "small-ft": "small_24layers",
    "base": "base_48layers",
    "base-ft": "base_48layers",
}


def _compute_move_embedding(
    patch_location: torch.Tensor,
    cur_location: Optional[torch.Tensor],
) -> torch.Tensor:
    if cur_location is None:
        move_embedding = torch.zeros(
            (patch_location.shape[0], 2),
            dtype=torch.float32,
            device=patch_location.device,
        )
        move_embedding = sinusoidal_position_encoding_2d(move_embedding, 256)
        return move_embedding

    return sinusoidal_position_encoding_2d(
        (patch_location - cur_location).float(),
        256,
    )


def _load_model(cfg, checkpoint_path, device):
    """Load model from checkpoint (.ckpt or .pt)."""
    if checkpoint_path.endswith(".ckpt"):
        model = MambaEyePL.load_from_checkpoint(checkpoint_path)
        model = model.model
    elif checkpoint_path.endswith(".pt"):
        model = MambaEye(**cfg.model)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        raise ValueError("Checkpoint path must end with .ckpt or .pt")
    model.to(device)
    model.eval()
    return model


def _resolve_checkpoint(cfg, model_name, hf_repo, hf_checkpoint):
    """Resolve the checkpoint path, downloading from HuggingFace if needed."""
    checkpoint_path = cfg.get("ckpt_path", None)
    if checkpoint_path is None and hf_checkpoint is None:
        raise ValueError("Either ckpt_path or model_name must be provided.")
    if checkpoint_path is None:
        from huggingface_hub import hf_hub_download
        print(f"Downloading {hf_checkpoint} from HuggingFace repo {hf_repo}...")
        checkpoint_path = hf_hub_download(repo_id=hf_repo, filename=hf_checkpoint)
    return checkpoint_path



# --- GIF generation helpers ---
_canvas_image = None
_categories = None
_sequence_length = None
_patch_size = None


def _init_gif_worker(canvas_img, cats, seq_len, ps):
    global _canvas_image, _categories, _sequence_length, _patch_size
    _canvas_image = canvas_img
    _categories = cats
    _sequence_length = seq_len
    _patch_size = ps


def _generate_gif_frame(args):
    step, seq_pos_history, cur_pos, step_probs = args
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.cm as cm
    import numpy as np
    from io import BytesIO
    import imageio

    plt.style.use("default")
    fig = plt.figure(figsize=(14, 7), facecolor="white")

    ax_img = fig.add_axes([0.05, 0.1, 0.45, 0.8])
    ax_bar = fig.add_axes([0.65, 0.15, 0.3, 0.7])
    ax_img.set_facecolor("white")
    ax_bar.set_facecolor("white")

    # Visited mask
    visited_mask = np.zeros(
        (_canvas_image.shape[0], _canvas_image.shape[1], 1), dtype=np.float32
    )
    for px, py in seq_pos_history:
        visited_mask[px : px + _patch_size, py : py + _patch_size] = 1.0

    # Lighten unvisited areas (fade to white)
    unseen = np.ones_like(_canvas_image) * 0.92 + _canvas_image * 0.08
    display_img = _canvas_image * visited_mask + unseen * (1 - visited_mask)

    ax_img.imshow(display_img)
    ax_img.axis("off")
    ax_img.set_title(
        f"Inference Step {step + 1} / {_sequence_length}",
        color="#222222",
        fontsize=16,
        pad=15,
        fontweight="bold",
    )

    # Current patch highlight
    cur_x, cur_y = cur_pos
    rect = mpatches.Rectangle(
        (cur_y, cur_x),
        _patch_size,
        _patch_size,
        linewidth=3,
        edgecolor="#0066FF",
        facecolor="none",
    )
    rect_fill = mpatches.Rectangle(
        (cur_y, cur_x),
        _patch_size,
        _patch_size,
        linewidth=0,
        edgecolor="none",
        facecolor="#0066FF",
        alpha=0.2,
    )
    ax_img.add_patch(rect)
    ax_img.add_patch(rect_fill)

    # Top-5 bar chart
    top5_idx = np.argsort(step_probs)[-5:][::-1]
    top5_probs = step_probs[top5_idx]

    max_label_len = 25
    top5_labels = []
    for idx in top5_idx:
        label = _categories[idx].split(",")[0]
        if len(label) > max_label_len:
            label = label[: max_label_len - 3] + "..."
        top5_labels.append(label.title())

    y_pos = np.arange(len(top5_labels))
    colors = cm.plasma(top5_probs / (np.max(top5_probs) + 1e-6) * 0.8 + 0.2)

    bars = ax_bar.barh(
        y_pos, top5_probs, align="center", color=colors, height=0.6, edgecolor="none"
    )
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(top5_labels, color="#222222", fontsize=12)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Confidence", color="#555555", fontsize=12, labelpad=10)
    ax_bar.set_title(
        "Top Predictions", color="#222222", fontsize=16, fontweight="bold", pad=20
    )
    ax_bar.set_xlim(0, 1.05)
    ax_bar.xaxis.grid(True, linestyle="--", alpha=0.15, color="#888888")
    ax_bar.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax_bar.spines[spine].set_visible(False)
    ax_bar.spines["bottom"].set_color("#CCCCCC")
    ax_bar.tick_params(axis="x", colors="#555555")
    ax_bar.tick_params(axis="y", colors="#222222", length=0, pad=10)

    for bar in bars:
        w = bar.get_width()
        ax_bar.text(
            w + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.1%}",
            ha="left",
            va="center",
            color="#0066FF",
            fontsize=12,
            fontweight="bold",
        )

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    frame = imageio.v2.imread(buf)
    return step, frame


def _run_single_image_inference(cfg, model, device):
    """Run inference on a single image, print top-5 predictions, and generate a GIF."""
    import multiprocessing
    import numpy as np
    import imageio

    image_path = cfg.get("image_path")
    patch_size = cfg.dataset.val.patch_size
    sequence_length = cfg.dataset.val.sequence_length
    scan_pattern = cfg.get("scan_pattern", "random")
    resize_mode = cfg.get("resize_mode", "none")

    # ImageNet categories
    categories = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]

    # Load and preprocess image
    print(f"Loading image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    canvas_size = max(width, height)

    # Apply resize mode
    if resize_mode == "fit":
        target_canvas = cfg.dataset.val.max_canvas_size
        ratio = min(target_canvas / width, target_canvas / height)
        width = int(width * ratio)
        height = int(height * ratio)
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        canvas_size = target_canvas
    elif resize_mode == "full":
        target_canvas = cfg.dataset.val.max_canvas_size
        img = img.resize((target_canvas, target_canvas), Image.Resampling.LANCZOS)
        width, height = target_canvas, target_canvas
        canvas_size = target_canvas

    totensor = T.ToTensor()
    img_tensor = totensor(img)

    canvas = torch.zeros(3, canvas_size, canvas_size, dtype=torch.float32)
    x_offset = (canvas_size - img_tensor.shape[1]) // 2
    y_offset = (canvas_size - img_tensor.shape[2]) // 2
    canvas[
        :,
        x_offset : x_offset + img_tensor.shape[1],
        y_offset : y_offset + img_tensor.shape[2],
    ] = img_tensor

    img_height = img_tensor.shape[1]
    img_width = img_tensor.shape[2]

    x_end = max(x_offset + 1, x_offset + img_height)
    x_end = min(x_end, canvas_size - patch_size + 1)
    y_end = max(y_offset + 1, y_offset + img_width)
    y_end = min(y_end, canvas_size - patch_size + 1)

    img_canvas = canvas.unsqueeze(0).to(device)  # (1, 3, canvas_size, canvas_size)

    # Generate scan positions
    seq_pos = generate_scan_positions(
        x_start=x_offset,
        x_stop=x_end,
        y_start=y_offset,
        y_stop=y_end,
        patch_size=patch_size,
        sequence_length=sequence_length,
        scan_pattern=scan_pattern,
        rng=random,
    )

    patch_sequences = torch.tensor(seq_pos, dtype=torch.long, device=device).unsqueeze(
        0
    )  # (1, L, 2)
    cur_location = None

    patches_list = []
    moves_list = []

    # Extract sequence patches
    for i in range(sequence_length):
        patch_location = patch_sequences[:, i, :]  # (1, 2)

        move_embedding = _compute_move_embedding(patch_location, cur_location)
        cur_location = patch_location

        rows = patch_location[:, 0].unsqueeze(1) + torch.arange(
            patch_size, device=device
        ).unsqueeze(0)
        cols = patch_location[:, 1].unsqueeze(1) + torch.arange(
            patch_size, device=device
        ).unsqueeze(0)

        rows = rows.clamp(0, img_canvas.shape[2] - patch_size)
        cols = cols.clamp(0, img_canvas.shape[3] - patch_size)

        patch = img_canvas[
            torch.arange(1, device=device).unsqueeze(1).unsqueeze(2),
            :,
            rows.unsqueeze(2),
            cols.unsqueeze(1),
        ]  # (1, patch_size, patch_size, C)

        patch = patch.permute(0, 3, 1, 2).contiguous()  # (1, C, P, P)
        patch_flat = patch.flatten(start_dim=1)  # (1, C*P*P)

        patches_list.append(patch_flat)
        moves_list.append(move_embedding)

    img_sequence = torch.stack(patches_list, dim=1)  # (1, L, C*P*P)
    move_sequence = torch.stack(moves_list, dim=1)  # (1, L, move_dim)

    # Model inference
    print(f"Running inference ({sequence_length} steps, scan={scan_pattern})...")
    classification_output = model(img_sequence, move_sequence)  # (1, L, num_classes)

    # Softmax probabilities for all steps (needed for GIF)
    all_probs = F.softmax(classification_output[0], dim=-1).cpu().numpy()  # (L, num_classes)

    # Final step prediction summary
    final_probs = all_probs[-1]
    top5_idx = final_probs.argsort()[-5:][::-1]
    top5_vals = final_probs[top5_idx]

    print(f"\n{'='*50}")
    print(f"  Inference Result: {image_path}")
    print(f"  Image size: {width}x{height} | Canvas: {canvas_size}x{canvas_size}")
    print(f"  Sequence length: {sequence_length} | Scan: {scan_pattern}")
    print(f"{'='*50}")
    print(f"  {'Rank':<6} {'Class':<30} {'Confidence':>10}")
    print(f"  {'-'*46}")
    for rank, (prob, idx) in enumerate(zip(top5_vals, top5_idx), 1):
        class_name = categories[idx].split(",")[0].title()
        print(f"  {rank:<6} {class_name:<30} {prob:>9.2%}")
    print(f"{'='*50}")

    # Generate inference video
    print("Generating video frames (parallel)...")
    canvas_image = np.transpose(canvas.cpu().numpy(), (1, 2, 0))  # (H, W, 3)

    tasks = []
    for step in range(sequence_length):
        tasks.append((step, seq_pos[: step + 1], seq_pos[step], all_probs[step]))

    frames = [None] * sequence_length
    with multiprocessing.Pool(
        processes=multiprocessing.cpu_count(),
        initializer=_init_gif_worker,
        initargs=(canvas_image, categories, sequence_length, patch_size),
    ) as pool:
        for step, frame in tqdm(
            pool.imap_unordered(_generate_gif_frame, tasks), total=sequence_length
        ):
            frames[step] = frame

    print("Saving inference.mp4...")
    imageio.mimsave("inference.mp4", frames, fps=30, format="FFMPEG")
    print("Saved inference.mp4 successfully!")

    return int(top5_idx[0])


def _run_dataset_validation(cfg, model, device):
    """Run validation on the full dataset and save accuracy results."""
    val_dir = cfg.dataset.val.img_dir
    sequence_length = cfg.dataset.val.sequence_length
    canvas_size = cfg.dataset.val.max_canvas_size
    patch_size = cfg.dataset.val.patch_size
    resize_mode = cfg.get("resize_mode", "none")
    scan_pattern = cfg.get("scan_pattern", "random")

    # Create dataset
    val_dataset = InferenceDataset(
        val_dir,
        patch_size=patch_size,
        canvas_size=canvas_size,
        resize_mode=resize_mode,
    )

    val_dataloader = torch.utils.data.DataLoader(val_dataset, **cfg.dataloader.val)

    acc_list = [0.0 for _ in range(sequence_length)]
    total_sample = 0

    print(f"Starting validation with {len(val_dataset)} samples...")

    for batch_idx, batch in enumerate(tqdm(val_dataloader, desc="Validation")):
        img_canvas, label, sizes, offsets, bounds = batch

        img_canvas = img_canvas.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        batch_size_current = img_canvas.shape[0]

        x_offset = offsets[:, 0].to(device)
        y_offset = offsets[:, 1].to(device)
        x_end = bounds[:, 0].to(device)
        y_end = bounds[:, 1].to(device)

        patches_list = []
        moves_list = []

        patch_sequences = []
        for b in range(batch_size_current):
            seq_pos = generate_scan_positions(
                x_start=x_offset[b].item(),
                x_stop=x_end[b].item(),
                y_start=y_offset[b].item(),
                y_stop=y_end[b].item(),
                patch_size=patch_size,
                sequence_length=sequence_length,
                scan_pattern=scan_pattern,
                rng=random,
            )
            patch_sequences.append(
                torch.tensor(seq_pos, dtype=torch.long, device=device)
            )

        patch_sequences = torch.stack(patch_sequences, dim=0)  # (B, L, 2)

        cur_location = None

        for i in range(sequence_length):
            patch_location = patch_sequences[:, i, :]  # (B, 2)

            move_embedding = _compute_move_embedding(
                patch_location=patch_location,
                cur_location=cur_location,
            )
            cur_location = patch_location

            rows = patch_location[:, 0].unsqueeze(1) + torch.arange(
                patch_size, device=device
            ).unsqueeze(0)
            cols = patch_location[:, 1].unsqueeze(1) + torch.arange(
                patch_size, device=device
            ).unsqueeze(0)

            rows = rows.clamp(0, img_canvas.shape[2] - patch_size)
            cols = cols.clamp(0, img_canvas.shape[3] - patch_size)

            patch = img_canvas[
                torch.arange(batch_size_current, device=device)
                .unsqueeze(1)
                .unsqueeze(2),
                :,
                rows.unsqueeze(2),
                cols.unsqueeze(1),
            ]  # (B, patch_size, patch_size, C)

            patch = patch.permute(0, 3, 1, 2).contiguous()  # (B, C, P, P)
            patch_flat = patch.flatten(start_dim=1)  # (B, C*P*P)

            patches_list.append(patch_flat)
            moves_list.append(move_embedding)

        img_sequence = torch.stack(patches_list, dim=1)
        move_sequence = torch.stack(moves_list, dim=1)

        classification_output = model(img_sequence, move_sequence)

        _, preds = torch.max(classification_output, dim=-1)

        for sequence_step in range(sequence_length):
            correct = torch.sum(preds[:, sequence_step] == label).item()
            acc_list[sequence_step] += correct

        total_sample += batch_size_current

        if batch_idx % 10 == 0:
            current_accuracy = (
                float(acc_list[-1]) / total_sample if total_sample > 0 else 0.0
            )
            print(f"Batch {batch_idx}: Current accuracy = {current_accuracy:.4f}")

    # Calculate final accuracies
    acc_list = [acc / total_sample for acc in acc_list]

    print(f"\nValidation completed!")
    print(f"Total samples: {total_sample}")
    print(f"Final accuracy at step {sequence_length}: {acc_list[-1]:.4f}")
    print(
        f"Best accuracy: {max(acc_list):.4f} at step {acc_list.index(max(acc_list)) + 1}"
    )

    # Save accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, sequence_length + 1), acc_list, "b-", linewidth=2)
    plt.xlabel("Sequence Step")
    plt.ylabel("Accuracy")
    plt.title(f"Validation Accuracy vs Sequence Step (Final: {acc_list[-1]:.4f})")
    plt.grid(True, alpha=0.3)
    plt.xlim(1, sequence_length)
    plt.ylim(0, 1)

    plt.axhline(
        y=max(acc_list),
        color="r",
        linestyle="--",
        alpha=0.7,
        label=f"Best: {max(acc_list):.4f}",
    )
    plt.axhline(
        y=acc_list[-1],
        color="g",
        linestyle="--",
        alpha=0.7,
        label=f"Final: {acc_list[-1]:.4f}",
    )
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        f"validation_accuracy_canvas{canvas_size}_scan{scan_pattern}_resize{resize_mode}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Accuracy plot saved as 'validation_accuracy.png'")

    results = {
        "sequence_length": sequence_length,
        "total_samples": total_sample,
        "accuracy_per_step": acc_list,
        "final_accuracy": acc_list[-1],
        "best_accuracy": max(acc_list),
        "best_step": acc_list.index(max(acc_list)) + 1,
    }

    torch.save(results, "validation_results.pt")
    print(f"Detailed results saved as 'validation_results.pt'")

    return acc_list[-1]


@hydra.main(version_base=None, config_path="configs", config_name="config")
@torch.inference_mode()
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Resolve model alias and config
    model_name = cfg.get("model_name", None)
    if model_name in MODEL_ALIASES:
        hf_repo, hf_checkpoint = MODEL_ALIASES[model_name]
        if model_name in MODEL_CONFIGS:
            config_name = MODEL_CONFIGS[model_name]
            model_cfg = OmegaConf.load(
                f"{hydra.utils.get_original_cwd()}/configs/model/{config_name}.yaml"
            )
            cfg.model = OmegaConf.merge(cfg.model, model_cfg)
            print(f"Auto-applied model config: {config_name}")
    else:
        hf_repo = cfg.get("hf_repo", "usingcolor/mambaeye")
        hf_checkpoint = cfg.get("hf_checkpoint", None)

    # Resolve and load checkpoint
    checkpoint_path = _resolve_checkpoint(cfg, model_name, hf_repo, hf_checkpoint)
    model = _load_model(cfg, checkpoint_path, device)

    # Route: single image inference vs dataset validation
    image_path = cfg.get("image_path", None)
    if image_path is not None:
        return _run_single_image_inference(cfg, model, device)
    else:
        return _run_dataset_validation(cfg, model, device)


if __name__ == "__main__":
    main()
