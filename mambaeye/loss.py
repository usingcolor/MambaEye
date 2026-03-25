import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaEyeLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        classification_sequence: torch.Tensor,
        gt_label: torch.Tensor,
        information_ratio: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            classification_sequence: (B, L, D) logits per step
            gt_label:                (B,)       single label for the whole sequence
            information_ratio:       (L,) or (B, L) with values in [0,1]; if None, uses linspace(0..1)
            
        Returns:
            loss: scalar tensor representing the computed loss
        """
        B, L, D = classification_sequence.shape
        device = classification_sequence.device
        dtype = classification_sequence.dtype

        # ---- labels validation ----
        assert (
            gt_label.min().item() >= 0 and gt_label.max().item() < self.num_classes
        ), (
            f"label should be in [0, {self.num_classes - 1}], but got {gt_label.min()} and {gt_label.max()}"
        )

        if information_ratio is None:
            t = torch.ones(B, L, device=device, dtype=dtype)
        else:
            t = information_ratio
            if t.dim() == 1:
                t = t.unsqueeze(0).expand(B, L)
            t = t.to(device=device, dtype=dtype).clamp_(0, 1)

        # Prior: uniform probabilities
        prior_logit = torch.full((B, D), 1.0 / D, device=device, dtype=dtype)  # (B, D)

        # Target: one-hot encoded ground truth
        target_logit = F.one_hot(gt_label, num_classes=D).to(dtype)  # (B, D)

        # Noisy logits: convex combination of target and prior
        t_expanded = t.unsqueeze(-1)  # (B, L, 1)
        scheduled_logit = (
            t_expanded * target_logit.unsqueeze(1)
            + (1 - t_expanded) * prior_logit.unsqueeze(1)
        )  # (B, L, D)

        loss = (
            self.ce_loss(
                classification_sequence.view(-1, D), scheduled_logit.view(-1, D)
            )
            .view(B, L)
            .mean()
        )

        return loss
