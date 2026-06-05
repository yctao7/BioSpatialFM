"""DINO-style cross-entropy loss for the channel-masking consistency task (Lmask).

Replaces the BYOL-style cosine alignment with the iBOT-style formulation:
  - Teacher view: full marker panel, head-projected CLS, centered + sharpened by
    teacher_temp via softmax to produce a target distribution over the head's
    high-dim prototype space (default 65536).
  - Student view(s): masked panel, head-projected CLS, log-softmax with
    student_temp.
  - Loss: cross-entropy between teacher target and each student log-distribution,
    averaged across student views (supports K views per teacher view -- K-to-1
    pairing -- for stronger gradient signal when the masking task is otherwise too
    easy).
  - A separate `center` EMA buffer is maintained on this module (independent of
    the main DINOLoss center) and updated from the teacher's outputs each step,
    DDP-synced if a process group is initialized. This prevents collapse without
    relying on KoLeo for the mask branch.

Pairing:
  Each sample's K student views match the SAME sample's single teacher view
  (1-to-1 within each (sample, view-k) pair). No cross-sample mixing -- mask
  consistency is per-sample by definition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class MaskCEloss(nn.Module):
    """
    Cross-entropy alignment of K student-masked CLS distributions to a single
    teacher-full-panel CLS distribution, with EMA centering for collapse control.

    Args:
        out_dim: Output dimensionality of the DINO head (matches DINOLoss out_dim).
        teacher_temp: Temperature for the teacher distribution (sharpening).
        student_temp: Temperature for the student distribution.
        center_momentum: EMA momentum for the center buffer.
    """
    def __init__(
        self,
        out_dim: int,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_outputs, teacher_output):
        """
        Args:
            student_outputs: list of [N, out_dim] tensors, one per student view.
                Each tensor is the head-projected CLS for the masked-channel
                forward of every sample in the (concatenated) batch. The list
                length is K (number of student views).
            teacher_output: [N, out_dim] tensor of head-projected CLS for the
                teacher's full-panel forward of every sample.

        N here is the total number of samples across all panel groups (concatenated
        in the same order across teacher and student forwards), so cross-entropy
        is computed per-sample, then averaged.

        Returns:
            scalar loss (averaged over K views and N samples).
        """
        # Teacher target distribution (no grad).
        with torch.no_grad():
            t_logits = (teacher_output - self.center) / self.teacher_temp
            t_dist = F.softmax(t_logits, dim=-1)

        # Average CE across K student views.
        total = 0.0
        for s_out in student_outputs:
            s_log = F.log_softmax(s_out / self.student_temp, dim=-1)
            total = total + -(t_dist * s_log).sum(dim=-1).mean()
        loss = total / max(1, len(student_outputs))

        # Update center from teacher output.
        self.update_center(teacher_output)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)  # [1, out_dim]
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            n = teacher_output.shape[0] * dist.get_world_size()
        else:
            n = teacher_output.shape[0]
        batch_center = batch_center / max(1, n)
        self.center.mul_(self.center_momentum).add_(batch_center * (1 - self.center_momentum))
