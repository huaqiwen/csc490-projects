import math
from typing import Tuple

import torch
from torch import Tensor

from detection.modules.loss_function import DetectionLossConfig
from detection.types import Detections

def rotation_matrix(yaw: float) -> Tensor:
    """Generate a 2D rotation matrix given the yaw."""
    c = math.cos(yaw)
    s = math.sin(yaw)

    rotation = torch.tensor([
        [c, -s],
        [s, c]
    ])

    return rotation

def scale_matrix(x_size: float, y_size: float) -> Tensor:
    scaled = torch.zeros((2, 2))
    scaled[0,0] = x_size
    scaled[1,1] = y_size
    scaled /= 4.0

    return scaled

def isotropic_heatmap(grid_coords: Tensor, center: Tensor, scale: float, size: Tuple, yaw: float) -> Tensor:
    H, W, _ = grid_coords.shape
    cx, cy = center

    # Resize grid_coords to be [(H * W) x 2] for easier calculation
    temp = grid_coords.reshape(H * W, 2)

    # Calculate raw heat with the input Gaussian kernel
    heatmap = torch.exp(-((cx - temp[:, 0]) ** 2 + (cy - temp[:, 1]) ** 2) / scale)

    # Normalize and reshape back to [H x W]
    heatmap /= heatmap.max()
    heatmap.resize_(H, W)

    return heatmap

def anisotropic_heatmap(grid_coords: Tensor, center: Tensor, scale: float, size: Tuple, yaw: float) -> Tensor:
    H, W, _ = grid_coords.shape
    cx, cy = center

    # Resize grid_coords to be [(H * W) x 2] for easier calculation
    stack = grid_coords.reshape(H * W, 2).float()

    # Create the rotation and scaling matricies which form the covariance matrix
    R = rotation_matrix(yaw)
    S = scale_matrix(size[0], size[1])
    
    # Set up the mean and covariance
    mean = torch.tensor([[cx, cy]])
    cov = R.matmul(S.matmul(S.matmul(R.T)))

    # Compute the determinant and the covariance inverse
    cov_det = torch.det(cov).item()
    cov_inv = torch.inverse(cov)

    # temp after mean shift
    stack -= mean

    # Compute the gaussian pdf
    exponent = -0.5 * cov_inv.matmul(stack.T).T * stack
    exponent = torch.sum(exponent, axis=1)
    factor = 1.0 / (abs(2 * math.pi) * math.sqrt(cov_det))

    heatmap_pdf = factor * torch.exp(exponent)

    # Normalize and reshape back to [H x W]
    heatmap_pdf /= heatmap_pdf.max()
    heatmap_pdf.resize_(H, W)

    return heatmap_pdf

def create_heatmap(grid_coords: Tensor, center: Tensor, scale: float, size: Tuple, yaw: float) -> Tensor:
    """Return a heatmap based on a Gaussian kernel with center `center` and scale `scale`.

    Specifically, each pixel with coordinates (x, y) is assigned a heatmap value
    using a Gaussian kernel centered on (cx, cy) with scale s:

                e^(-((x - cx)^2 + (y - cy)^2) / s)

    Subsequently, the heatmap is normalized such that its maximum value is 1.

    Args:
        grid_coords: An [H x W x 2] tensor containing the (x, y) coordinates of every
            pixel in an [H x W] image. For example, for a [2 x 3] image, `grid_coords`
            contains the elements (0, 0), (0, 1), (0, 2), ..., (1, 2).
        center: A [2] tensor containing the (x, y) coordinate of the center.
            This argument controls the kernel's center.
        scale: A scalar value that controls the kernel's scale.
        size: (x_size, y_size) pair denoting size of label
        yaw: The rotation angle of the label
    Returns:
        An [H x W] heatmap tensor, normalized such that its peak is 1.
    """
    
    use_isotropic_guassian = True
    if use_isotropic_guassian:
        return isotropic_heatmap(grid_coords, center, scale, size, yaw)
    else:
        return anisotropic_heatmap(grid_coords, center, scale, size, yaw)


class DetectionLossTargetBuilder:
    """Builds the target tensors for training using the `DetectionLoss`."""

    def __init__(
        self, bev_size: Tuple[int, int, int], config: DetectionLossConfig
    ) -> None:
        """Initialization.

        Args:
            bev_size: The [depth, height, width] of the bird's eye view voxel grid.
                depth is the size along the z-axis, height is along the y-axis, and
                width is along the x-axis.
            config: The detection loss configuration.
        """
        self._bev_size = bev_size
        self._heatmap_threshold = config.heatmap_threshold
        self._heatmap_norm_scale = config.heatmap_norm_scale

    def build_target_tensor_for_label(
        self, cx: float, cy: float, yaw: float, x_size: float, y_size: float
    ) -> Tensor:
        """Return the training target tensor for the given bounding box.

        This method computes a 7-dimension vector for each pixel (i, j) in the
        [H x W] BEV image. In order, the 7 channels contain:
        1. heatmap: The heatmap score for (i, j); 1 if (i, j) is the closest pixel
            to a labels's center (cx, cy) and otherwise decreasing by distance.
        2. offset_x: The offset between the labels's x-axis center cx and the
            pixel coordinate i; i.e. offset_x = cx - i.
        3. offset_y: The offset between the labels's y-axis center cy and the
            pixel coordinate j; i.e. offset_y = cy - j.
        4. x_size: The size of the labels's bounding box along the x-axis.
        5. y_size: The size of the labels's bounding box along the y-axis.
        6. sin_theta: sin_theta = sin(yaw), where yaw is the heading of the
            label in radians. To decode yaw, yaw = atan2(sin_theta, cos_theta).
        7. cos_theta: cos_theta = cos(yaw), where yaw is the heading of the
            label in radians. To decode yaw, yaw = atan2(sin_theta, cos_theta).

        Args:
            cx: The x-axis center of the label, in BEV image coordinates.
            cy: The y-axis center of the label, in BEV image coordinates.
            yaw: The yaw of the label, in radians and in BEV image coordinates.
            x_size: The x-axis size of the label, in BEV image coordinates.
            y_size: The y-axis size of the label, in BEV image coordinates.

        Returns:
            A [7 x H x W] tensor representing the training target for one bounding box,
                where H and W are the height and width of the BEV image respectively. The 7
                channels are (heatmap, offset_x, offset_y, x_size, y_size, sin_theta, cos_theta).
        """
        _, H, W = self._bev_size

        # 1. Build a [H x W x 2] tensor of BEV pixel coordinates.
        # For example, for a 2 x 3 image, `grid_coords` is a [2 x 3 x 2] tensor
        # containing the elements (0, 0), (0, 1), (0, 2), ..., (1, 2).
        W_coords, H_coords = torch.arange(W), torch.arange(H)
        H_grid_coords, W_grid_coords = torch.meshgrid(H_coords, W_coords, indexing="ij")
        grid_coords = torch.stack([W_grid_coords, H_grid_coords], dim=-1)  # [H x W x 2]

        # 2. Create heatmap training targets by invoking the `create_heatmap` function.
        center = torch.tensor([cx, cy])
        scale = (x_size ** 2 + y_size ** 2) / self._heatmap_norm_scale
        heatmap = create_heatmap(grid_coords, center=center, scale=scale, size=(x_size, y_size), yaw=yaw)  # [H x W]

        # 3. Create offset training targets.
        # Given the label's center (cx, cy), the target offset at pixel (i, j) equals
        # (cx - i, cy - j) if the heatmap value at (i, j) exceeds self._heatmap_threshold.
        # If the heatmap value at (i, j) is less than or equal to self._heatmap_threshold,
        # the target offset equals (0, 0) instead.

        # Generate the thresholded heatmap mask tensor for future reference 
        heatmap_mask = heatmap <= self._heatmap_threshold

        # Calculate target offset and mask off points where the heatmap value is below threshold
        offsets = (center - grid_coords).long()
        offsets[heatmap_mask] = torch.tensor([0, 0])

        # 4. Create box size training target.
        # Given the label's bounding box size (x_size, y_size), the target size at pixel (i, j)
        # equals (x_size, y_size) if the heatmap value at (i, j) exceeds self._heatmap_threshold.
        # If the heatmap value at (i, j) is less than or equal to self._heatmap_threshold,
        # the target size equals (0, 0) instead.
        sizes = torch.zeros_like(grid_coords)
        sizes[:, :] = torch.tensor([x_size, y_size])
        sizes[heatmap_mask] = torch.tensor([0, 0])

        # 5. Create heading training targets.
        # Given the label's heading angle yaw, the target heading at pixel (i, j)
        # equals (sin(yaw), cos(yaw)) if the heatmap value at (i, j) exceeds self._heatmap_threshold.
        # If the heatmap value at (i, j) is less than or equal to self._heatmap_threshold,
        # the target heading equals (0, 0) instead.
        headings = torch.zeros(H, W, 2)
        headings[:, :] = torch.tensor([math.sin(yaw), math.cos(yaw)])
        headings[heatmap_mask] = torch.tensor([0., 0.])

        # 6. Concatenate training targets into a [7 x H x W] tensor.
        targets = torch.cat([heatmap[:, :, None], offsets, sizes, headings], dim=-1)
        return targets.permute(2, 0, 1)  # [7 x H x W]


    def build_target_tensor(self, labels: Detections) -> Tensor:
        """Return the training target tensor for the given labels.

        Args:
            labels: A set of ground truth detections.

        Returns:
            A [7 x H x W] float32 tensor representing the training target for one frame of labels,
                where H and W are the height and width of the BEV image respectively. The 7
                channels are (heatmap, offset_x, offset_y, x_size, y_size, sin_theta, cos_theta).
        """
        _, H, W = self._bev_size

        # 1. Build a list of N [7 x H x W] target tensors for each of the N labels.
        target_tensors = []
        for index in range(len(labels)):
            target_tensor_for_label = self.build_target_tensor_for_label(
                labels.centroids_x[index].item(),
                labels.centroids_y[index].item(),
                labels.yaws[index].item(),
                labels.boxes_x[index].item(),
                labels.boxes_y[index].item(),
            )
            target_tensors.append(target_tensor_for_label)

        # 2. Combine the target tensors into a single [7 x H x W] target tensor.
        # For each pixel (i, j) of the aggregated tensor, we keep the maximum heatmap
        # value from the N labels. We also keep the other targets (i.e. offset, size, heading)
        # corresponding to the same winning label.
        target_tensor = torch.stack(target_tensors, dim=0)  # [N x 7 x H x W]
        heatmaps = target_tensor[:, 0]  # [N x H x W]
        heatmap_max_value, heatmap_argmax_id = heatmaps.max(dim=0)  # [H x W], [H x W]
        gridh, gridw = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        target_tensor_max = target_tensor[heatmap_argmax_id, :, gridh, gridw]
        assert torch.all(heatmap_max_value == target_tensor_max[:, :, 0])
        return target_tensor_max.permute(2, 0, 1)  # [7 x H x W]
