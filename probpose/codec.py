from itertools import product
import numpy as np
from torch import Tensor

from probpose.util import to_numpy

from probpose.heatmap import get_heatmap_expected_value


def generate_probmaps(
    heatmap_size: tuple[int, int],
    keypoints: np.ndarray,
    keypoints_visible: np.ndarray,
    sigmas: np.ndarray,
    sigma: float = 0.55,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Object Keypoint Similarity (OKS) maps for keypoints.

    This function generates OKS maps that represent the expected OKS score at each
    pixel location given the ground truth keypoint locations. The concept was
    introduced in `ProbPose`_ to enable probabilistic keypoint detection.

    Args:
        heatmap_size (Tuple[int, int]): Heatmap size in [W, H]
        keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
        keypoints_visible (np.ndarray): Keypoint visibilities in shape
            (N, K)
        sigma (float): The sigma value to control the spread of the OKS map.
            If None, per-keypoint sigmas from COCO will be used. Default: 0.55

    Returns:
        tuple:
        - heatmaps (np.ndarray): The generated OKS maps in shape
            (K, H, W) where [W, H] is the `heatmap_size`
        - keypoint_weights (np.ndarray): The target weights in shape
            (N, K)

    .. _`ProbPose`: https://arxiv.org/abs/2412.02254
    """

    N, K, _ = keypoints.shape
    W, H = heatmap_size

    heatmaps = np.zeros((K, H, W), dtype=np.float32)
    keypoint_weights = keypoints_visible.copy()

    bbox_area = np.sqrt(H / 1.25 * W / 1.25)

    for n, k in product(range(N), range(K)):
        kpt_sigma = sigmas[k]
        # skip unlabled keypoints
        if keypoints_visible[n, k] < 0.5:
            continue

        y_idx, x_idx = np.indices((H, W))
        dx = x_idx - keypoints[n, k, 0]
        dy = y_idx - keypoints[n, k, 1]
        dist = np.sqrt(dx**2 + dy**2)

        vars = (kpt_sigma * 2) ** 2
        s = vars * bbox_area * 2
        s = np.clip(s, 0.55, 3.0)
        if sigma is not None and sigma > 0:
            s = sigma
        e_map = dist**2 / (2 * s)
        oks_map = np.exp(-e_map)

        keypoint_weights[n, k] = (oks_map.max() > 0).astype(int)

        heatmaps[k] = oks_map
    return heatmaps, keypoint_weights


class ProbMap:
    r"""Generate per-pixel expected OKS heatmaps for keypoint detection.
    See the paper: `ProbPose: A Probabilistic Approach to 2D Human Pose Estimation`
    by Purkrabek et al. (2025) for details.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - image size: [w, h]
        - heatmap size: [W, H]

    Encoded:

        - heatmap (np.ndarray): The generated OKS heatmap in shape (K, H, W)
            where [W, H] is the `heatmap_size`, and K is the number of keypoints.
            Each pixel value represents the expected OKS score if that pixel is
            predicted as the keypoint location, given the ground truth location.
        - keypoint_weights (np.ndarray): The target weights in shape (K,)

    Args:
        input_size (tuple): Image size in [w, h]
        heatmap_size (tuple): Heatmap size in [W, H]
        heatmap_type (str): The heatmap type to encode the keypoints. Options
            are:

            - ``'gaussian'``: Gaussian heatmap
            - ``'combined'``: Combination of a binary label map and offset
                maps for X and Y axes.

        sigma (float): The sigma value of the Gaussian heatmap when
            ``heatmap_type=='gaussian'``. Defaults to 2.0
        radius_factor (float): The radius factor of the binary label
            map when ``heatmap_type=='combined'``. The positive region is
            defined as the neighbor of the keypoint with the radius
            :math:`r=radius_factor*max(W, H)`. Defaults to 0.0546875
        blur_kernel_size (int): The Gaussian blur kernel size of the heatmap
            modulation in DarkPose. Defaults to 11

    .. _`ProbPose: A Probabilistic Approach to 2D Human Pose Estimation`:
        https://arxiv.org/abs/2412.02254
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        heatmap_size: tuple[int, int],
        sigmas: np.ndarray,
        sigma: float = 2.0,
        radius_factor: float = 0.0546875,
        blur_kernel_size: int = 11,
        increase_sigma_with_padding=False,
    ) -> None:
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.radius_factor = radius_factor
        self.blur_kernel_size = blur_kernel_size
        self.scale_factor = (
            (np.array(input_size) - 1) / (np.array(heatmap_size) - 1)
        ).astype(np.float32)
        self.increase_sigma_with_padding = increase_sigma_with_padding
        self.sigmas = sigmas
        self.sigma = sigma

    def encode(
        self,
        keypoints: np.ndarray,
        keypoints_visible: np.ndarray | None = None,
        id_similarity: float | None = 0.0,
        keypoints_visibility: np.ndarray | None = None,
    ) -> dict:
        """Encode keypoints into heatmaps. Note that the original keypoint
        coordinates should be in the input image space.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)
            id_similarity (float): The usefulness of the identity information
                for the whole pose. Defaults to 0.0
            keypoints_visibility (np.ndarray): The visibility bit for each
                keypoint (N, K). Defaults to None

        Returns:
            dict:
            - heatmap (np.ndarray): The generated heatmap in shape
                (C_out, H, W) where [W, H] is the `heatmap_size`, and the
                C_out is the output channel number which depends on the
                `heatmap_type`. If `heatmap_type=='gaussian'`, C_out equals to
                keypoint number K; if `heatmap_type=='combined'`, C_out
                equals to K*3 (x_offset, y_offset and class label)
            - keypoint_weights (np.ndarray): The target weights in shape
                (K,)
        """
        assert keypoints.shape[0] == 1, (
            f"{self.__class__.__name__} only support single-instance keypoint encoding"
        )

        if keypoints_visibility is None:
            keypoints_visibility = np.zeros(keypoints.shape[:2], dtype=np.float32)

        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

        heatmaps, keypoint_weights = generate_probmaps(
            heatmap_size=self.heatmap_size,
            keypoints=keypoints / self.scale_factor,
            keypoints_visible=keypoints_visible,
            keypoints_visibility=keypoints_visibility,
            sigmas=self.sigmas,
            sigma=self.sigma,
            increase_sigma_with_padding=self.increase_sigma_with_padding,
        )

        print(f"{heatmaps.shape=}")

        annotated = keypoints_visible > 0

        in_image = np.logical_and(
            keypoints[:, :, 0] >= 0,
            keypoints[:, :, 0] < self.input_size[0],
        )
        in_image = np.logical_and(
            in_image,
            keypoints[:, :, 1] >= 0,
        )
        in_image = np.logical_and(
            in_image,
            keypoints[:, :, 1] < self.input_size[1],
        )

        encoded = dict(
            heatmaps=heatmaps,
            keypoint_weights=keypoint_weights,
            annotated=annotated,
            in_image=in_image,
            keypoints_scaled=keypoints,
            heatmap_keypoints=keypoints / self.scale_factor,
            identification_similarity=id_similarity,
        )

        return encoded

    def decode(self, encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decode keypoint coordinates from heatmaps. The decoded keypoint
        coordinates are in the input image space.

        Args:
            encoded (np.ndarray): Heatmaps in shape (K, H, W)

        Returns:
            tuple:
            - keypoints (np.ndarray): Decoded keypoint coordinates in shape
                (N, K, D)
            - scores (np.ndarray): The keypoint scores in shape (N, K). It
                usually represents the confidence of the keypoint prediction
        """
        heatmaps = encoded.copy()
        W, H = self.heatmap_size

        keypoints, scores = get_heatmap_expected_value(heatmaps, self.sigmas)

        # unsqueeze the instance dimension for single-instance results
        keypoints = keypoints[None]
        scores = scores[None]

        keypoints = keypoints / [W - 1, H - 1] * self.input_size

        return keypoints, scores


class Codec:
    def __init__(
        self,
        input_size: tuple[int, int],
        heatmap_size: tuple[int, int],
        sigmas: np.ndarray,
    ):
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.sigmas = sigmas
        self.sigma = 2.0
        self.probmap = ProbMap(
            input_size=input_size,
            heatmap_size=heatmap_size,
            sigmas=sigmas,
        )

    def decode(self, pred: tuple[Tensor, Tensor, Tensor, Tensor, Tensor]):
        heatmaps, probabilities, visibilities, oks, errors = pred
        B, C, H, W = heatmaps.shape

        preds = self.probmap.decode(to_numpy(heatmaps))
        # probabilities = to_numpy(probabilities).reshape((B, 1, C))
        # visibilities = to_numpy(visibilities).reshape((B, 1, C))
        # oks = to_numpy(oks).reshape((B, 1, C))
        # errors = to_numpy(errors).reshape((B, 1, C))

        # # Normalize errors by dividing with the diagonal of the heatmap
        # htm_diagonal = np.sqrt(H**2 + W**2)
        # errors = errors / htm_diagonal

        return preds

    def decode_heatmap(self, heatmaps: Tensor):
        preds = self.probmap.decode(to_numpy(heatmaps))
        return preds

    def encode(
        self,
        keypoints: np.ndarray,
        keypoints_visible: np.ndarray | None = None,
        id_similarity: float | None = 0.0,
    ) -> dict:
        """Encode keypoints into heatmaps. Note that the original keypoint
        coordinates should be in the input image space.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)
            id_similarity (float): The usefulness of the identity information
                for the whole pose. Defaults to 0.0
            keypoints_visibility (np.ndarray): The visibility bit for each
                keypoint (N, K). Defaults to None

        Returns:
            dict:
            - heatmap (np.ndarray): The generated heatmap in shape
                (C_out, H, W) where [W, H] is the `heatmap_size`, and the
                C_out is the output channel number which depends on the
                `heatmap_type`. If `heatmap_type=='gaussian'`, C_out equals to
                keypoint number K; if `heatmap_type=='combined'`, C_out
                equals to K*3 (x_offset, y_offset and class label)
            - keypoint_weights (np.ndarray): The target weights in shape
                (K,)
        """
        assert keypoints.shape[0] == 1, (
            f"{self.__class__.__name__} only support single-instance keypoint encoding"
        )

        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

        heatmaps, keypoint_weights = generate_probmaps(
            heatmap_size=self.heatmap_size,
            keypoints=keypoints,
            keypoints_visible=keypoints_visible,
            sigmas=self.sigmas,
            sigma=self.sigma,
        )

        annotated = keypoints_visible > 0

        in_image = np.logical_and(
            keypoints[:, :, 0] >= 0,
            keypoints[:, :, 0] < self.input_size[0],
        )
        in_image = np.logical_and(
            in_image,
            keypoints[:, :, 1] >= 0,
        )
        in_image = np.logical_and(
            in_image,
            keypoints[:, :, 1] < self.input_size[1],
        )

        encoded = dict(
            heatmaps=heatmaps,
            keypoint_weights=keypoint_weights,
            annotated=annotated,
            in_image=in_image,
            keypoints_scaled=keypoints,
            heatmap_keypoints=keypoints,
            identification_similarity=id_similarity,
        )

        return heatmaps, in_image
