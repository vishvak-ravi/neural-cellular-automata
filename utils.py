import torch, torchvision
from torchvision.transforms.transforms import CenterCrop
import torch.nn.functional as F
from torch import nn
from torch.optim.adamw import AdamW

GEN_SIZE = (48, 48)

device = "cuda" if torch.cuda.is_available() else "cpu"


def init_board(img_path: str, init_val: float = None) -> torch.Tensor:
    # Load RGBA → float32 ∈ [0,1]
    img = (
        torchvision.io.read_image(
            img_path, mode=torchvision.io.image.ImageReadMode.RGB_ALPHA
        ).float()
        / 255.0
    )
    img = CenterCrop(GEN_SIZE)(img)
    # Binary alpha: 0 if fully transparent else 1
    img[3] = (img[3] > 0).float()
    img[:3] *= img[3].unsqueeze(0)  # multiply RGB channels by alpha mask

    target = img
    C, H, W = img.shape
    if init_val is not None:
        features = torch.full((16, H, W), init_val)
    else:
        features = torch.rand(16, H, W)

    features[3:, H // 2, W // 2] = 1.0  # set the seed

    return features, target


def get_perception(state_grid: torch.Tensor) -> torch.Tensor:
    """
    Applies Sobel filters to each channel of the input state_grid across a batch.
    """

    # Sobel kernels
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=state_grid.dtype,
        device=state_grid.device,
    ).to(device)
    sobel_y = sobel_x.t().to(device)

    # Prepare kernels for depthwise conv2d
    B, C, H, W = state_grid.shape
    sobel_x = sobel_x.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    sobel_y = sobel_y.view(1, 1, 3, 3).repeat(C, 1, 1, 1)

    # Add batch dimension and apply depthwise convolution
    state_grid = state_grid.to(device)

    x_grid = F.conv2d(state_grid, sobel_x, padding=1, groups=C)
    y_grid = F.conv2d(state_grid, sobel_y, padding=1, groups=C)

    # Reshape back to batch format
    x_grid = x_grid.view(B, C, H, W)
    y_grid = y_grid.view(B, C, H, W)

    state_grid = torch.cat((state_grid, x_grid, y_grid), dim=1)
    return state_grid


class CARule(torch.nn.Module):
    def __init__(self, dense: bool = True):
        super().__init__()
        self.dense = dense
        if dense:
            self.layer1 = nn.Linear(
                48,
                128,
            )
            self.layer2 = nn.Linear(
                128,
                16,
            )
        else:
            self.layer1 = nn.Conv2d(48, 128, 1)
            self.layer2 = nn.Conv2d(128, 16, 1)
        nn.init.constant_(self.layer2.weight, 0)
        nn.init.constant_(self.layer2.bias, 0)
        self.act1 = nn.ReLU()
        self.act2 = nn.Tanh()

    def forward(self, x):
        """
        x : B x C x H x W
        """
        x = get_perception(x)
        if self.dense:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.act2(x)

        if self.dense:
            x = x.reshape(B, H, W, -1)
            x = x.permute(0, 3, 1, 2)
        return x


def destroy(
    board: torch.Tensor, destroy_radius: int = 3, center: torch.Tensor = None
) -> torch.Tensor:
    assert center is not None and destroy_radius is not None
    assert destroy_radius % 2 == 1
    B, C, H, W = board.shape
    if center is not None:
        center = torch.randint(0, GEN_SIZE(0), size=(2))  # maybe normal?
    padded_board = F.pad(board, (destroy_radius, destroy_radius))
    padded_board[
        :,
        :,
        center[0] - destroy_radius : center[0] + destroy_radius,
        center[1] - destroy_radius : center[1] + destroy_radius,
    ] = 0
    return padded_board[
        :, :, destroy_radius : destroy_radius + H, destroy_radius : destroy_radius + W
    ]
