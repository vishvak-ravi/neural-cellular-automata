import torch, torchvision
from torchvision.transforms.transforms import Pad
import torch.nn.functional as F
from torch import nn
from torch.optim.adamw import AdamW

DEF_STATE_SIZE = 16
PAD_AMT = 12
EPS = 0.5

device = "cuda" if torch.cuda.is_available() else "cpu"


def init_board(img_path: str, state_size: int = DEF_STATE_SIZE) -> torch.Tensor:
    # Load RGBA → float32 ∈ [0,1]
    img = (
        torchvision.io.read_image(
            img_path, mode=torchvision.io.image.ImageReadMode.RGB_ALPHA
        ).float()
        / 255.0
    )
    img = Pad(PAD_AMT)(img)
    # Binary alpha: 0 if fully transparent else 1
    img[3] = (img[3] > 0).float()
    img[:3] *= img[3].unsqueeze(0)  # multiply RGB channels by alpha mask

    target = img
    C, H, W = img.shape
    features = torch.zeros((state_size, H, W))
    features[3, H // 2, W // 2] = 1.0  # set the seed

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


class CAUpdate(torch.nn.Module):
    def __init__(self, state_size=DEF_STATE_SIZE, learned_features=False):
        super().__init__()
        self.learned_features = learned_features
        if learned_features:
            self.layer0 = nn.Conv2d(state_size * 3, state_size * 4, 3, padding=1)
            self.layer1 = nn.Conv2d(state_size * 4, state_size * 8, 1)
        else:
            self.layer1 = nn.Conv2d(state_size * 3, state_size * 8, 1)
        self.layer2 = nn.Conv2d(state_size * 8, state_size, 1)
        nn.init.constant_(self.layer2.weight, 0)
        nn.init.constant_(self.layer2.bias, 0)
        self.act1 = nn.ReLU()
        self.act2 = nn.Tanh()

    def forward(self, x):
        """
        x : B x n x H x W
        """
        if self.learned_features:
            x = self.layer0(x)
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.act2(x)
        return x


class CAGetBoard(torch.nn.Module):
    def __init__(self, state_size=DEF_STATE_SIZE, learned_features=False):
        super().__init__()
        self.model = CAUpdate(state_size=state_size, learned_features=learned_features)

    def forward(self, x):
        """
        x : B x n x H x W

        n = RGBA + hidden states
        """
        boards = x
        perception = get_perception(boards)
        dboard = self.model(perception)

        B, C, H, W = dboard.shape
        pre_alive = (
            (F.max_pool2d(boards[:, 3:4, :, :], 3, stride=1, padding=1) > 0.1)
            .int()[:, 0]
            .unsqueeze(1)
        ).to(device)
        dboard = dboard * (
            torch.rand(B, 1, H, W, device=device) < EPS
        )  # only some cells update
        boards = boards + dboard
        post_alive = (
            (F.max_pool2d(boards[:, 3:4, :, :], 3, stride=1, padding=1) > 0.1)
            .int()[:, 0]
            .unsqueeze(1)
        ).to(device)
        boards = boards * (
            pre_alive & post_alive
        )  # only update cells that were alive both before and after the update
        boards[:, :3].clamp_(0.0, 1.0)
        return boards


destroy_masks = {}  # radius → (K,2) offsets


def _offsets(r: int, device):
    if r not in destroy_masks:
        pts = [
            (dy, dx)
            for dy in range(-r, r + 1)
            for dx in range(-r, r + 1)
            if dy * dy + dx * dx < r * r
        ]
        destroy_masks[r] = torch.tensor(pts, dtype=torch.long, device=device)
    return destroy_masks[r]


def destroy(
    board: torch.Tensor,
    center: torch.Tensor,  # (B,2)  [y,x]
    radius: torch.Tensor,
) -> torch.Tensor:
    """
    `destroy_radius` can be int or (B,) tensor of ints (per-sample radii).
    """

    B, C, H, W = board.shape
    destroy_radius = radius * ((H * W) ** 0.5)
    pad = int(destroy_radius.max())
    padded = F.pad(board, (pad, pad, pad, pad))

    for r in destroy_radius.unique():
        idx = (destroy_radius == r).nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        off = _offsets(int(r), board.device)  # (K,2)
        coords = center[idx].long().unsqueeze(1) + off  # (B_r,K,2)
        coords += pad
        y, x = coords[..., 0], coords[..., 1]  # (B_r,K)
        b_idx = idx.view(-1, 1).expand(-1, off.size(0))
        padded[b_idx, :, y, x] = 0

    return padded[:, :, pad : pad + H, pad : pad + W]


def to_onnx(torch_model: CAGetBoard, save_name: str, img_shape: tuple):
    example_input = torch.zeros(img_shape)

    # quantize + prune and retrain later
    onxx_program = torch.onnx.export(torch_model, example_input, dynamo=True)
    onxx_program.save(f"docs/models/{save_name}.onnx")
