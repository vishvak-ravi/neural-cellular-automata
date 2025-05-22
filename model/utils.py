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
    def __init__(self, state_size=DEF_STATE_SIZE):
        super().__init__()
        self.layer1 = nn.Conv2d(state_size * 3, 128, 1)
        self.layer2 = nn.Conv2d(128, state_size, 1)
        nn.init.constant_(self.layer2.weight, 0)
        nn.init.constant_(self.layer2.bias, 0)
        self.act1 = nn.ReLU()
        self.act2 = nn.Tanh()

    def forward(self, x):
        """
        x : B x n x H x W
        """
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.act2(x)
        return x


class CAGetBoard(torch.nn.Module):
    def __init__(self, state_size=DEF_STATE_SIZE):
        super().__init__()
        self.model = CAUpdate(state_size=state_size)

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
    
    
destroy_masks = {}

def destroy(board: torch.Tensor,
            center: torch.Tensor,          # (B, 2) – [y, x] per sample
            destroy_radius: int = 3) -> torch.Tensor:
    B, C, H, W = board.shape
    pad = destroy_radius
    padded = F.pad(board, (pad, pad, pad, pad))           # left, right, top, bottom

    # pre-compute offset list for this radius
    if destroy_radius not in destroy_masks:
        offs = [(dy, dx)
                for dy in range(-pad, pad + 1)
                for dx in range(-pad, pad + 1)
                if dy*dy + dx*dx < pad*pad]
        destroy_masks[destroy_radius] = torch.as_tensor(
            offs, dtype=torch.long, device=board.device)  # (K, 2)

    offsets = destroy_masks[destroy_radius]               # (K, 2)
    K = offsets.size(0)

    # absolute coords in padded tensor
    coords = center.long().unsqueeze(1) + offsets.unsqueeze(0) + pad  # (B, K, 2)
    y, x = coords[..., 0], coords[..., 1]                             # (B, K)

    b = torch.arange(B, device=board.device).view(B, 1).expand(-1, K) # (B, K)

    # zero-out the selected pixels across all channels
    padded[b, :, y, x] = 0

    return padded[:, :, pad:pad+H, pad:pad+W]
  
def to_onnx(torch_model: CAGetBoard, save_name: str, img_shape: tuple):
    example_input = torch.ones(
        1, DEF_STATE_SIZE, img_shape[0] + PAD_AMT, img_shape[1] + PAD_AMT
    )

    # quantize + prune and retrain later
    onxx_program = torch.onnx.export(torch_model, example_input, dynamo=True)
    onxx_program.optimize()
    onxx_program.save(f"data/params/{save_name}.onnx")
