import torch, torchvision
from torchvision.transforms.transforms import CenterCrop
import torch.nn.functional as F
from torch import nn
from torch.optim.adamw import AdamW

STEPS = 100000
EPS = 0.5


def init_board(img_path: str, init_val: float = None) -> torch.Tensor:
    # Load RGBA → float32 ∈ [0,1]
    img = (
        torchvision.io.read_image(
            img_path, mode=torchvision.io.image.ImageReadMode.RGB_ALPHA
        ).float()
        / 255.0
    )
    img = CenterCrop((48, 48))(img)
    target = img
    # Binary alpha: 0 if fully transparent else 1
    img[3] = (img[3] > 0).float()
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
    )
    sobel_y = sobel_x.t()

    # Prepare kernels for depthwise conv2d
    B, C, H, W = state_grid.shape
    sobel_x = sobel_x.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    sobel_y = sobel_y.view(1, 1, 3, 3).repeat(C, 1, 1, 1)

    # Add batch dimension and apply depthwise convolution
    x_grid = F.conv2d(state_grid, sobel_x, padding=1, groups=C)
    y_grid = F.conv2d(state_grid, sobel_y, padding=1, groups=C)

    # Reshape back to batch format
    x_grid = x_grid.view(B, C, H, W)
    y_grid = y_grid.view(B, C, H, W)

    state_grid = torch.cat((state_grid, x_grid, y_grid), dim=1)
    return state_grid


class CARule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Linear(48, 128)
        self.conv2 = nn.Linear(128, 16)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x


def update_board(perception: torch.Tensor, update_rule: CARule) -> torch.Tensor:
    B, _, H, W = perception.shape
    rand_mask = (
        torch.abs(torch.rand(B, H, W, device=perception.device)) > EPS
    )  # boolean mask to selectively update cells
    features_to_update = perception.permute(0, 2, 3, 1)[rand_mask]  # (N, C)
    feature_step = update_rule(features_to_update)  # (N, 16)
    rt = torch.zeros(B, 16, H, W, device=perception.device)
    rt.permute(0, 2, 3, 1)[rand_mask] = feature_step
    return rt


def train(img_path: str, update_rule: CARule, bs: int = 8):
    board, target = init_board(img_path, 0.0)
    boards = torch.stack([board.clone() for _ in range(bs)], dim=0)
    boards.requires_grad_(True)
    target = target[:3, :, :]
    torchvision.utils.save_image(target, f"a_target.png")
    opt = AdamW(update_rule.parameters())
    loss_fn = torch.nn.MSELoss()

    steps_till_opt = int((64 + (96 - 64) * torch.rand(1)).item())
    for steps in range(STEPS):
        # true_proportion = alive_mask.float().mean().item()
        # print(f"Proportion of True in alive_mask: {true_proportion}")
        perception = get_perception(boards)
        dboard = update_board(perception, update_rule)
        alive_mask = (
            (F.max_pool2d(boards[:, 3:4, :, :], 3, stride=1, padding=1) > 0.1)
            .int()[:, 0]
            .unsqueeze(1)
        )
        boards = boards + dboard * alive_mask
        boards[:, :3].clamp_(0.0, 1.0)
        steps_till_opt -= 1

        if steps_till_opt == 0:
            steps_till_opt = int((64 + (96 - 64) * torch.rand(1)).item())
            board_imgs = boards[
                :, :3, :, :
            ]  # Extract RGB channels for the entire batch
            loss_val = loss_fn(
                target.unsqueeze(0).expand_as(board_imgs), board_imgs
            )  # Expand target to match batch size
            loss_val.backward()
            opt.step()
            opt.zero_grad()
            board, _ = init_board("data/mudkip.png", 0.0)
            boards = torch.stack([board.clone() for _ in range(bs)], dim=0)
            board.requires_grad_(True)
            torchvision.utils.save_image(board_imgs[0], f"board_img_step_{steps}.png")
            print(loss_val)


if __name__ == "__main__":
    update_rule = CARule()
    train("data/mudkip.png", update_rule=update_rule)
