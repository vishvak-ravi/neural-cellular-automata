import torch, torchvision
import torch.nn.functional as F
from torch.optim.adamw import AdamW

from enum import Enum

from utils import CAGetBoard, init_board, get_perception

device = "cuda" if torch.cuda.is_available() else "cpu"

STEPS = 8000
LR = 2e-3


class NCATask(Enum):
    GROW = 0
    STABLE = 1
    REGENERATE = 2


def train(
    img_path: str,
    get_board: CAGetBoard,
    bs: int = 16,
    pool_size: int = 1024,
    task: NCATask = NCATask.REGENERATE,
):
    # set up the pool and save the resized target image
    if task == NCATask.GROW:
        pool_size = bs
    seed, target = init_board(img_path, 0.0)
    pool = torch.stack([seed.clone() for _ in range(pool_size)], dim=0)
    pool = pool.to(device)
    target = target.to(device)
    torchvision.utils.save_image(target, f"a_target.png")

    # optimizer + loss
    opt = AdamW(
        get_board.parameters(),
        lr=LR,
    )
    loss_fn = torch.nn.MSELoss(reduction="none").to(device)

    for steps in range(STEPS):  # number of optimization steps
        steps_till_opt = int(
            (64 + (96 - 64) * torch.rand(1)).item()
        )  # number of CA steps until optimization
        for _ in range(steps_till_opt):
            pool_sample = torch.randperm(pool_size)[:bs]
            sampled_boards = pool[pool_sample]

            if (
                task == NCATask.STABLE
            ):  # non growing tasks require modifying the pool...
                with torch.no_grad():
                    pre_imgs = sampled_boards[:, :4]
                    pre_loss = (pre_imgs - target).square().mean((1, 2, 3))
                worst = pre_loss.argmax()
                sampled_boards[worst] = seed

            sampled_boards = get_board(sampled_boards)

            # update pool
            pool[pool_sample] = sampled_boards

        # time to calculate loss!
        board_imgs = sampled_boards[:, :4, :, :]
        loss_vals = loss_fn(target.unsqueeze(0).expand_as(board_imgs), board_imgs)
        loss_val = loss_vals.mean()

        # grad norm helps
        grads = torch.autograd.grad(loss_val, get_board.parameters(), retain_graph=True)
        normalized_grads = [g / (g.norm() + 1e-8) for g in grads]
        for p, g in zip(get_board.parameters(), normalized_grads):
            if p.grad is None:
                p.grad = g.clone()
            else:
                p.grad.copy_(g)

        opt.step()
        opt.zero_grad()
        pool = pool.detach()

        loss_vals = loss_vals.mean(dim=(1, 2, 3))

        pool[pool_sample] = (
            seed.unsqueeze(0).repeat(bs, 1, 1, 1).to(device).detach()
        )

        if steps % 100 == 0:
            print(loss_vals)
            torchvision.utils.save_image(
                board_imgs, f"data/boards/{steps}.png", nrow=int(bs**0.5)
            )
    # Save model weights at the end of training
    torch.save(get_board.state_dict(), "data/params/mudkip.pt")


if __name__ == "__main__":
    get_board = CAGetBoard().to(device)
    train("data/mudkip.png", get_board, task=NCATask.STABLE)
