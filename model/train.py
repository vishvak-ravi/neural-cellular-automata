import torch, torchvision
import torch.nn.functional as F
from torch.optim.adamw import AdamW

from enum import Enum

from utils import CAGetBoard, init_board, get_perception
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import os
import uuid

device = "cuda" if torch.cuda.is_available() else "cpu"

LR = 2e-3
STEPS = 8000
ITER_RANGE = (64, 96)


class NCATask(Enum):
    GROW = "grow"
    PERSIST = "persist"
    REGENERATE = "regenerate"


def train(
    img_path: str,
    state_size: int = 16,
    bs: int = 8,
    pool_size: int = 1024,
    task: NCATask = NCATask.REGENERATE,
):
    img_name = img_path.split("/")[-1].split(".")[0]
    run_name = f"{img_name}_{task.value}"
    exp_path = f"data/experiments/{run_name}_{uuid.uuid4()}"
    os.makedirs(exp_path, exist_ok=False)

    update_board = CAGetBoard(state_size=state_size).to(device)

    # set up the pool and save the resized target image
    if task == NCATask.GROW:
        pool_size = bs
    seed, target = init_board(img_path, state_size)
    seed = seed.to(device)
    target = target.to(device)
    target = target.unsqueeze(0).repeat(bs, 1, 1, 1)
    pool = torch.stack([seed.clone() for _ in range(pool_size)], dim=0)

    # optimizer + loss
    opt = AdamW(
        update_board.parameters(),
        lr=LR,
    )
    lr_sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda step: 1.0 if step < 2000 else 0.1
    )
    loss_fn = torch.nn.MSELoss().to(device)

    for steps in range(STEPS):  # number of optimization steps
        steps_till_opt = int(
            (ITER_RANGE[0] + (ITER_RANGE[1] - ITER_RANGE[0]) * torch.rand(1)).item()
        )  # number of CA steps until optimization

        pool_sample = torch.randperm(pool_size)[:bs]
        sampled_boards = pool[pool_sample]
        if (
            task == NCATask.PERSIST or task == NCATask.REGENERATE
        ):  # non growing tasks require modifying the pool...
            with torch.no_grad():
                pre_imgs = sampled_boards[:, :4]
                pre_loss = (pre_imgs - target).square().mean((1, 2, 3))
            rank = pre_loss.argsort(descending=True)

            sampled_boards = sampled_boards[rank]
            sampled_boards[0].copy_(seed)
        for _ in range(steps_till_opt):
            sampled_boards = update_board(sampled_boards)
        # time to calculate loss!
        board_imgs = sampled_boards[:, :4, :, :]
        loss_val = loss_fn(target, board_imgs)

        # grad norm helps
        grads = torch.autograd.grad(
            loss_val, update_board.parameters(), retain_graph=True
        )
        normalized_grads = [g / (g.norm() + 1e-8) for g in grads]
        for p, g in zip(update_board.parameters(), normalized_grads):
            if p.grad is None:
                p.grad = g.clone()
            else:
                p.grad.copy_(g)

        opt.step()
        lr_sched.step()
        opt.zero_grad()

        if task == NCATask.GROW:
            pool[pool_sample] = (
                seed.unsqueeze(0).repeat(bs, 1, 1, 1).to(device).detach()
            )
        elif task == NCATask.PERSIST:
            pool[pool_sample] = sampled_boards.detach()

        if steps % 1000 == 0:
            print(loss_val)
            torchvision.utils.save_image(
                board_imgs,
                f"{exp_path}/{steps}.png",
                nrow=int(bs**0.5),
            )
            # Save model weights at the end of training
            if steps > 100:
                torch.save(update_board.state_dict(), f"{exp_path}/params_{steps}.pt")
    # Save model weights at the end of training
    torch.save(update_board.state_dict(), f"{exp_path}/params.pt")


if __name__ == "__main__":
    tasks = [NCATask.GROW, NCATask.PERSIST]
    img_paths = list(Path("data/src/small").glob("*.png"))
    for task in tasks:
        for img_path in img_paths:
            train(f"{img_path}", state_size=16, task=task)
    # train("charizard_x.png", state_size=24, task=NCATask.PERSIST)
