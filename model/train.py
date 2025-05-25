import torch, torchvision
from torch.optim.adamw import AdamW
from torch.utils.tensorboard import SummaryWriter

from enum import Enum

from utils import CAGetBoard, init_board, destroy, PAD_AMT
from pathlib import Path
import os, time, uuid

device = "cuda" if torch.cuda.is_available() else "cpu"

LR = 2e-3
STEPS = 30000
ITER_RANGE = (64, 96)
DESTROY_RAD_RANGE = (0.1, 0.4)


class NCATask(Enum):
    GROW = "grow"
    PERSIST = "persist"
    REGENERATE = "regenerate"


def train(
    img_path: str,
    state_size: int = 16,
    learned_features: bool = False,
    bs: int = 8,
    pool_size: int = 1024,
    task: NCATask = NCATask.REGENERATE,
    load_from: str = None,
):
    img_name = img_path.split("/")[-1].split(".")[0]
    run_name = f"{img_name}_{task.value}"
    exp_name = f"{run_name}_{uuid.uuid4()}"
    exp_path = f"data/experiments/{exp_name}"
    os.makedirs(exp_path, exist_ok=False)

    logger = SummaryWriter(f"runs/{exp_name}")
    logger.add_text("exp_path", exp_path)

    update_board = CAGetBoard(
        state_size=state_size, learned_features=learned_features
    ).to(device)
    if load_from is not None:
        update_board.load_state_dict(torch.load(load_from))

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
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    loss_fn = torch.nn.MSELoss().to(device)

    for steps in range(STEPS):  # number of optimization steps
        t0 = time.time()
        steps_till_opt = int(
            (ITER_RANGE[0] + (ITER_RANGE[1] - ITER_RANGE[0]) * torch.rand(1)).item()
        )  # number of CA steps until optimization

        pool_sample = torch.randperm(pool_size)[:bs]
        sampled_boards = pool[pool_sample]
        if (
            task == NCATask.PERSIST or task == NCATask.REGENERATE
        ):  # non growing tasks require modifying the sample...
            with torch.no_grad():
                pre_imgs = sampled_boards[:, :4]
                pre_loss = (pre_imgs - target).square().mean((1, 2, 3))
                rank = pre_loss.argsort(descending=True)

                sampled_boards = sampled_boards[rank]
                sampled_boards[0] = seed

                if task == NCATask.REGENERATE:
                    B, _, H, W = sampled_boards.shape
                    centers = torch.cat(
                        (
                            torch.randint(PAD_AMT, H - PAD_AMT, (2, 1)),
                            torch.randint(PAD_AMT, W - PAD_AMT, (2, 1)),
                        ),
                        dim=1,
                    ).to(device)
                    radius = DESTROY_RAD_RANGE[0] + (
                        DESTROY_RAD_RANGE[1] - DESTROY_RAD_RANGE[0]
                    ) * torch.rand(
                        2,
                    )
                    sampled_boards[-2:] = destroy(sampled_boards[-2:], centers, radius)

        opt.zero_grad()
        for _ in range(steps_till_opt):
            sampled_boards = update_board(sampled_boards)
        # time to calculate loss!
        board_imgs = sampled_boards[:, :4, :, :]
        loss_val = loss_fn(target, board_imgs)

        # grad norm helps
        # collect (name,param) pairs first
        named_params = list(update_board.named_parameters())
        grads = torch.autograd.grad(
            loss_val, [p for _, p in named_params], retain_graph=True
        )

        for (name, p), g in zip(named_params, grads):
            if "layer0" not in name:  # keep first layer raw
                g = g / (g.norm() + 1e-8)
            if p.grad is None:
                p.grad = g.clone()
            else:
                p.grad.copy_(g)

        opt.step()
        lr_sched.step()

        if task == NCATask.GROW:
            pool[pool_sample] = (
                seed.unsqueeze(0).repeat(bs, 1, 1, 1).to(device).detach()
            )
        else:
            pool[pool_sample] = sampled_boards.detach()

        if steps % 100 == 0:
            logger.add_scalar("time", time.time() - t0, steps)
            logger.add_scalar("loss", loss_val.item(), steps)
            grid = torchvision.utils.make_grid(
                board_imgs[:, :3], nrow=int(bs**0.5), normalize=True, value_range=(0, 1)
            )
            logger.add_image("board", grid, steps, dataformats="CHW")

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
    logger.close()


if __name__ == "__main__":
    # tasks = [NCATask.REGENERATE, NCATask]
    # img_paths = list(Path("data/src/medium").glob("*.png"))
    # for task in tasks:
    #     for img_path in img_paths:
    #         state_size = 32 if "mewtwo" in img_path.name else 16
    #         train(f"{img_path}", state_size=state_size, task=task)

    train(
        "data/src/medium/arceus.png",
        state_size=32,
        task=NCATask.REGENERATE,
        # load_from="data/experiments/mewtwo_regenerate_0fb64bb6-5bc2-41c2-8bea-5e50145bd242/params.pt",
    )
