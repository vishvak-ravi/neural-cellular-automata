import torch, torchvision
import torch.nn.functional as F
from torch.optim.adamw import AdamW

from enum import Enum

from utils import CARule, init_board, get_perception

device = "cuda" if torch.cuda.is_available() else "cpu"

STEPS = 8000
EPS = 0.5
LR = 2e-3


class NCATask(Enum):
    GROW = 0
    STABLE = 1
    REGENERATE = 2


def train(
    img_path: str,
    update_rule: CARule,
    bs: int = 16,
    pool_size: int = 1024,
    task: NCATask = NCATask.REGENERATE,
):
    # set up the pool and save the resized target image
    if task == NCATask.GROW:
        pool_size = bs
    board, target = init_board(img_path, 0.0)
    pool = torch.stack([board.clone() for _ in range(pool_size)], dim=0)
    pool = pool.to(device)
    target = target.to(device)
    torchvision.utils.save_image(target, f"a_target.png")

    # optimizer + loss
    opt = AdamW(
        update_rule.parameters(),
        lr=LR,
    )
    loss_fn = torch.nn.MSELoss(reduction="none").to(device)

    for steps in range(STEPS):  # number of optimization steps
        steps_till_opt = int(
            (64 + (96 - 64) * torch.rand(1)).item()
        )  # number of CA steps until optimization
        for _ in range(steps_till_opt):
            pool_sample = torch.randperm(pool_size)[:bs]
            boards = pool[pool_sample]

            perception = get_perception(boards)
            dboard = update_rule(perception)

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

            # update pool
            pool[pool_sample] = boards

        # time to calculate loss!
        board_imgs = boards[:, :4, :, :]
        loss_vals = loss_fn(target.unsqueeze(0).expand_as(board_imgs), board_imgs)
        loss_val = loss_vals.mean()

        # grad norm helps
        grads = torch.autograd.grad(
            loss_val, update_rule.parameters(), retain_graph=True
        )
        normalized_grads = [g / (g.norm() + 1e-8) for g in grads]
        for p, g in zip(update_rule.parameters(), normalized_grads):
            if p.grad is None:
                p.grad = g.clone()
            else:
                p.grad.copy_(g)

        opt.step()
        opt.zero_grad()
        pool = pool.detach()

        board, _ = init_board("data/mudkip.png", 0.0)
        boards = boards.to(device)
        loss_vals = loss_vals.mean(dim=(1, 2, 3))

        # adjust the pool
        if task == NCATask.GROW:
            pool = board.unsqueeze(0).repeat(bs, 1, 1, 1).to(device).detach()
        elif task == NCATask.STABLE:  # non growing tasks require modifying the pool...
            worst_loss_batch_idx = loss_vals.argmax().item()
            worst_loss_pool_idx = pool_sample[worst_loss_batch_idx]
            pool[worst_loss_pool_idx] = board.to("cuda").detach()

        if steps % 100 == 0:
            print(loss_vals)
            torchvision.utils.save_image(
                board_imgs, f"data/boards/{steps}.png", nrow=int(bs**0.5)
            )
    # Save model weights at the end of training
    torch.save(update_rule.state_dict(), "data/params/mudkip.pt")


if __name__ == "__main__":
    update_rule = CARule(dense=False).to(device)
    train("data/mudkip.png", update_rule=update_rule, task=NCATask.GROW)
