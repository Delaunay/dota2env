from argparse import ArgumentParser

from luafun.dataset.draft import Dota2PickBan
from luafun.model.actor_critic import SimpleDrafter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
import torch.optim as optim


def stack_obs(args):
    return torch.cat([x for x, _, _ in args]), \
           torch.cat([y for _, y, _ in args]), \
           torch.cat([z for _, _, z in args]),


def train(args, dataset):
    sampler = RandomSampler(dataset)

    loader = DataLoader(
        dataset,
        batch_size=4096,
        sampler=sampler,
        num_workers=0,
        collate_fn=stack_obs)

    model = SimpleDrafter().cuda()
    loss = nn.CrossEntropyLoss(ignore_index=-1).cuda()
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=1e-6,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    for _ in range(10):
        total_cost = 0
        total_count = 0

        for x, y, z in loader:
            ppick, pban = model(x.cuda())

            optimizer.zero_grad()
            cost = (loss(ppick, y.cuda()) + loss(pban, z.cuda()))
            optimizer.step()

            total_count += 1
            total_cost += cost.detach()

        print(total_cost / total_count)


def main():
    draft_file = '/home/setepenre/work/LuaFun/opendota_CM_20210421.zip'

    parser = ArgumentParser()
    parser.add_argument('--draft', type=str, default=draft_file)
    args = parser.parse_args()

    with Dota2PickBan(args.draft) as dataset:
        train(args, dataset)


if __name__ == '__main__':
    main()
