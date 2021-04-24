from argparse import ArgumentParser

import luafun.game.constants as const
from luafun.dataset.draft import Dota2PickBan
from luafun.model.actor_critic import SimpleDrafter, LSTMDrafter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
import torch.optim as optim


def stack_obs(args):
    return torch.stack([x for x, _, _ in args]), \
           torch.stack([y for _, y, _ in args]), \
           torch.stack([z for _, _, z in args]),


def cat_obs(args):
    return torch.cat([x for x, _, _ in args]), \
           torch.cat([y for _, y, _ in args]), \
           torch.cat([z for _, _, z in args]),


class HeroRole(nn.Module):
    def __init__(self, hero_encoder):
        super(HeroRole, self).__init__()
        self.role_count = len(const.ROLES)

        self.hero_encoder = hero_encoder
        self.hero_role_decoder = nn.Linear(hero_encoder.out_size, self.role_count, bias=False)

    def forward(self, x):
        x = self.hero_encoder(x)
        return self.hero_role_decoder(x)


class RoleTrainer:
    def __init__(self, model):
        self.model = model
        self.role_count = len(const.ROLES)

        roles = dict()
        for i, role in enumerate(const.ROLES):
            roles[role] = i

        self.x = torch.zeros((const.HERO_COUNT, const.HERO_COUNT))
        self.y = torch.zeros((const.HERO_COUNT, self.role_count))

        for hero in const.HEROES:
            self.x[hero['offset'], hero['offset']] = 1

            for role, _ in hero['roles'].items():
                self.y[hero['offset'], roles[role]] = 1

        self.loss = nn.BCEWithLogitsLoss().cuda()

        self.x = self.x.cuda()
        self.y = self.y.cuda()

    def forward(self):
        role = self.model(self.x)
        cost = self.loss(role, self.y)
        cost.backward()
        return cost.item()


def pertub(params, var=2):
    with torch.no_grad():
        for param in params:
            if param.grad is not None:
                param += param.grad * torch.randn_like(param) * var


def train(args, dataset):
    sampler = RandomSampler(dataset)

    batch_size = 4096
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        collate_fn=stack_obs)

    model = LSTMDrafter()
    hero_role = HeroRole(model.hero_encoder)

    for param in list(model.parameters()) + list(hero_role.hero_role_decoder.parameters()):
        if len(param.shape) >= 2:
            # kaiming_uniform_
            # kaiming_normal_
            # nn.init.kaiming_uniform_(param, a=0, mode='fan_in', nonlinearity='relu')

            # xavier_uniform_
            # xavier_normal_
            nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))

    # simple_drafter_2795_7.27
    epoch = 0

    if False:
        # lstm_drafter_6534_7.27.pt
        model_name = 'lstm_drafter'
        epoch = 6534
        state = torch.load(f'{model_name}_{epoch}_7.27.pt')
        model.load_state_dict(state)

    model = model.cuda()
    hero_role = hero_role.cuda()
    trainer = RoleTrainer(hero_role)

    loss = nn.CrossEntropyLoss(ignore_index=-1).cuda()
    optimizer = optim.Adam(
        params=list(model.parameters()) + list(hero_role.hero_role_decoder.parameters()),
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    prev = 0
    current_cost = 0

    print(f'epoch, cost, role, grad, update')

    while True:
        epoch += 1
        try:
            total_cost = 0
            total_count = 0
            total_role = 0

            optimizer.zero_grad()
            total_role += trainer.forward()
            optimizer.step()

            for x, y, z in loader:
                optimizer.zero_grad()
                total_role += trainer.forward()

                bs = x.shape[0]
                ppick, pban = model(x.cuda())

                ppick = ppick.view(bs * 12, const.HERO_COUNT)
                pban = ppick.view(bs * 12, const.HERO_COUNT)

                y = y.view(bs * 12)
                z = z.view(bs * 12)

                cost = (loss(ppick, y.cuda()) + loss(pban, z.cuda()))
                cost.backward()
                optimizer.step()

                total_count += 1
                total_cost += cost.item()

            grad = sum([
                abs(p.grad.abs().mean().item()) for p in model.parameters() if p.grad is not None
            ])

            prev = current_cost
            current_cost = total_cost / total_count

            values = [
                epoch,
                current_cost,
                total_role / total_count,
                grad,
                current_cost - prev
            ]
            print(', '.join([str(v) for v in values]))

            if grad < 1e-5 or abs(current_cost - prev) < 1e-5:
                pertub(model.parameters())
                optimizer = optim.Adam(
                    params=list(model.parameters()) + list(hero_role.hero_role_decoder.parameters()),
                    lr=1e-4,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                )
                print('pertub')

        except KeyboardInterrupt:
            break
        # except:
        #     break

    torch.save(model.state_dict(), f'lstm_drafter_{epoch}_7.27.pt')


def main():
    draft_file = '/home/setepenre/work/LuaFun/opendota_CM_20210421.zip'

    parser = ArgumentParser()
    parser.add_argument('--draft', type=str, default=draft_file)
    args = parser.parse_args()

    with Dota2PickBan(args.draft, '7.27') as dataset:
        train(args, dataset)


if __name__ == '__main__':
    main()
