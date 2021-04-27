from argparse import ArgumentParser
from collections import defaultdict

import luafun.game.constants as const
from luafun.dataset.draft import Dota2PickBan, Dota2Matchup
from luafun.model.drafter import SimpleDrafter, LSTMDrafter, DraftJudge
from luafun.model.components import CategoryEncoder
from luafun.model.categorizer import HeroRole
from luafun.environment.draftenv import Dota2DraftEnv
from luafun.utils.options import datafile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
import torch.optim as optim

import random
import json
import math


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


class Trainer:
    def __init__(self):
        self.hero_encoder = CategoryEncoder(const.HERO_COUNT, 128)
        self.hero_role = HeroRole(self.hero_encoder)
        self.drafter = LSTMDrafter(self.hero_encoder)
        self.draft_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.judge = DraftJudge(self.hero_encoder)
        self.role_trainer = RoleTrainer(self.hero_role)
        self.metrics = dict()

    def init_CM_dataloader(self, dataset):
        sampler = RandomSampler(dataset)

        batch_size = 4096
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            collate_fn=Trainer.stack_cm_obs)

        return loader

    @staticmethod
    def stack_cm_obs(args):
        return torch.stack([x for x, _, _ in args]), \
               torch.stack([y for _, y, _ in args]), \
               torch.stack([z for _, _, z in args]),

    @staticmethod
    def cat_cm_obs(args):
        return torch.cat([x for x, _, _ in args]), \
               torch.cat([y for _, y, _ in args]), \
               torch.cat([z for _, _, z in args]),

    def init_judge_dataloader(self, dataset):
        sampler = RandomSampler(dataset)

        batch_size = 4096
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            collate_fn=Trainer.stack_judge_obs)

        return loader

    @staticmethod
    def stack_judge_obs(args):
        return torch.stack([x for x, _ in args]), torch.stack([y for _, y in args])

    def parameters(self):
        params = (
                list(self.drafter.parameters()) +
                list(self.hero_encoder.parameters()) +
                list(self.judge.parameters()) +
                list(self.hero_role.parameters())
        )

        param_ids = set()
        unique_params = []
        for p in params:
            if id(p) in param_ids:
                continue

            param_ids.add(id(p))
            unique_params.append(p)

        return unique_params

    def init_model(self):
        for param in list(self.parameters()):
            if len(param.shape) >= 2:
                nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))

    def init_optim(self):
        optimizer = optim.Adam(
            params=self.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
        return optimizer, scheduler

    def cuda(self):
        self.hero_encoder = self.hero_encoder.cuda()
        self.hero_role = self.hero_role.cuda()
        self.drafter = self.drafter.cuda()
        self.draft_loss = self.draft_loss.cuda()
        self.judge = self.judge.cuda()

    @staticmethod
    def pertub(params, var=2):
        with torch.no_grad():
            for param in params:
                if param.grad is not None:
                    param += param.grad * torch.randn_like(param) * var

    def save(self, epoch):
        torch.save(
            self.drafter.state_dict(),
            datafile('weights', f'lstm_drafter_{epoch}_7.27.pt'))

        torch.save(
            self.judge.state_dict(),
            datafile('weights', f'judge_{epoch}_7.27.pt'))

        torch.save(
            self.hero_encoder.state_dict(),
            datafile('weights', f'henc_{epoch}_7.27.pt'))

        return

    def resume(self):
        epoch = 0
        current_cost = 0

        if True:
            model_name = 'lstm_drafter'
            epoch = 3674
            drafter_state = torch.load(datafile('weights', f'{model_name}_{epoch}_7.27.pt'))
            self.drafter.load_state_dict(drafter_state, strict=False)

            judge_state = torch.load(datafile('weights', f'judge_44_7.27.pt'))
            self.judge.load_state_dict(judge_state)

            hero_encoder_state = torch.load(datafile('weights', f'henc_44_7.27.pt'))
            self.hero_encoder.load_state_dict(hero_encoder_state)

        return epoch, current_cost

    def train_draft_supervised_epoch(self, epoch, loader, optimizer, scheduler):
        optimizer.zero_grad()
        role_cost = self.role_trainer.forward()
        optimizer.step()

        count = 0
        total_cost = 0
        acc_pick = 0
        acc_ban = 0

        for x, y, z in loader:
            optimizer.zero_grad()

            bs = x.shape[0]
            ppick, pban = self.drafter(x.cuda())

            ppick = ppick.view(bs * 12, const.HERO_COUNT)
            pban = ppick.view(bs * 12, const.HERO_COUNT)

            y = y.view(bs * 12).cuda()
            z = z.view(bs * 12).cuda()

            cost = (self.draft_loss(ppick, y) + self.draft_loss(pban, z))
            cost.backward()
            optimizer.step()

            acc_pick = self.accuracy(ppick, y, ignore_index=-1)
            acc_ban = self.accuracy(pban, z, ignore_index=-1)

            count += 1
            total_cost += cost.item()

        scheduler.step()
        n = len(loader.dataset)
        return dict(
            epoch=epoch,
            cm_draft_cost=total_cost / count,
            role_cost=role_cost,
            acc_pick=acc_pick / (n * 5),    # 5 Picks
            acc_ban=acc_ban / (n * 7),      # 7 Bans
        )

    def train_draft_supervised(self):
        dataset = Dota2PickBan(datafile('dataset', 'opendota_CM_20210421.zip'), '7.27')
        optimizer, scheduler = self.init_optim()
        loader = self.init_CM_dataloader(dataset)

        epoch, current_cost = self.resume()
        err = None
        while True:
            epoch += 1
            try:
                prev = current_cost
                metrics = self.train_draft_supervised_epoch(epoch, loader, optimizer, scheduler)

                grad = sum([
                    abs(p.grad.abs().mean().item()) for p in self.drafter.parameters() if p.grad is not None
                ])

                current_cost = metrics['cm_draft_cost']
                cost_diff = current_cost - prev

                metrics['grad'] = grad
                metrics['cm_draft_cost_diff'] = cost_diff

                print(', '.join([str(v) for v in metrics.values()]))

                if grad < 1e-5 or abs(cost_diff) < 1e-5:
                    Trainer.pertub(self.drafter.parameters())
                    optimizer, scheduler = self.init_optim()
                    print('pertub')

            except KeyboardInterrupt:
                break
            except Exception as e:
                err = e
                break

        self.save(epoch)
        if err is not None:
            raise err

    def train_draft_judge_supervised_epoch(self, epoch, loader, optimizer, scheduler):
        optimizer.zero_grad()
        role_cost = self.role_trainer.forward()
        optimizer.step()

        count = 0
        total_cost = 0
        total_acc = 0

        for x, win_target in loader:
            optimizer.zero_grad()

            win_target = win_target.cuda()
            win_predict = self.judge(x.cuda())
            cost = nn.CrossEntropyLoss()(win_predict, win_target)
            cost.backward()
            optimizer.step()

            count += 1
            total_cost += cost.item()
            total_acc += self.accuracy(win_predict, win_target)

        scheduler.step()
        return dict(
            role=role_cost,
            judge_cost=total_cost / count,
            judge_acc=total_acc / len(loader.dataset)
        )

    def accuracy(self, predict, target, ignore_index=None):
        with torch.no_grad():
            _, predicted = torch.max(predict, 1)

            if ignore_index:
                filter = target != ignore_index
            else:
                filter = 1

            return ((predicted == target) * filter).sum().item()

    def train_draft_judge_supervised(self, epochs):
        dataset = Dota2Matchup(datafile('dataset', 'drafting_all.zip'))
        optimizer, scheduler = self.init_optim()
        loader = self.init_judge_dataloader(dataset)

        epoch = 0
        current_cost = 0
        # epoch, current_cost = self.resume()
        err = None
        for _ in range(epochs):
            epoch += 1
            try:
                prev = current_cost
                metrics = self.train_draft_judge_supervised_epoch(epoch, loader, optimizer, scheduler)

                grad = sum([
                    abs(p.grad.abs().mean().item()) for p in self.judge.parameters() if p.grad is not None
                ])

                current_cost = metrics['judge_cost']
                cost_diff = current_cost - prev

                metrics['grad'] = grad
                metrics['judge_cost_diff'] = cost_diff

                print(', '.join([str(v) for v in metrics.values()]))

                if grad < 1e-5 or abs(current_cost - prev) < 1e-5:
                    Trainer.pertub(self.judge.parameters())
                    optimizer, scheduler = self.init_optim()
                    print('pertub')

            except KeyboardInterrupt:
                break
            except Exception as e:
                err = e
                break

        self.save(epoch)
        if err is not None:
            raise err

    def train_draft_rl_selfplay(self):
        env = Dota2DraftEnv()
        optimizer, scheduler = self.init_optim()
        stats = defaultdict(int)
        self.resume()

        while True:

            for _ in range(32):
                cost, total_reward = self.train_draft_rl_selfplay_episode(env, optimizer, scheduler, stats)
                print(cost, total_reward)

            self.train_draft_judge_supervised(2)

            total = 0
            for k, v in stats.items():
                total += v

            first_pick = stats["rad_first_pick"] + stats["dire_first_pick"]
            radiant = stats["rad_first_pick"] + stats["rad_last_pick"]

            print()
            print(f'- First Pick Win Rate: {first_pick * 100 / total:6.2f} ({first_pick})')
            print(f'- Radiant Win Rate   : {radiant * 100 / total:6.2f} ({radiant})')

            for k, v in stats.items():
                print(f'    - {k:>15}: {v / total * 100:6.2f} ({v}))')
            print()

    def train_draft_rl_selfplay_episode(self, env, optimizer, scheduler, stats):
        state, reward, done, info = env.reset()

        rnn_state = None
        optimizer.zero_grad()

        reward_rad = torch.zeros(25)
        value_rad = torch.zeros(25)
        logprobs_rad = torch.zeros(25)
        entropy_rad = torch.zeros(25)

        reward_dire = torch.zeros(25)
        value_dire = torch.zeros(25)
        logprobs_dire = torch.zeros(25)
        entropy_dire = torch.zeros(25)
        i = 0

        total_reward = 0

        while not done:
            state = torch.stack(state).cuda()

            (pick, ban), (pick_logprobs, ban_logprobs), (pick_entropy, ban_entropy), value, rnn_state\
                = self.drafter.action(state, prev=rnn_state, reserved=env.reserved_offsets)

            radiant = (pick[0, 0].item(), ban[0, 0].item())
            dire = (pick[1, 0].item(), ban[1, 0].item())
            state, reward, done, info = env.step((radiant, dire))

            # radiant
            reward_rad[i] = reward[0]
            value_rad[i] = value[0]

            logprobs_rad[i] = pick_logprobs[0] + ban_logprobs[0]
            entropy_rad[i] = pick_entropy[0] + ban_entropy[0]

            # dire
            reward_dire[i] = reward[1]
            value_dire[i] = value[1]

            logprobs_dire[i] = pick_logprobs[1] + ban_logprobs[1]
            entropy_dire[i] = pick_entropy[1] + ban_entropy[1]
            i += 1

            total_reward += reward[0]

        with torch.no_grad():
            radiant_draft_state = state[0].unsqueeze(0).cuda()
            last_reward = self.judge(radiant_draft_state)

            radiant_win = last_reward[0, 0].item()

            reward_rad[i - 1] += radiant_win
            reward_dire[i - 1] += 1 - radiant_win

            if radiant_win > 0.50:
                if env.radiant_started:
                    stats[f'rad_first_pick'] += 1
                else:
                    stats[f'rad_last_pick'] += 1
            else:
                if env.radiant_started:
                    stats[f'dire_last_pick'] += 1
                else:
                    stats[f'dire_first_pick'] += 1

            total_reward += last_reward[0, 0].item()

        discount_rate = math.exp(0.10)
        prev_reward_rad = reward_rad[24]
        prev_reward_dire = reward_dire[24]

        # To help training we discount reward from the end to the beginning
        # value now becomes a prediction of future income as well as current reward
        for i in range(23):
            prev_reward_rad = reward_rad[23 - i] + prev_reward_rad / discount_rate
            reward_rad[23 - i] = prev_reward_rad

            prev_reward_dire = reward_dire[23 - i] + prev_reward_dire / discount_rate
            reward_dire[23 - i] = prev_reward_dire

        advantage_rad = (reward_rad - value_rad).cuda()
        advantage_dire = (reward_dire - value_dire).cuda()

        logprobs_rad = logprobs_rad.cuda()
        entropy_rad = entropy_rad.cuda()

        logprobs_dire = logprobs_dire.cuda()
        entropy_dire = entropy_dire.cuda()

        value_loss_rad = advantage_rad.pow(2).mean()
        action_loss_rad = -(advantage_rad.detach() * logprobs_rad).mean()
        loss_rad = (value_loss_rad + action_loss_rad - entropy_rad.mean())

        value_loss_dire = advantage_dire.pow(2).mean()
        action_loss_dire = -(advantage_dire.detach() * logprobs_dire).mean()
        loss_dire = (value_loss_dire + action_loss_dire - entropy_dire.mean())

        if random.random() < 0.5:
            loss = loss_rad     # + loss_dire
        else:
            loss = loss_dire

        loss.backward()

        # cost.backward()
        optimizer.step()
        scheduler.step()

        return loss.item(), total_reward


if __name__ == '__main__':
    # loss = nn.CrossEntropyLoss()
    # input = torch.randn(3, 2, requires_grad=True)
    # print(input)
    # target = torch.empty(3, dtype=torch.long).random_(2)
    # target[0] = 0
    # print(target)
    # output = loss(input, target)
    # output.backward()

    t = Trainer()
    t.cuda()
    t.init_optim()

    # self-play
    # t.train_draft_rl_selfplay()

    # train judge
    # t.train_draft_judge_supervised()

    # supervised train
    t.train_draft_supervised()

