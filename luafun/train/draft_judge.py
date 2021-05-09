from argparse import ArgumentParser
from collections import defaultdict

import luafun.game.constants as const
from luafun.dataset.draft import Dota2PickBan, Dota2Matchup
from luafun.model.drafter import SimpleDrafter, LSTMDrafter, DraftJudge
from luafun.model.components import HeroEmbedding
from luafun.model.categorizer import HeroRole
from luafun.environment.draftenv import Dota2DraftEnv
from luafun.utils.options import datapath, datafile
from luafun.train.metrics import MetricWriter

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
        self.hero_encoder = HeroEmbedding(128)
        self.hero_role = HeroRole(self.hero_encoder)
        self.drafter = LSTMDrafter(self.hero_encoder)
        self.draft_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.judge = DraftJudge(self.hero_encoder)
        # self.role_trainer = RoleTrainer(self.hero_role)
        self.writer = MetricWriter(datapath('metrics'))
        self.meta = dict()

    def init_CM_dataloader(self, dataset):
        sampler = RandomSampler(dataset)

        batch_size = 2048
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

        batch_size = 2048
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            collate_fn=Trainer.stack_judge_obs)

        return loader

    @staticmethod
    def stack_judge_obs(args):
        return torch.stack([x for x, _, _, _ in args]), \
               torch.stack([y for _, y, _, _ in args]), \
               torch.stack([z for _, _, z, _ in args]), \
               torch.stack([d for _, _, _, d in args])

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

    def init_hero_encoder(self):
        weight = list(self.hero_encoder.parameters())[0]

        max_w = 0
        roles = set()
        for hero in const.HEROES:
            role = hero.get('roles')

            for r, w in role.items():
                roles.add(r)
                max_w = max(int(w), max_w)

        roles = {k: i for i, k in enumerate(list(roles))}
        for hero in const.HEROES:
            offset = hero['offset']
            role = hero.get('roles')

            for i in range(len(roles)):
                weight[i, offset] = 0

            for k, w in role.items():
                i = roles[k]
                weight[i, offset] = int(w) / max_w

        self.hero_encoder.baseline.linear.weight = nn.Parameter(weight)

    def save(self):
        import io

        buffer = io.BytesIO()
        torch.save(self.drafter.state_dict(), buffer)
        self.writer.save_weights('lstm_drafter', buffer.getbuffer())

        buffer = io.BytesIO()
        torch.save(self.judge.state_dict(), buffer)
        self.writer.save_weights('judge', buffer.getbuffer())

        buffer = io.BytesIO()
        torch.save(self.hero_encoder.state_dict(), buffer)
        self.writer.save_weights('henc', buffer.getbuffer())
        self.writer.save_meta(self.meta)

    def resume(self):
        lstm_drafter = self.writer.load_weights('lstm_drafter')
        self.drafter.load_state_dict(lstm_drafter)

        judge = self.writer.load_weights('judge')
        self.judge.load_state_dict(judge)

        henc = self.writer.load_weights('henc')
        self.hero_encoder.load_state_dict(henc)

        self.meta = self.writer.load_meta()
        return

    def train_draft_supervised_epoch(self, epoch, loader, optimizer, scheduler):
        optimizer.zero_grad()
        # role_cost = self.role_trainer.forward()
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
            name='CM-draft',
            epoch=epoch,
            cm_draft_cost=total_cost / count,
            # role_cost=role_cost,
            acc_pick=acc_pick / (n * 5),    # 5 Picks
            acc_ban=acc_ban / (n * 7),      # 7 Bans
        )

    def train_draft_supervised(self, epochs):
        dataset = Dota2PickBan(datafile('dataset', 'opendota_CM_20210421.zip'))
        optimizer, scheduler = self.init_optim()
        loader = self.init_CM_dataloader(dataset)

        epoch = self.meta.get('cm_draft_epoch', 0)
        current_cost = self.meta.get('cm_draft_cost', 0)

        err = None
        for _ in range(epochs):
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
                self.writer.save_metrics(metrics)

                # self.train_draft_rl_selfplay(5, dict())

                if grad < 1e-5 or abs(cost_diff) < 1e-5:
                    Trainer.pertub(self.drafter.parameters())
                    optimizer, scheduler = self.init_optim()
                    print('pertub')

            except Exception as e:
                err = e
                break

        self.meta['cm_draft_epoch'] = epoch
        self.meta['cm_draft_cost'] = current_cost

        if err is not None:
            raise err

    def train_draft_judge_supervised_epoch(self, epoch, loader, optimizer, scheduler):
        optimizer.zero_grad()
        # role_cost = self.role_trainer.forward()
        optimizer.step()

        count = 0
        total_cost_norm = 0
        total_cost = 0
        total_acc = 0

        prev_cost_win = 1
        prev_cost_est = 1
        prev_cost_win_c = 0
        prev_cost_est_c = 0

        for x, win_target, meta, rank in loader:
            x = x.cuda()
            rank = rank.cuda()
            win_target = win_target.cuda()
            meta = meta.cuda()

            self.judge.compute_estimates = True

            optimizer.zero_grad()
            win_predict, estimates = self.judge(x, rank)
            # Normalize cost, to make them both as important
            cost_win = nn.CrossEntropyLoss()(win_predict, win_target)
            cost_est = nn.L1Loss()(estimates, meta)
            cost = (cost_win / prev_cost_win) + (cost_est / prev_cost_est)

            cost.backward()
            optimizer.step()

            prev_cost_win = cost_win.item()
            prev_cost_est = cost_est.item()

            if prev_cost_est_c == 0:
                prev_cost_est_c = prev_cost_est
                prev_cost_win_c = prev_cost_win

            count += 1
            if count == 1:
                total_cost_norm += 1
            else:
                total_cost_norm += cost.item()

            total_cost += ((cost_win / prev_cost_win_c) + (cost_est / prev_cost_est_c)).item()
            total_acc += self.accuracy(win_predict, win_target)

        scheduler.step()
        return dict(
            name='judge',
            epoch=epoch,
            # role=role_cost,
            judge_cost_norm=total_cost_norm / count,
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

    def train_draft_judge_supervised(self, epochs, optim_scheduler=None):
        dataset = Dota2Matchup(datafile('dataset', 'ranked_allpick_7.28_picks_wip.zip'))

        if optim_scheduler is None:
            optim_scheduler = self.init_optim()

        optimizer, scheduler = optim_scheduler
        loader = self.init_judge_dataloader(dataset)

        epoch = self.meta.get('judge_epoch', 0)
        current_cost = self.meta.get('judge_cost', 0)

        self.judge = DraftJudge(self.hero_encoder, compute_estimates=True).cuda()

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
                self.writer.save_metrics(metrics)

                # grad < 1e-5       : Training is stuck
                # cost_diff < 1e-5  : Training is stuck
                # grad > 10         : Init has been bad for some time
                if grad < 1e-5 or abs(current_cost - prev) < 1e-5:
                    Trainer.pertub(self.judge.parameters())
                    optimizer, scheduler = self.init_optim()
                    print('pertub')

                if metrics['judge_acc'] > 0.80:
                    break

            except Exception as e:
                err = e
                break

        self.meta['judge_epoch'] = epoch
        self.meta['judge_cost'] = current_cost


        if err is not None:
            raise err

    def train_draft_rl_selfplay(self, episodes, win_stats=None, optim_scheduler=None):
        env = Dota2DraftEnv()

        if optim_scheduler is None:
            optim_scheduler = self.init_optim()

        if win_stats is None:
            win_stats = defaultdict(int)

        optimizer, scheduler = optim_scheduler

        cost_self = 0
        radiant_reward = 0
        dire_reward = 0
        episode = self.meta.get('episodes_self', 0)
        start = episode % episodes

        for i in range(start, episodes):
            metrics = self.train_draft_rl_selfplay_episode(env, optimizer, scheduler, win_stats)
            self.writer.save_metrics(metrics)

            cost_self += metrics['cost_self']
            radiant_reward += metrics['radiant_reward']
            dire_reward += metrics['dire_reward']

            if i % 1000 == 0:
                self.save()

        self.meta['episodes_self'] = episode + episodes
        metrics = dict(
            name='self',
            episode=self.meta['episodes_self'],
            cost_self=cost_self / episodes,
            radiant_reward=radiant_reward / episodes,
            dire_reward=dire_reward / episodes
        )
        print(', '.join([str(v) for v in metrics.values()]))

        total = 0
        for k, v in win_stats.items():
            total += v

        first_pick = win_stats["rad_first_pick"] + win_stats["dire_first_pick"]
        radiant = win_stats["rad_first_pick"] + win_stats["rad_last_pick"]

        self.meta['first_pick'] = first_pick
        self.meta['radiant'] = radiant

        print()
        print(f'- First Pick Win Rate: {first_pick * 100 / total:6.2f} ({first_pick})')
        print(f'- Radiant Win Rate   : {radiant * 100 / total:6.2f} ({radiant})')

        for k, v in win_stats.items():
            print(f'    - {k:>15}: {v / total * 100:6.2f} ({v})')
        print()
        self.meta['win_stats'] = win_stats

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
            self.judge.compute_estimates = False
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

        discount_rate = math.exp(-0.10)
        prev_reward_rad = reward_rad[24]
        prev_reward_dire = reward_dire[24]

        # To help training we discount reward from the end to the beginning
        # value now becomes a prediction of future income as well as current reward
        for i in range(23):
            prev_reward_rad = reward_rad[23 - i] + prev_reward_rad * discount_rate
            reward_rad[23 - i] = prev_reward_rad

            prev_reward_dire = reward_dire[23 - i] + prev_reward_dire * discount_rate
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

        # if random.random() < 0.5:
        #     loss = loss_rad     # + loss_dire
        # else:
        #     loss = loss_dire

        loss = loss_rad + loss_dire
        loss.backward()

        # cost.backward()
        optimizer.step()
        scheduler.step()

        return dict(
            cost_self=loss.item(),
            radiant_reward=last_reward[0, 0].item(),
            dire_reward=last_reward[0, 1].item()
        )

    def free_grads(self):
        import gc

        for param in self.parameters():
            param.grad = None

        gc.collect()
        torch.cuda.empty_cache()

    def train_unified(self, uid=None):
        if uid is not None:
            self.writer = MetricWriter(datapath('metrics'), uid)
            self.resume()

        win_stats = self.meta.get('win_stats', defaultdict(int))
        k = self.meta.get('k', 0)
        err = None

        # We need the judge for self-play
        self.train_draft_judge_supervised(50)
        self.free_grads()

        dataset = Dota2PickBan(datafile('dataset', 'opendota_CM_20210421.zip'), '7.27')
        match_count = len(dataset)

        print(match_count)

        while True:
            try:
                # Do some draft against yourself
                self.train_draft_rl_selfplay(match_count * 10, win_stats)
                self.free_grads()

                # Learn Human strategies
                self.train_draft_supervised(10)
                self.free_grads()

                self.save()
                print('Saved')
                self.meta['k'] = k
                k += 1
            except KeyboardInterrupt:
                print('exiting out of the training loop')
                break
            except Exception as e:
                err = e
                break

        if err is not None:
            raise err


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

    # self-play
    # t.train_draft_rl_selfplay()
    t.init_model()
    t.init_hero_encoder()
    t.save()

    # train judge
    # t.train_draft_judge_supervised()

    # supervised train
    # t.train_draft_supervised()

    # Train all together
    uid = None
    uid = 'c66acbc68bbc40d69b16f9ddd88a1657'
    # t.train_unified(uid)
    t.train_draft_judge_supervised(1000)
