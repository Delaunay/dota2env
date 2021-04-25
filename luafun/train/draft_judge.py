from argparse import ArgumentParser

import luafun.game.constants as const
from luafun.dataset.draft import Dota2PickBan, Dota2Matchup
from luafun.model.drafter import SimpleDrafter, LSTMDrafter, DraftJudge
from luafun.model.components import CategoryEncoder
from luafun.model.categorizer import HeroRole
from luafun.environment.draftenv import Dota2DraftEnv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
import torch.optim as optim


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
        torch.save(self.drafter.state_dict(), f'lstm_drafter_{epoch}_7.27.pt')
        torch.save(self.judge.state_dict(), f'judge_{epoch}_7.27.pt')
        torch.save(self.hero_encoder.state_dict(), f'henc_{epoch}_7.27.pt')
        return

    def resume(self):
        epoch = 0
        current_cost = 0

        if True:
            model_name = 'lstm_drafter'
            epoch = 3674
            state = torch.load(f'{model_name}_{epoch}_7.27.pt')

            self.drafter.load_state_dict(state, strict=False)
            self.judge.load_state_dict(torch.load(f'judge_44_7.27.pt'))
            # self.hero_encoder.load_state_dict(torch.load(f'henc_{epoch}_7.27.pt'))

        return epoch, current_cost

    def train_draft_supervised_epoch(self, epoch, loader, optimizer, scheduler):
        optimizer.zero_grad()
        role_cost = self.role_trainer.forward()
        optimizer.step()

        count = 0
        total_cost = 0

        for x, y, z in loader:
            optimizer.zero_grad()

            bs = x.shape[0]
            ppick, pban = self.drafter(x.cuda())

            ppick = ppick.view(bs * 12, const.HERO_COUNT)
            pban = ppick.view(bs * 12, const.HERO_COUNT)

            y = y.view(bs * 12)
            z = z.view(bs * 12)

            cost = (self.draft_loss(ppick, y.cuda()) + self.draft_loss(pban, z.cuda()))
            cost.backward()
            optimizer.step()

            count += 1
            total_cost += cost.item()

        scheduler.step()
        return role_cost, total_cost / count

    def train_draft_supervised(self):
        draft_file = '/home/setepenre/work/LuaFun/opendota_CM_20210421.zip'

        dataset = Dota2PickBan(draft_file, '7.27')
        optimizer, scheduler = self.init_optim()
        loader = self.init_CM_dataloader(dataset)

        epoch, current_cost = self.resume()
        err = None
        while True:
            epoch += 1
            try:
                prev = current_cost
                role_cost, current_cost = self.train_draft_supervised_epoch(epoch, loader, optimizer, scheduler)

                grad = sum([
                    abs(p.grad.abs().mean().item()) for p in self.drafter.parameters() if p.grad is not None
                ])

                values = [
                    epoch,
                    current_cost,
                    role_cost,
                    grad,
                    current_cost - prev
                ]
                print(', '.join([str(v) for v in values]))

                if grad < 1e-5 or abs(current_cost - prev) < 1e-5:
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

            _, predicted = torch.max(win_predict, 1)
            acc = (predicted == win_target).sum()

            count += 1
            total_cost += cost.item()
            total_acc += acc.item()

        scheduler.step()
        return role_cost, total_cost / count, total_acc / len(loader.dataset)

    def train_draft_judge_supervised(self, epochs):
        draft_file = '/home/setepenre/work/LuaFun/drafting_all.zip'

        dataset = Dota2Matchup(draft_file)
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
                role_cost, current_cost, total_acc = self.train_draft_judge_supervised_epoch(epoch, loader, optimizer, scheduler)

                grad = sum([
                    abs(p.grad.abs().mean().item()) for p in self.judge.parameters() if p.grad is not None
                ])

                values = [
                    epoch,
                    current_cost,
                    role_cost,
                    grad,
                    current_cost - prev,
                    total_acc
                ]
                print(', '.join([str(v) for v in values]))

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

        self.resume()

        while True:

            for _ in range(32):
                cost, total_reward = self.train_draft_rl_selfplay_episode(env, optimizer, scheduler)
                print(cost, total_reward)

            self.train_draft_judge_supervised(2)

    def train_draft_rl_selfplay_episode(self, env, optimizer, scheduler):
        state, reward, done, info = env.reset()

        rnn_state = None
        optimizer.zero_grad()

        advantage = torch.zeros(25, 2)
        logprobs = torch.zeros(25, 2)
        entropy = torch.zeros(25, 2)
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
            advantage[i, 0] = reward[0] - value[0]
            logprobs[i, 0] = pick_logprobs[0] + ban_logprobs[0]
            entropy[i, 0] = pick_entropy[0] + ban_entropy[0]

            # dire
            advantage[i, 1] = reward[1] - value[1]
            logprobs[i, 1] = pick_logprobs[1] + ban_logprobs[1]
            entropy[i, 1] = pick_entropy[1] + ban_entropy[1]
            i += 1

            total_reward += sum(reward)

        advantage = advantage.cuda()
        logprobs = logprobs.cuda()
        entropy = entropy.cuda()

        with torch.no_grad():
            radiant_draft_state = state[0].unsqueeze(0).cuda()
            last_reward = self.judge(radiant_draft_state)

            advantage[i - 1, 0] += (last_reward[0, 0] > 0.50) * 1

            # dire
            advantage[i - 1, 1] += (last_reward[0, 1] > 0.50) * 5

            total_reward += last_reward.sum().item()

        value_loss = advantage.pow(2).mean()
        action_loss = -(advantage.detach() * logprobs).mean()

        loss = (value_loss + action_loss - entropy.mean())
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
    # t.init_optim()
    # t.train_draft_rl_selfplay()
    # t.train_draft_judge_supervised()



