import torch.nn as nn

import luafun.game.constants as const


class HeroRole(nn.Module):
    def __init__(self, hero_encoder):
        super(HeroRole, self).__init__()
        self.role_count = len(const.ROLES)

        self.hero_encoder = hero_encoder
        self.hero_role_decoder = nn.Linear(hero_encoder.out_size, self.role_count, bias=False)

    def forward(self, x):
        x = self.hero_encoder(x)
        return self.hero_role_decoder(x)
