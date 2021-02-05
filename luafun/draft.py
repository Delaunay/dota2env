from dataclasses import dataclass
from enum import IntEnum, auto

import torch

from luafun.game.action import DraftAction
import luafun.game.constants as const
from luafun.utils.python_fix import asdict
from luafun.game.ipc_send import TEAM_RADIANT, TEAM_DIRE


class DraftFields(IntEnum):
    """Draft field enumeration used to map fields to tensor index"""
    Pick0 = 0
    Pick1 = auto()
    Pick2 = auto()
    Pick3 = auto()
    Pick4 = auto()
    Pick5 = auto()
    Pick6 = auto()
    Pick7 = auto()
    Pick8 = auto()
    Pick9 = auto()
    Ban01 = auto()
    Ban02 = auto()
    Ban03 = auto()
    Ban04 = auto()
    Ban05 = auto()
    Ban06 = auto()
    Ban07 = auto()
    Ban08 = auto()
    Ban09 = auto()
    Ban10 = auto()
    Ban11 = auto()
    Ban12 = auto()
    Ban13 = auto()
    Ban14 = auto()
    Size = auto()


DIRE_DRAFT_REMAP = {
    DraftFields.Pick0: DraftFields.Pick5,
    DraftFields.Pick1: DraftFields.Pick6,
    DraftFields.Pick2: DraftFields.Pick7,
    DraftFields.Pick3: DraftFields.Pick8,
    DraftFields.Pick4: DraftFields.Pick9,
    # --
    DraftFields.Pick5: DraftFields.Pick0,
    DraftFields.Pick6: DraftFields.Pick1,
    DraftFields.Pick7: DraftFields.Pick2,
    DraftFields.Pick8: DraftFields.Pick3,
    DraftFields.Pick9: DraftFields.Pick4,
}


@dataclass
class DraftStatus:
    """Draft struct which represent the drafting state of the game"""
    # Picks are first because it will always be 10 picks
    # putting bans last makes it easy to grow/shrink the struct
    Pick0: int = -1
    Pick1: int = -1
    Pick2: int = -1
    Pick3: int = -1
    Pick4: int = -1
    Pick5: int = -1
    Pick6: int = -1
    Pick7: int = -1
    Pick8: int = -1
    Pick9: int = -1
    # Captains mode has 14 bans
    # Ranked all pick has 5 bans max (10 bans with a 50% of being banned)
    Ban01: int = -1
    Ban02: int = -1
    Ban03: int = -1
    Ban04: int = -1
    Ban05: int = -1
    Ban06: int = -1
    Ban07: int = -1
    Ban08: int = -1
    Ban09: int = -1
    Ban10: int = -1
    Ban11: int = -1
    Ban12: int = -1
    Ban13: int = -1
    Ban14: int = -1

    def as_tensor(self, faction):
        """Generate a one-hot encoded tensor for the ban/picks.
        Ally team is first, then enemies and bans are last

        Examples
        --------
        >>> draft = DraftStatus()
        >>> draft.Pick4 = 1
        >>> draft.Ban01 = 2
        >>> draft.Pick5 = 3
        >>> tensor = draft.as_tensor(TEAM_RADIANT)
        >>> tensor[DraftFields.Pick4][:5]
        tensor([0., 1., 0., 0., 0.])
        >>> tensor[DraftFields.Ban01][:5]
        tensor([0., 0., 1., 0., 0.])
        >>> tensor[DraftFields.Pick5][:5]
        tensor([0., 0., 0., 1., 0.])

        >>> draft = DraftStatus()
        >>> draft.Pick4 = 1
        >>> draft.Ban01 = 2
        >>> draft.Pick5 = 3
        >>> tensor = draft.as_tensor(TEAM_DIRE)
        >>> tensor[DraftFields.Pick4][:5]
        tensor([0., 0., 0., 0., 0.])
        >>> tensor[DraftFields.Pick9][:5]
        tensor([0., 1., 0., 0., 0.])
        >>> tensor[DraftFields.Ban01][:5]
        tensor([0., 0., 1., 0., 0.])
        >>> tensor[DraftFields.Pick5][:5]
        tensor([0., 0., 0., 0., 0.])
        >>> tensor[DraftFields.Pick0][:5]
        tensor([0., 0., 0., 1., 0.])

        """
        draft_status = torch.zeros((DraftFields.Size, const.HERO_COUNT))

        remap = dict()
        if faction == TEAM_DIRE:
            remap = DIRE_DRAFT_REMAP

        for k, v in asdict(self).items():
            if v >= 0:
                original = int(getattr(DraftFields, k))
                f = int(remap.get(original, original))
                draft_status[f, v] = 1

        return draft_status


# TODO make sure actions are correct here
# i.e no double hero selection
# respect bans although bots cannot bans in the game
class DraftTracker:
    """Track Bots decision and reflect it on the draft status"""
    def __init__(self):
        self.draft = DraftStatus()
        self.rhero = 0
        self.dhero = 0
        self.bans = 0

    def update(self, radiant, dire):
        # Ignore bans on purpose
        h = radiant.get(DraftAction.SelectHero)
        if h:
            setattr(self.draft, f'Pick{self.rhero}', h)
            self.rhero += 1

        h = dire.get(DraftAction.SelectHero)
        if h:
            setattr(self.draft, f'Pick{self.dhero + 5}', h)
            self.dhero += 1

    def as_tensor(self, faction):
        return self.draft.as_tensor(faction)
