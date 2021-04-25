from dataclasses import dataclass
from enum import IntEnum, auto
import json
import logging

import torch

from luafun.game.action import DraftAction
import luafun.game.constants as const
from luafun.utils.python_fix import asdict
from luafun.game.ipc_send import TEAM_RADIANT, TEAM_DIRE


log = logging.getLogger(__name__)


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


class DraftTracker:
    """Track Bots decision and reflect it on the draft status

    Examples
    --------
    >>> tracker = DraftTracker()

    This is the kind of message we receive when a ban is successful

    >>> msg = DraftTracker._ban(TEAM_DIRE, 'npc_dota_hero_morphling')
    >>> msg
    {'B': 'npc_dota_hero_morphling', 'S': 1, 'T': 3}
    >>> tracker.update(msg)
    >>> tracker.status
    {2: ['', '', '', '', ''], 3: ['', '', '', '', ''], 'bans': ['npc_dota_hero_morphling']}

    This is the kind of message we receive when a pick is successful

    >>> msg = DraftTracker._pick(TEAM_RADIANT, 'npc_dota_hero_drow_ranger')
    >>> msg
    {'P': 'npc_dota_hero_drow_ranger', 'S': 1, 'T': 2}
    >>> tracker.update(msg)
    >>> tracker.status
    {2: ['npc_dota_hero_drow_ranger', '', '', '', ''], 3: ['', '', '', '', ''], 'bans': ['npc_dota_hero_morphling']}

    To keep track of human picks a summary of the pick state is send everytime
    the draft changes (ban excluded)

    >>> msg = DraftTracker._picks_summary(TEAM_RADIANT, 'npc_dota_hero_drow_ranger', p2='npc_dota_hero_antimage')
    >>> msg
    {'PS': ['npc_dota_hero_drow_ranger', '', 'npc_dota_hero_antimage', '', '', '', '', '', '', ''], 'S': 1, 'T': 2}
    >>> tracker.update(msg)
    >>> tracker.status
    {2: ['npc_dota_hero_drow_ranger', '', 'npc_dota_hero_antimage', '', ''], 3: ['', '', '', '', ''], 'bans': ['npc_dota_hero_morphling']}

    Get a the draft state in a one-hot encoded vector
    >>> tracker.as_tensor(TEAM_RADIANT).shape
    torch.Size([24, 121])
    """
    def __init__(self):
        self.draft = DraftStatus()
        self.rhero = 0
        self.dhero = 0
        self.radiant = ['', '', '', '', '']
        self.dire = ['', '', '', '', '']
        self.bans = []
        self.dire_known = []
        self.radiant_known = []

    @property
    def status(self):
        return {
            TEAM_RADIANT: self.radiant,
            TEAM_DIRE: self.dire,
            'bans': self.bans
        }

    def __str__(self):
        return json.dumps(self.status, indent=2)

    @staticmethod
    def _ban(team, hero):
        return {"B": hero, "S": 1, "T": team}

    @staticmethod
    def _pick(team, hero):
        return {"P": hero, "S": 1, "T": team}

    @staticmethod
    def _picks_summary(team, p0='', p1='', p2='', p3='', p4='', p5='', p6='', p7='', p8='', p9=''):
        return {"PS": [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9], "S": 1, "T": team}

    def name_to_offset(self, hero):
        if isinstance(hero, int):
            return hero

        return const.HERO_LOOKUP.from_name(hero)['offset']

    def pick(self, team, hero):
        if team == TEAM_RADIANT:
            array = self.radiant
            count = self.rhero
            attribute = f'Pick{count}'
            self.rhero += 1
        else:
            array = self.dire
            count = self.dhero
            attribute = f'Pick{count + 5}'
            self.dhero += 1

        array[count] = hero
        hero_id = self.name_to_offset(hero)
        setattr(self.draft, attribute, hero_id)

    def ban(self, team, hero):
        count = len(self.bans) + 1

        attribute = f'Ban{count}'
        if count < 10:
            attribute = f'Ban0{count}'

        self.bans.append(hero)
        hero_id = self.name_to_offset(hero)
        setattr(self.draft, attribute, hero_id)

    def end_draft(self, state):
        print(state)

    def merge(self, ar1, ar2):
        for i in range(len(ar1)):
            v = ar1[i]
            if v == '' and ar2[i] != '':
                v = ar2[i]

            ar1[i] = v

    def update(self, state):
        if not isinstance(state, dict):
            return

        success = state.get('S', 0)
        ban = state.get('B')
        pick = state.get('P')
        team = state.get('T')

        msg = state.get('M')
        if msg:
            log.debug(msg)

        if success == 1 and ban is not None:
            self.ban(team, ban)

        if success == 1 and pick is not None:
            self.pick(team, pick)

        # Human Picks
        # FIXME: the picks are not revealed to the other team right away
        # we need one draft status for each team
        picks = state.get('PS')
        if picks:
            if team == TEAM_RADIANT:
                self.merge(self.radiant, picks[:5])
                self.radiant_known = picks

            if team == TEAM_DIRE:
                self.merge(self.dire, picks[5:])
                self.dire_known = picks

    def as_tensor(self, faction):
        return self.draft.as_tensor(faction)
