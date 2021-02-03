import json
import os


TEAM_RADIANT = 2
TEAM_DIRE = 3


def get_draft_message(draft):
    from luafun.game.action import DraftAction

    return {
        DraftAction.EnableDraft: draft,
        DraftAction.SelectHero: None,
        DraftAction.BanHero: None,
        DraftAction.Lane: None
    }


def new_ipc_message(draft=None):
    """Basic ipc message we can send to the bots

    Parameters
    ----------
    draft: Optional[int]
        0, or 1, 0 to disable ML drafting and fallback to hardcoded hero selection.
    """
    base = {
        'uid': 0,
        TEAM_RADIANT: {
            0: dict(),
            1: dict(),
            2: dict(),
            3: dict(),
            4: dict(),
        },
        TEAM_DIRE: {
            5: dict(),
            6: dict(),
            7: dict(),
            8: dict(),
            9: dict(),
        }
    }

    if draft is not None:
        base[TEAM_RADIANT]['HS'] = get_draft_message(draft)
        base[TEAM_DIRE]['HS'] = get_draft_message(draft)

    return base


def ipc_send(f2, data, uid):
    """Write a lua file with the data we want bots to receive"""
    f1 = f2 + '_tmp'

    # Remove old file so we can override it
    while True:
        try:
            if os.path.exists(f2):
                os.remove(f2)
            break
        except PermissionError:
            pass
    # --

    # Keep track of the message id we are sending
    uid.value += 1
    data['uid'] = uid.value
    json_string = json.dumps(data, separators=(',', ':'))

    with open(f1, 'w') as file:
        file.write(f'return \'{json_string}\'')

    # Renaming is almost always atomic
    os.rename(f1, f2)
