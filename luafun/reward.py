
Solo = 0
Team = 1

#     Name                 , Reward    , Type
values = [
    ('Win'                 ,    5      , Team),
    ('Hero Death'          ,   -1      , Solo),
    ('Courier Death'       ,   -2      , Team),
    ('XP Gained'           ,    0.002  , Solo),
    ('Gold Gained'         ,    0.006  , Solo),
    ('Gold Spent'          ,    0.0006 , Solo),
    ('Health Changed'      ,    2      , Solo), # %of health = ( x + 1 - (1 - x) ^ 4) / 2
    ('Mana Changed'        ,    0.75   , Solo),
    ('Killed Hero'         ,   -0.6    , Solo),
    ('Last Hit'            ,   -0.16   , Solo), # Reduce the reward since we already got a bit chunk from exp & gold
    ('Deny'                ,    0.15   , Solo),
    ('Gained Aegis'        ,    5      , Team),
    ('Ancient HP Change'   ,    5      , Team),
    ('Megas Unlocked'      ,    4      , Team),
    ('T1 Tower'            ,    2.25   , Team),
    ('T2 Tower'            ,    3      , Team),
    ('T3 Tower'            ,    4.5    , Team),
    ('T4 Tower'            ,    2.25   , Team), # 2/3 = building health + 1/3 on destroy
    ('Outpost'             ,    2.25   , Team),
    ('Barracks'            ,    6      , Team), # 2/3 = building health + 1/3 on destroy
    ('Lane Assign'         ,   -0.15   , Solo),
]


class Reward:
    def __init__(self):
        self.constant = {
            k: v for k, v, _ in values
        }

    def reward(self, state):
        return 0

    def __call__(self, ally, enemy):
        return self.reward(ally) - self.reward(enemy)
