from luafun.dotaenv import action_space, Dota2Env


class MockState:
    def get_entities(self, x, y):
        return None, None, None


class MockDraftTracker:
    def __init__(self):
        pass

    def update(self, x, y):
        return


class MockEnv:
    def __init__(self):
        self.draft_tracker = MockDraftTracker()
        self.heroes = {
            0: {'hid': 1},
            1: {'hid': 1},
            2: {'hid': 1},
            3: {'hid': 1},
            4: {'hid': 1},
            5: {'hid': 1},
            6: {'hid': 1},
            7: {'hid': 1},
            8: {'hid': 1},
            9: {'hid': 1},
        }

    def dire_state(self):
        return MockState()

    def radiant_state(self):
        return MockState()


def dump(act):
    for k, f in act.items():
        if k == 'uid':
            continue

        for p, v in f.items():
            print(f'{k} {p} {v}')


def preprocessor_test():
    act = action_space().sample()
    dump(act)
    result = Dota2Env._action_preprocessor(MockEnv(), act)
    dump(result)


if __name__ == '__main__':
    preprocessor_test()
