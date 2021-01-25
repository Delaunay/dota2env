import json


with open('exporting_trees.json', 'r') as f:
    data = f.readlines()


trees = []

for line in data:
    tid, x, y, z = json.loads(line)

    if x == 0 and y == 0 and z == 0:
        continue

    trees.append((tid, x, y, z))

with open('trees.json', 'w') as f:
    json.dump(trees, f)

print(len(trees))
