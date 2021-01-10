

nodes = 5
used = [0] * (nodes + 2)
arch = [0, 1, 1, 2, 2, 4, 2, 5, 3, 6, 0, 7, 2, 4, 2, 1, 4, 0, 1, 0]
for i in range(nodes):
    x_id, x_op, y_id, y_op = arch[4 * i], arch[4 * i + 1], arch[4 * i + 2], arch[4 * i + 3]
    used[x_id] += 1
    used[y_id] += 1

print(used)
concat = [i for i in range(nodes + 2) if used[i] == 0]
print(concat)
for i,_ in enumerate(concat):
    print(_)