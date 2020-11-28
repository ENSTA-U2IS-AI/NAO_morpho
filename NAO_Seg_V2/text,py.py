
used = [0] * (5+ 2)
print(used)
depth=4
down_layer = [i for i in range(depth)]
up_layer = [i for i in range(depth, depth * 2)]
for _,i in enumerate(down_layer):
    print(i)
# for _,i in enumerate(up_layer):
#     print(i)
# print(down_layer)
# print(up_layer)