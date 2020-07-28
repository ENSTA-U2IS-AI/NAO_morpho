import numpy as np

DownOps = [
            'avg_pool',
            'max_pool',
            'down_cweight_3×3',
            'down_conv_3×3',
            'pix_shuf_pool'
]

NormalOps = [
            'identity',
            'cweight_3×3',
            'conv_3×3',
            'pix_shuf_3×3'
]

UpOps = [
            'up_cweight_3×3',
            'up_conv_3×3'
]
nodes = 4
used = [0] * (nodes + 2)
print(used)
used[2]+=1
print(used)