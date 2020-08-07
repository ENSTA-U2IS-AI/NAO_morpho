import numpy as np

Ops =    {
    #down ops,normal ops,up ops
            0:'avg_pool',
            1:'max_pool',
            2:'down_cweight_3×3',
            3:'down_conv_3×3',
            4:'pix_shuf_pool',
            5:'identity',
            6:'cweight_3×3',
            7:'conv_3×3',
            8:'pix_shuf_3×3',
            9:'up_cweight_3×3',
            10:'up_conv_3×3'
}

fixed_arch = "0 1 1 1 2 6 2 8 2 5 3 8 3 6 1 4 3 8 3 6 1 9 0 6 1 10 0 7 0 8 3 5 2 8 2 8 1 10 4 6 "


def parse_seq_to_arch(arch,nodes=5):
    if isinstance(arch, str):
        arch = list(map(int, arch.strip().split()))
    elif isinstance(arch, list) and len(arch) == 2:
        arch = arch[0] + arch[1]
    DownCell_arch = arch[:4 * nodes]  # every cell contain 4 nodes
    # print(self.DownCell_arch)
    UpCell_arch = arch[4 * nodes:]  # every cell contain 4 nodes
    # print(self.UpCell_arch)
    return [DownCell_arch,UpCell_arch]

# def parse_arch_to_ops_seq(arch):
#     new_arch = []
#     for i in range(len(arch)):
#         print(str(Ops[arch[i]]))
#     return new_arch

if __name__ == '__main__':
    arch = parse_seq_to_arch(fixed_arch)
    print(arch[0])
    down_cell_seq = parse_arch_to_ops_seq(arch[0])
    print(down_cell_seq)