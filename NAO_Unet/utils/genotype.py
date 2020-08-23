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

fixed_arch = "0 3 1 3 1 1 0 3 0 3 0 2 3 7 2 6 1 3 0 1 1 10 0 7 1 10 1 10 1 10 1 10 0 7 3 7 2 8 0 8 "


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