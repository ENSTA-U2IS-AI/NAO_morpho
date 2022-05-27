import numpy as np

Ops =    {
    #down ops,normal ops,up ops
            0:'sep_3×3',
            1:'sep_5×5',
            2:'avg',
            3:'max',
            4:'identity',
            5:'mor_3×3'

}

#with mor ops
fixed_arch = "0 4 0 0 0 1 0 4 0 5 3 1 2 1 0 5 3 4 0 5 0 2 0 2 2 4 2 2 3 5 0 5 2 5 4 5 0 2 2 2 "

def parse_seq_to_arch(arch,nodes=5):
    if isinstance(arch, str):
        arch = list(map(int, arch.strip().split()))
    elif isinstance(arch, list) and len(arch) == 2:
        arch = arch[0] + arch[1]
    Normal_arch = arch[:4 * nodes]  # every cell contain 4 nodes
    Down_arch = arch[4 * nodes:]  # every cell contain 4 nodes
    return [Normal_arch,Down_arch]


if __name__ == '__main__':
    arch = parse_seq_to_arch(fixed_arch)
    print(arch[0])
