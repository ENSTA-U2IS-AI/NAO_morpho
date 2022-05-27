import sys
# from utils import genotype as gt
import genotype as gt
import os
from graphviz import Digraph

os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'
def plot(genotype, filepath,format='png'):
    """visualizer the down cell, up cell and the segmentation network"""
    edge_info = {
        'fontsize':'25',
        'fontname':'times'
    }
    node_info = {
        'style': 'filled',
        'shape': 'rect',
        'align': 'center',
        'fontsize': '25',
        'height': '0.5',
        'width': '0.5',
        'penwidth': '2',
        'fontname': 'times'
    }
    graph_info = Digraph(
        format=format,
        edge_attr=edge_info,
        node_attr=node_info,
        engine='dot')
    # graph_info.body.extend(['randkdir=LR'])
    graph_info.graph_attr.update({'rankdir': 'LR'})
    # graph_info.graph_attr.update({'rankdir': 'TB'})

    #input nodes info
    graph_info.node("c_{k-2}", fillcolor='darkseagreen2')
    graph_info.node("c_{k-1}", fillcolor='darkseagreen2')

    #intermediate nodes info
    assert len(genotype) % 4 == 0
    steps = len(genotype) // 4

    for i in range(steps):
        graph_info.node(str(i), fillcolor='lightblue')

    for i in range(steps):
        # for k in [2*i, 2*i+1]:
        node_prev = genotype[4*i]
        if node_prev == 0:
            u = 'c_{k-2}'
        elif node_prev == 1:
            u = 'c_{k-1}'
        else:
            u = str(node_prev-2)
        v = str(i)
        graph_info.edge(u, v, label=gt.Ops[genotype[4*i+1]], fillcolor='gray')

        node_prev = genotype[4*i+2]
        if node_prev == 0:
            u = 'c_{k-2}'
        elif node_prev == 1:
            u = 'c_{k-1}'
        else:
            u = str(node_prev-2)
        v = str(i)
        graph_info.edge(u, v, label=gt.Ops[genotype[4*i+3]], fillcolor='gray')

    graph_info.node('c_{k}', fillcolor='palegoldenrod')
    for i in range(steps):
        graph_info.edge(str(i), 'c_{k}', fillcolor='gray')

    graph_info.render(filepath, view=True)

def create_exp_dir(path, desc='Experiment dir: {}'):
    if not os.path.exists(path):
        os.makedirs(path)
    print(desc.format(path))

def main(format='png'):

    genotype_name = 'VisualCells'

    store_path = './cell_visualize/' + '/{}'.format(genotype_name)
    create_exp_dir(store_path)

    arch = gt.parse_seq_to_arch(gt.fixed_arch)
    Normal_arch = arch[0]
    Reduce_arch = arch[1]
    print(Normal_arch)
    print(Reduce_arch)

    plot(Normal_arch, store_path + '/NormalCell', format=format)
    plot(Reduce_arch, store_path + '/ReduceCell', format=format)


if __name__ == '__main__':
    # support {'jpeg', 'png', 'pdf', 'tiff', 'svg', 'bmp'
    # 'tif', 'tiff'}
    main(format='png')