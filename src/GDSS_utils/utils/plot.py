import math
import networkx as nx
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pickle
# import warnings
# warnings.filterwarnings("ignore", category=matplotlib.cbook.MatplotlibDeprecationWarning)

options = {
    'node_size': 0.7,
    'edge_color' : 'black',
    'linewidths': 1,
    'width': 0.3
}

def plot_graphs_list(graphs, title='title', max_num=100, save_dir=None, N=0):
    batch_size = len(graphs)
    max_num = min(batch_size, max_num)
    img_c = int(math.ceil(np.sqrt(max_num)))
    figure = plt.figure()
    st = 90 #400
    title = title + str(st)
    step = 20
    # g_id = [300, 309, 319, 339, 359, 379, 399]
    #
    graphs = graphs[st:st+10]
    #
    if not isinstance(graphs[-1], nx.Graph):
        G = graphs[-1].g.copy()
    else:
        G = graphs[-1].copy()
    assert isinstance(G, nx.Graph)
    # G.remove_nodes_from(list(nx.isolates(G)))
    pos = nx.spring_layout(G, iterations=50)
    degrees = dict(G.degree())
    vmin, vmax = min(degrees.values()), max(degrees.values())
    norm_degrees = [(degree - vmin) / (vmax - vmin) for degree in degrees.values()]

    for i in range(10):
        idx = i + max_num*N
        if not isinstance(graphs[idx], nx.Graph):
            G = graphs[idx].g.copy()
        else:
            G = graphs[idx].copy()
        assert isinstance(G, nx.Graph)
        # G.remove_nodes_from(list(nx.isolates(G)))

        ax = plt.subplot(img_c, img_c, i + 1)
        nx.draw(G, pos, with_labels=False, node_color=norm_degrees,
                cmap=plt.cm.viridis, **options)
    figure.suptitle(title)
    cbar_ax = figure.add_axes([0.005, 0.84, 0.007, 0.06], frameon=False)
    cbar = figure.colorbar(ax.collections[-1], cax=cbar_ax, orientation='vertical', shrink=0.1, )
    cbar.ax.tick_params(axis='both', which='both', length=0, labelsize=3)  # Adjust label size
    cbar.outline.set_visible(False)
    cbar.outline.set_linewidth(0)  # Set outline width to 0
    cbar.set_ticks([0, 1])  # Set ticks at the low and high ends of the colorbar
    cbar.ax.tick_params(axis='y', direction='in', pad=0.25)  # Adjust tick direction


    save_fig(save_dir=save_dir, title=title)


def save_fig(save_dir=None, title='fig', dpi=600):
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, wspace=0.01)
    if save_dir is None:
        plt.show()
    else:
        fig_dir = os.path.join(*['samples', 'fig', save_dir])
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        plt.savefig(os.path.join(fig_dir, title+'.png'),
                    bbox_inches='tight', dpi=dpi, 
                    transparent=False)
        plt.close()
    return


def save_graph_list(log_folder_name, exp_name, gen_graph_list):

    if not(os.path.isdir('./samples/pkl/{}'.format(log_folder_name))):
        os.makedirs(os.path.join('./samples/pkl/{}'.format(log_folder_name)))
    with open('./samples/pkl/{}/{}.pkl'.format(log_folder_name, exp_name), 'wb') as f:
            pickle.dump(obj=gen_graph_list, file=f, protocol=pickle.HIGHEST_PROTOCOL)
    save_dir = './samples/pkl/{}/{}.pkl'.format(log_folder_name, exp_name)
    return save_dir

if __name__ == '__main__':
    graph_ckpt_path = '/media/ubuntu/6EAA3539AA34FEE1/LXY/GraphGeneration/BetaGraph_eta/outputs/gdss-ego/2024-05-21/15-37-23-graph-tf-model/34999.pkl'
    graph_ckpt_name = graph_ckpt_path.split('/')[-1]
    save_path = graph_ckpt_path.split(graph_ckpt_name)[0]
    with open(graph_ckpt_path, 'rb') as f:
        graph_list = pickle.load(f)
    plot_graphs_list(graph_list, save_dir=save_path)


