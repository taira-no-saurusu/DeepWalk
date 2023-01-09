import logging
import sys
from io import open
from os import path
from time import time
from glob import glob
from gensim.models import Word2Vec
from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict, Iterable
import random
from random import shuffle
from itertools import product,permutations
from scipy.io import loadmat
from scipy.sparse import issparse
from io import open
import matplotlib.pyplot as plt
import networkx as nx

# 何かを書きました！！！！！！
#眠たい

#インプットするファイル名
INPUT = "p2p-Gnutella08.edgelist"

#アウトプットするファイル名
OUTPUT = "p2p-Gnutella08.txt"
#１ノードあたりに実行するウォーク数
NUMBER_WALKS = 5

#1ウォークあたりの長さ
WALK_LENGTH = 20

#ランダムシード値
SEED = 0

#埋め込み後の各ノードの次元数
REPRESENTATION_SIZE = 2

#skipgramので使われるウィンドウサイズ
WINDOW_SIZE = 5

#並列プロセス数
WORKERS = 1

#グラフクラス、辞書型を継承
class Graph(defaultdict):
    def __init__(self):
        super(Graph,self).__init__(list)

    #全ノードをリストで返す
    def nodes(self):
        return self.keys()

    #ノードv1-v2間にエッジが存在するかを判定
    def has_edge(self, v1, v2):
        if v2 in self[v1] or v1 in self[v2]:
            return True
        return False

    #整える
    def make_consistent(self):
       
        for k in iterkeys(self):
            self[k] = list(sorted(set(self[k])))
    
        self.remove_self_loops()

        return self

    #ループを消去する
    def remove_self_loops(self):

        removed = 0

        for x in self:
            if x in self[x]:
                self[x].remove(x)
                removed += 1

        return self

    def make_undirected(self):

        for v in list(self):
            for other in self[v]:
                 if v != other:
                    self[other].append(v)

    
    """ ランダムウォークの一つのウォーク単体を返す

        path_length  : ランダムウォークの長さ
        alpha        : リスタートする確率
        start        : ランダムウォークを始めるノード
    """
    def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):

        G = self
        if start:
            path = [start]
        else:

            # ランダムにグラフからノードを一つ取得
            path = [rand.choice(list(G.keys()))]

        while len(path) < path_length:
            cur = path[-1]
            if len(G[cur]) > 0:
                if rand.random() >= alpha:
                    path.append(rand.choice(G[cur]))
                else:
                    path.append(path[0])
            else:
                break
        return [str(node) for node in path]


"""エッジリストからグラフインスタンス生成

    file_       : ファイル名
    undirected  : 無向グラフかどうか
"""
def load_edgelist(file_, undirected=True):
        G = Graph()
        with open(file_) as f:
            for l in f:
                x, y = l.strip().split()
                x = int(x)
                y = int(y)
                G[x].append(y)
                if undirected:
                    G[y].append(x)
        
        G.make_consistent()
        return G   

"""
networkxからグラフクラス生成
"""
def from_networkx(G_input, undirected=True):
    G = Graph()

    for idx, x in enumerate(G_input.nodes()):
        for y in iterkeys(G_input[x]):
            G[x].append(y)

    if undirected:
        G.make_undirected()

    return G
"""
networkxのGraphクラスからkarateclubを描画すると共に正解ラベルとしてラベルのリストを返す
"""
def draw_karateclub(G):
    pos = nx.spring_layout(G)

    color_list = [0 if G.nodes[i]["club"] == "Mr. Hi" else 1 for i in G.nodes()]
    # 色別に描画
    nx.draw_networkx(G, pos, node_color=color_list, cmap=plt.cm.RdYlBu)
    plt.show()
    return color_list

"""
埋め込まれたデータの可視化
Gx : networkxのGraphクラス
"""
def draw_embedded_vector(Gx ,wv):

    x = list()
    y = list()
    node_list = list()
    colors = list()
    fig, ax = plt.subplots()
    for node in Gx.nodes:
        # int型のままではイテレートできないので、string型に変換する
        vector = wv[str(node)]
        x.append(vector[0])
        y.append(vector[1])
        # 注釈として、ノードの番号を追記する
        # 座標(x,y)は(vector[0],vector[1])を指定
        ax.annotate(str(node), (vector[0], vector[1]))
        if Gx.nodes[node]["club"] == "Officer":
            colors.append("b")
        else:
            colors.append("r")
    for i in range(len(x)):
        ax.scatter(x[i], y[i], c=colors[i])
    plt.show()
"""
クラスタリング結果を可視化
点の色はクラスタごとで異なる
"""
def draw_cluseter(Gx,wv,pred):
    x = list()
    y = list()
    node_list = list()
    colors = list()
    fig, ax = plt.subplots()
    for node in Gx.nodes:
        # int型のままではイテレートできないので、string型に変換する
        vector = wv[str(node)]
        x.append(vector[0])
        y.append(vector[1])
        # 注釈として、ノードの番号を追記する
        # 座標(x,y)は(vector[0],vector[1])を指定
        ax.annotate(str(node), (vector[0], vector[1]))
        if pred[node]==0:
            colors.append("b")
            
        elif pred[node] == 1:
            colors.append("r")

        elif pred[node] == 2:
            colors.append("y")
            
        elif pred[node] == 3:
            colors.append("g")
            
    for i in range(len(x)):
        ax.scatter(x[i],y[i],c=colors[i])
    plt.show


"""
ウォークを実行する
G : Graphインスタンス
num_paths   : １ノードあたり実行するウォーク数
path_length : １ウォークあたりのウォークの長さ
alpha       : 
start       : ウォークを開始するノードの設定?

"""
def build_deepwalk_corpus(G, num_paths, path_length, alpha=0,
                          rand=random.Random(0)):
  
    walks = []

    #グラフの全ノードインデックスをリストにして取得
    nodes = list(G.nodes())

    for cnt in range(num_paths):
        #ノードリストをシャッフル、これをした方が結果がよくなるらしい
        rand.shuffle(nodes)
        for node in nodes:
            walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))

    return walks




#メインクラス
def main():
    """
    グラフ生成
    G = load_edgelist(ファイル名:String, 無効グラフかどうか:boolean)
    """
    G = load_edgelist(INPUT,True)
    # ノード数出力
    print("Number of nodes: {}".format(len(G.nodes())))

    #ウォーク数算出(ノード数*1ノードあたりのウォーク数)と出力
    num_walks = len(G.nodes()) * NUMBER_WALKS
    print("Number of walks: {}".format(num_walks))

    #データサイズ(ウォーク数*ウォークの長さ)の算出と出力
    data_size = num_walks * WALK_LENGTH
    print("Data size (walks*length): {}".format(data_size))

    print("Walking Now!!!!!!")

    #ウォーク取得(長さは全部等しい)
    walks = build_deepwalk_corpus(G, num_paths=NUMBER_WALKS,
                                        path_length=WALK_LENGTH, alpha=0, rand=random.Random(SEED))

    print("Embedding Now!!!!!")
    model = Word2Vec(walks, size = REPRESENTATION_SIZE,window = WINDOW_SIZE, min_count = 0, sg = 1, hs = 1,workers = WORKERS)
    
    model.wv.save_word2vec_format(OUTPUT)

    vec = model.wv.__getitem__([str(i) for i in G.nodes()])

    print(len(vec))

    



    

#mainクラスの実行
if __name__ == "__main__":
  sys.exit(main())