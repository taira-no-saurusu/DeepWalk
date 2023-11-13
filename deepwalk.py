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
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np
from statistics import stdev  # 標準偏差
from sklearn.decomposition import PCA



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
    def random_walk(self,INPUT, path_length, rand=random.Random(), start=None):
        nx = generate_Graph("karateclub")

        G = self
        if start:
            path = [start]
        else:

            # ランダムにグラフからノードを一つ取得
            path = [rand.choice(list(G.keys()))]

        while len(path) < path_length:
            cur = path[-1]
            if len(G[cur]) > 0:
                # 隣接するノードリスト取得
                adj_list = G[cur]

                #連接するノードリストからそれぞれの重みを取得
                weight_list = G.get_adj_weight(cur, adj_list,INPUT=INPUT)

                #重みを割合に変更
                pro_list = get_pro(weight_list)

                #重みをもとに確立的に次のノードを選択するようにする。
                path.append(adj_list[np.random.choice(len(pro_list),p=pro_list)])
               
            else:
                break

        return [str(node) for node in path]
    
    #現在のノードと隣接するノード間の重みの情報をリストとして返す。
    def get_adj_weight(self, cur, adj_list, INPUT):
        di = dict(INPUT.edges)
        wgt = []
        cur = int(cur)
        for n in adj_list:
            n = int(n)
            small = 0
            big = 0
            if cur>n:
                small = n
                big = cur
            else:
                small = cur
                big = n
            weight = int(di[small, big]["weight"])
            wgt.append(weight)

        return wgt
    
#重みのリストを割合に変更
def get_pro(weight_list):
    su = sum(weight_list)
    pro = [i/su for i in weight_list]
    su = sum(pro)
    if sum != 1.0:
       pro[0]+= 1.0 - su
    return pro

"""
networkxのグラフインスタンスを生成
"""
def generate_Graph(name):
    G = nx.Graph()

    if name=="football":
        f = open('football.txt','r')
        datalist = f.readlines()

        for l in datalist:
            list = l.strip('\n').split(",")
            if list[2]=='0':
                continue
            else:
                G.add_edge(list[0], list[1], weight=int(list[2]))
                G.add_edge(list[1], list[0], weight=int(list[2]))

    elif name == "polbooks":
        f = open('polbooks.txt', 'r')
        datalist = f.readlines()

        for l in datalist:
            list = l.strip('\n').split(",")
            if list[2] == '0':
                continue
            else:
                G.add_edge(list[0], list[1], weight=int(list[2]))
                G.add_edge(list[1], list[0], weight=int(list[2]))

        pass
    elif name == "karateclub":
        G = nx.karate_club_graph()
    else:
        sys.exit("グラフ生成の引数の名前がおかしいです")

    return G

"""
label_listをテキストファイルから読み込んで返す
"""
def get_label_list(name):
    label_list= []
    if name == "football":
        f = open('label_football.txt', 'r')
        datalist = f.readlines()

        for l in datalist:
           label_list.append(int(l.rstrip("\n")))

    elif name == "polbooks":
        f = open('label_polbooks.txt', 'r')
        datalist = f.readlines()

        for l in datalist:
           label_list.append(int(l.rstrip("\n")))

    elif name == "karateclub":
        label_list = get_label_karateclub(False)

    else:
        sys.exit("グラフ生成の引数の名前がおかしいです")

    return label_list



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
networkxからグラフインスタンス生成
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
def get_label_karateclub(is_draw:bool):
    G = nx.karate_club_graph()
    pos = nx.spring_layout(G)

    color_list = [0 if G.nodes[i]["club"] == "Mr. Hi" else 1 for i in G.nodes()]
    if is_draw:
        # 色別に描画
        nx.draw_networkx(G, pos, node_color=color_list, cmap=plt.cm.RdYlBu)
    plt.show()
    return color_list


"""
ベクトルデータを描画
Y : ベクトルデータ
colorlist : ラベルリスト
"""
def draw_embedded_vector(Y,colorlist):
    if colorlist is None:
        colorlist = get_label_karateclub(False)

    fig, ax = plt.subplots()
    for i in range(len(colorlist)):
        ax.annotate(str(i), (Y[i, 0], Y[i, 1]))
        if colorlist[i] == 0:
            ax.scatter(Y[i, 0], Y[i, 1], c="r")
            pass
        elif colorlist[i] == 1:
            ax.scatter(Y[i, 0], Y[i, 1], c="b")
            pass
        elif colorlist[i] == 2:
            ax.scatter(Y[i, 0], Y[i, 1], c="y")

        elif colorlist[i] == 3:
            ax.scatter(Y[i, 0], Y[i, 1], c="g")

        elif colorlist[i] == 4:
            ax.scatter(Y[i, 0], Y[i, 1], c="c")

        elif colorlist[i] == 5:
            ax.scatter(Y[i, 0], Y[i, 1], c="m")

        elif colorlist[i] ==6:
            ax.scatter(Y[i, 0], Y[i, 1], c="coral")

        elif colorlist[i] == 7:
            ax.scatter(Y[i, 0], Y[i, 1], c="lightpink")

        elif colorlist[i] == 8:
            ax.scatter(Y[i, 0], Y[i, 1], c="skyblue")

        elif colorlist[i] == 9:
            ax.scatter(Y[i, 0], Y[i, 1], c="darkgray")

        elif colorlist[i] == 10:
            ax.scatter(Y[i, 0], Y[i, 1], c="goldenrod")

        elif colorlist[i] == 11:
            ax.scatter(Y[i, 0], Y[i, 1], c="lightgreen")

    plt.show()


"""
ウォークを実行する
G : Graphインスタンス
NUMBER_WALKS   : １ノードあたり実行するウォークの回数
WALK_LENGTH : １ウォークあたりのウォークの長さ
alpha       : 
start       : ウォークを開始するノードの設定?

"""
def build_deepwalk_corpus(INPUT,G, NUMBER_WALKS, WALK_LENGTH,
                          rand=random.Random()):
  
    walks = []

    #グラフの全ノードインデックスをリストにして取得
    nodes = list(G.nodes())

    for cnt in range(NUMBER_WALKS):
        #ノードリストをシャッフル、これをした方が結果がよくなるらしい
        rand.shuffle(nodes)
        for node in nodes:
            walks.append(G.random_walk(
                INPUT,WALK_LENGTH, rand=rand, start=node))

    return walks


"""
取得したコーパスから元のネットワークのエッジの重みの総和を求める
"""
def get_corpas_weight(INPUT, walks):
    w_list = dict(INPUT.edges)
    val = 0
    for walk in walks:
        for i in range(len(walk)-1):
            big = None
            small = None
            if int(walk[i]) >= int(walk[i+1]):
                big = int(walk[i])
                small = int(walk[i+1])
            else:
                big = int(walk[i+1])
                small = int(walk[i])

            val += int(w_list[small, big]["weight"])

    return val

"""
埋め込みを実施
"""
def embed(INPUT, UNDIRECTED, NUMBER_WALKS, WALK_LENGTH, REPRESENTATION_SIZE, WINDOW_SIZE, WORKERS):
    G = from_networkx(INPUT, UNDIRECTED)
    # ウォーク取得(長さは全部等しい)
    walks = build_deepwalk_corpus(INPUT,G, NUMBER_WALKS=NUMBER_WALKS, WALK_LENGTH=WALK_LENGTH, rand=random.Random())

    model = Word2Vec(walks, size=REPRESENTATION_SIZE,window=WINDOW_SIZE, min_count=0, sg=1, hs=1, workers=WORKERS)
    vec = model.wv.__getitem__([str(i) for i in range(len(G))])
    return vec,walks

"""
クラスタリングを実施
"""
def clustering(vec,N_CLUSTERS,METHOD):

    pred = None

    if METHOD == "kmedoids":
        pred = KMedoids(n_clusters=N_CLUSTERS).fit_predict(vec)

    elif METHOD == "kmeans":
        pred = KMeans(n_clusters = N_CLUSTERS).fit_predict(vec)

    else:
        sys.exit(f"クラスタリング手法が無効です：{METHOD}")
    
    return pred


"""
リストの中に含まれる tar の添字をリストで返
READ=TRUEの場合 +１して標準出力に対応
"""
def get_index(mask, tar, READ=False):
    if READ:
        return [i+1 for i, x in enumerate(mask) if x == tar]
    else:
        return [i for i, x in enumerate(mask) if x == tar]


"""
埋め込みからクラスタリングを実施
    INPUT               : networkXのGraphインスタンス
    IS_DIRECTED         : 無向グラフ
    NUMBER_WALKS        : 1ノードあたりに実行するウォークの回数
    WALK_LENGTH         : 1ウォークあたりの長さ
    REPRESENTATION_SIZE : 埋め込み後の次元数
    WORKERS             : 並列プロセス数
    N_CLUSTERS          : クラスタリングのクラスタ数

"""
def exection(INPUT, UNDIRECTED, NUMBER_WALKS, WALK_LENGTH, REPRESENTATION_SIZE, WINDOW_SIZE, WORKERS, N_CLUSTER,METHOD,TRUE_LABEL,SHOW=True):

    #埋め込みを実施
    vec ,walks= embed(INPUT,UNDIRECTED, NUMBER_WALKS, WALK_LENGTH,REPRESENTATION_SIZE, WINDOW_SIZE, WORKERS)

    # クラスタリングを実施
    pred = clustering(vec, N_CLUSTER, METHOD)

    if SHOW:
        #埋め込み結果を可視化
        print("＝＝＝＝＝＝＝＝埋め込み結果（正解ラベルに基づいて色付け）＝＝＝＝＝＝＝＝＝")
        draw_embedded_vector(vec,TRUE_LABEL)
        #クラスタリング結果に基づいて可視化
        print("＝＝＝＝＝＝＝＝＝＝＝＝＝＝クラスタリング結果＝＝＝＝＝＝＝＝＝＝＝＝＝＝")
        draw_embedded_vector(vec,pred)

    #ARI算出
    ari = adjusted_rand_score(TRUE_LABEL, pred)

    return vec,pred,ari,walks

"""
埋め込みからクラスタリング、ARI算出まで複数回実行
埋め込みからクラスタリングは、このメソッドからexectionメソッドを呼び出して実行
    INPUT               : networkXのGraphインスタンス
    IS_DIRECTED         : 無向グラフ
    NUMBER_WALKS        : 1ノードあたりに実行するウォークの回数
    WALK_LENGTH         : 1ウォークあたりの長さ
    REPRESENTATION_SIZE : 埋め込み後の次元数
    WINDOW_SIZE         : skipgram学習時のウィンドウサイズ
    WORKERS             : 並列プロセス数
    N_CLUSTERS          : クラスタリングのクラスタ数
    METHOD              : クラスタリング手法（k-means,k-medoids）
    TRUE_LABEL          : 正解ラベル
    SHOW                : 埋め込み図を表示するかどうか
"""
def multi_exection(TIME, INPUT, UNDIRECTED, NUMBER_WALKS, WALK_LENGTH, REPRESENTATION_SIZE, WINDOW_SIZE, WORKERS, N_CLUSTER, METHOD, TRUE_LABEL,SHOW):

    ARI_list = []
    max_vec = None
    min_vec = None
    max_pred = None
    min_pred = None
    max_ari = -100
    min_ari = 100
    max_walks = None
    min_walks = None
    for i in range(TIME):
        if SHOW:
            print(f"------------------------{i+1}回目実行-----------------------------")

        vec, pred, ari, walks = exec_100(INPUT, UNDIRECTED, NUMBER_WALKS, WALK_LENGTH,REPRESENTATION_SIZE, WINDOW_SIZE, WORKERS, N_CLUSTER, METHOD, TRUE_LABEL,SHOW)
        
        if SHOW:
            print(f"ari : {ari}")

        ARI_list.append(ari)

        if ari > max_ari:
            max_ari = ari
            max_vec = vec
            max_pred = pred
            max_walks = walks

        if min_ari > ari:
            min_ari = ari
            min_vec = vec
            min_pred = pred
            min_walks = walks
    
    if SHOW:
        # 埋め込み結果を可視化
        print("＝＝＝＝＝＝＝＝最大ARI埋め込み結果（正解ラベルに基づいて色付け）＝＝＝＝＝＝＝＝＝")
        draw_embedded_vector(max_vec, TRUE_LABEL)
        # クラスタリング結果に基づいて可視化
        print("＝＝＝＝＝＝＝＝＝＝＝＝＝＝最大ARIクラスタリング結果＝＝＝＝＝＝＝＝＝＝＝＝＝＝")
        draw_embedded_vector(max_vec, max_pred)

    
    print(f"最大ARI({get_index(ARI_list,max_ari,READ = True)}回目実行) : {max_ari}")
    print(f"最小ARI({get_index(ARI_list,min_ari,READ = True)}回目実行) : {min_ari}")
    print(f"平均ARI : {np.mean(ARI_list)}")
    print(f"標準偏差：{stdev(ARI_list)}")

    return ARI_list, max_walks, min_walks, max_vec, min_vec, max_pred, min_pred


"""
    一つの埋め込みからクラスタリングを100回実施し最も高いariの値を返す
    INPUT               : networkXのGraphインスタンス
    IS_DIRECTED         : 無向グラフ
    NUMBER_WALKS        : 1ノードあたりに実行するウォークの回数
    WALK_LENGTH         : 1ウォークあたりの長さ
    REPRESENTATION_SIZE : 埋め込み後の次元数
    WORKERS             : 並列プロセス数
    N_CLUSTERS          : クラスタリングのクラスタ数

"""
def exec_100(INPUT, UNDIRECTED, NUMBER_WALKS, WALK_LENGTH, REPRESENTATION_SIZE, WINDOW_SIZE, WORKERS, N_CLUSTER, METHOD, TRUE_LABEL, SHOW):

    #ariを格納 最終的に一番高いariを返す
    max_ari  = -100
    max_pred = []

    # 埋め込みを実施
    vec, walks = embed(INPUT, UNDIRECTED, NUMBER_WALKS,
                       WALK_LENGTH, REPRESENTATION_SIZE, WINDOW_SIZE, WORKERS)
    
    for i in range(100):
        # クラスタリングを実施
        pred = clustering(vec, N_CLUSTER, METHOD)
        # ARI算出
        ari = adjusted_rand_score(TRUE_LABEL, pred)

        if(max_ari<ari):
            max_ari = ari
            max_pred = pred

    
    if SHOW:
        # 埋め込み結果を可視化
        print("＝＝＝＝＝＝＝＝埋め込み結果（正解ラベルに基づいて色付け）＝＝＝＝＝＝＝＝＝")
        draw_embedded_vector(vec, TRUE_LABEL)
        # クラスタリング結果に基づいて可視化
        print("＝＝＝＝＝＝＝＝＝＝＝＝＝＝クラスタリング結果＝＝＝＝＝＝＝＝＝＝＝＝＝＝")
        draw_embedded_vector(vec, max_pred)
    


    return vec, max_pred, max_ari, walks



def get_network_weight(TIME1, TIME2, INPUT, UNDIRECTED, NUMBER_WALKS, WALK_LENGTH, REPRESENTATION_SIZE, WINDOW_SIZE, WORKERS, N_CLUSTER, METHOD, TRUE_LABEL,SHOW):
    ARI_list_all = []
    max_weight_list = []
    min_weight_list = []
    max_ari = []
    min_ari = []
    for i in range(TIME2):
        ARI_list, max_walks, min_walks, _, _, _, _ = multi_exection(TIME1, INPUT, UNDIRECTED, NUMBER_WALKS, WALK_LENGTH, REPRESENTATION_SIZE, WINDOW_SIZE, WORKERS, N_CLUSTER, METHOD, TRUE_LABEL, SHOW=False)
        ARI_list_all += ARI_list
        max_ari.append(max(ARI_list))
        min_ari.append(min(ARI_list))
        max_weight_list.append(get_corpas_weight(INPUT,max_walks))
        min_weight_list.append(get_corpas_weight(INPUT,min_walks))
    
    print(f"総実行回数：{TIME1*TIME2}")
    print(f"ARI平均 : {np.mean(ARI_list_all)}")
    print(f"最大ARI平均 : {np.mean(max_ari)}")
    print(f"最小ARI平均 : {np.mean(min_ari)}")
    print(f"ARIが良かった時の重み平均 : {np.mean(max_weight_list)}")
    print(f"ARIが悪かった時の重み平均 : {np.mean(min_weight_list)}")
    return ARI_list_all, max_weight_list, min_weight_list

#埋め込みを100回実行して、pcaを実行し寄与率の平均を返す
def get_contribution(INPUT, UNDIRECTED, NUMBER_WALKS, WALK_LENGTH, REPRESENTATION_SIZE, WINDOW_SIZE, WORKERS):
    raitoList = []
    max_vec = None
    max_ari = -100
    for i in range(100):
        vec, _ = embed(INPUT, UNDIRECTED, NUMBER_WALKS,
                           WALK_LENGTH, REPRESENTATION_SIZE, WINDOW_SIZE, WORKERS)
        #主成分分析
        pca = PCA()
        pca.fit(vec)
        raitoList.append(pca.explained_variance_ratio_)

    #平均寄与率を求める    
    mean_contribution = calculate_mean_contributions(raitoList)
    #四捨五入
    rounded_mean_contribution = convert_to_float_array(mean_contribution)


    #ARIが一番高い時の埋め込みベクトルの寄与率を返す
    return mean_contribution, rounded_mean_contribution


"""
指数表現の数値が格納されたリストを少数第5位で四捨五入したfloatのリストに変換する
"""
def convert_to_float_array(array):
    float_array = []
    for i in range(len(array)):
        float_array.append(round(array[i], 5))
    return float_array


"""
寄与率の平均を計算する
"""
def calculate_mean_contributions(data_contributions):
    data_contributions = np.array(data_contributions)
    mean_contributions = np.mean(data_contributions, axis=0)
    return mean_contributions.tolist()



#メインクラス
def main():
    pass
    

#mainクラスの実行
if __name__ == "__main__":
  sys.exit(main())