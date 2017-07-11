import getopt
from networkx.drawing.nx_agraph import *
import networkx as nx
import matplotlib.pyplot as plt
import sys
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout

def keywithmaxval(d):
    """ a) create a list of the dict's keys and values;
         b) return the key with the max value"""
    v = list(d.values())
    k = list(d.keys())
    return k[v.index(max(v))]

def keywithminval(d):
    """ a) create a list of the dict's keys and values;
         b) return the key with the max value"""
    v = list(d.values())
    k = list(d.keys())
    return k[v.index(min(v))]


class HyG:
    def __init__(self, edge, nv, bd=0, fname='',primal=0):
        self.dg = nx.DiGraph()
        self.g = nx.Graph()
        self.n = list()
        self.e = list()
        self.edge = dict()
        self.de = set()
        if primal:
            self.primal=nx.Graph()
        self.bd = nx.Graph(root=0, fitness=list())
        for i in range(nv):
            self.g.add_node(i, type='n')
            if primal:
                self.primal.add_node(i)
            self.n.append(i)
        for i in range(len(edge)):
            self.g.add_node(nv + i, type='e')
            self.e.append(nv + i)
            self.edge[nv + i] = list()
            for j in edge[i]:
                self.g.add_edge(nv + i, j, type=j)
                self.edge[nv + i].append(j)
            self.edge[nv + i].sort()
            if primal:
                for j in range(len(edge[i])):
                    for k in range(j+1,len(edge[i])):
                        self.primal.add_edge(edge[i][j],edge[i][k])
        # print self.n
        # print self.e
        # print self.edge
        # show_graph(self.g)
        if bd == 1:
            self.bd = nx.Graph(root=0)
            self.bd.add_node(0, label=list())
            for i in range(1, len(edge) + 1):
                self.bd.add_node(i, label=list(edge[i - 1]))
                temp_list = list()
                for v in edge[i - 1]:
                    if self.g.degree(v) > 1:
                        temp_list.append(v + 1)
                self.bd.add_edge(0, i, label=sorted(temp_list))
                # show_graph(self.bd)
        if bd == 2:
            try:
                self.bd = nx.read_edgelist(fname + '.decomp', edgetype=int, nodetype=int)
                self.bd.graph['root'] = 0
                self.bd.graph['fitness'] = [0 for x in xrange(len(self.n))]
                fit = self.bd.graph['fitness']
                for e in self.bd.edges():
                    fit[self.bd.edge[e[0]][e[1]]['weight'] - 1] += 1
                print self.bd.graph['fitness']
                # print self.bd.edges()
                # print self.g.edges()
            except Exception as e1:
                print "exception reading file"
                pass

    def add_node(self, n):
        try:
            self.g.add_node(n, type='n')
        except Exception as e:
            print e
            self.g.add_node(self.g.number_of_nodes(), type='n')
        return self

    def add_edge(self, e):
        if type(e) != list:
            raise Exception("edge must be a list")
        self.g.add_node(self.g.number_of_nodes(), type='e')
        for i in e:
            if i not in self.g.nodes():
                print(str(i) + " not in graph")
                self.g.add_node(i, type='n')
            self.g.add_edge(self.g.number_of_nodes() - 1, i, type='e')
        return self

    def remove_node(self,n):
        if n not in self.n:
            print(str(n)+" is not a node")
        try:
            self.g.remove_node(n)
        except Exception as e:
            print e,
            print "could not remove",n
        return self

    def degree(self, n=-1):
        if n == -1:
            deg = dict()
            for i in self.n:
                deg[i] = self.g.degree(i)
            return deg
        else:
            if n not in self.n:
                print n, " is not a node"
            else:
                return self.g.degree(n)

    def edges(self, n=list()):
        edg = list()
        # print n
        if len(n) == 0:
            for i in self.n:
                for j in self.g.edges(i):
                    if sorted(self.g.edge[j[1]].keys()) not in edg:
                        edg.append(sorted(self.g.edge[j[1]].keys()))
                        # print "1e",len(edg),edg
        if len(n) == 1:
            for j in self.g.edges(n):
                if sorted(self.g.edge[j[1]].keys()) not in edg:
                    edg.append(sorted(self.g.edge[j[1]].keys()))
                    # print "2e",edg,n
        if len(n) > 1:
            for k in n:
                for j in self.g.edges(k):
                    if sorted(self.g.edge[j[1]].keys()) not in edg:
                        edg.append(sorted(self.g.edge[j[1]].keys()))
                        # print "3e",edg, n
        return edg

    def neighbour(self, n):
        if n not in self.n:
            print(str(n) + " not a node")
            return -1
        else:
            nl = list()
            # print n,self.edges(n)
            for j in self.edges([n]):
                if j not in nl:
                    nl.append(list(set(j) - {n})[0])
            # print nl
            return nl

    def leaf(self, deg=1):
        l = list()
        for i in self.n:
            if self.g.degree(i) == deg:
                l.append(i)
        return l

    def make_flow_network(self):
        # show_graph(self.g)
        # show_graph(self.dg)
        for n in self.g.nodes():
            # print self.g.node[n]
            if self.g.node[n]['type'] == 'n':
                self.dg.add_node(n + 1)
                self.dg.add_node(n + self.g.number_of_nodes() + 1)
                self.dg.add_edge(n + 1, n + self.g.number_of_nodes() + 1, label=n, capacity=1)
            else:
                self.dg.add_node(n + 1)
                self.de.update({n + 1})
                for e in self.g.edges(n, data=True):
                    self.dg.add_edge(n + 1, e[1] + 1, capacity=float("inf"))
                    self.dg.add_edge(e[1] + self.g.number_of_nodes() + 1, n + 1, capacity=float("inf"))
        # print self.de
        # show_graph(self.dg, 1)
        return 0

    # def divide_b(self, A, B):
    #     print "in divide_B"
    #     self.bd.add_node(self.bd.number_of_nodes(), label=set())
    #     ed = list()
    #     for i in range(len(B)):
    #         ed.append(self.bd.edges()[B[i]][1])
    #     for i in range(len(ed)):
    #         self.bd.add_edge(self.bd.number_of_nodes() - 1, ed[i],
    #                          label=self.bd.edge[self.bd.graph['root']][ed[i]]['label'])
    #         self.bd.remove_edge(self.bd.graph['root'], ed[i])
    #         # show_graph(g)
    #     ed = list()
    #     for e in self.bd.edges(self.bd.number_of_nodes() - 1, data=True):
    #         for ele in e[2]['label']:
    #             for e1 in self.bd.edges(self.bd.graph['root'], data=True):
    #                 # print ele, e, e1
    #                 if ele in e1[2]['label']:
    #                     ed.append(ele)
    #     self.bd.node[self.bd.number_of_nodes() - 1]['label'] = set(ed)
    #     self.bd.add_edge(self.bd.number_of_nodes() - 1, self.bd.graph['root'], label=set(ed))
    #     # print ed
    #     # show_graph(self.bd, 1)

    def reduction(self,pri=0):
        while 1 <= min(self.primal.degree().values()) <= 2:
            while min(self.primal.degree().values()) == 1:
                node = keywithminval(self.primal.degree())
                # print "removed node", node
                self.primal.remove_node(node)
            while min(self.primal.degree().values()) == 2:
                node = keywithminval(self.primal.degree())
                neigh = list(self.primal.neighbors(node))
                # print "removed node", node, neigh
                self.primal.add_edge(neigh[0], neigh[1])
                self.primal.remove_node(node)
            # show_graph(self.primal, 1)
        if pri == 2:
            while len(self.n)>1 and 1<=min(self.degree().values())<=2:
                while len(self.n)>1 and min(self.degree().values()) == 1:
                    node=keywithminval(self.degree())
                    # print "removed node", node
                    neigh=self.g.neighbors(node)
                    self.remove_node(node)
                    self.n.remove(node)
                    # print self.edge[neigh[0]]
                    if self.g.degree(neigh[0])==0:
                        self.remove_node(neigh[0])
                        self.e.remove(neigh[0])
                        del self.edge[neigh[0]]
                while len(self.n)>1 and min(self.degree().values()) == 2:
                    node=keywithminval(self.degree())
                    neigh=self.g.neighbors(node)
                    neigh1=set(self.g.neighbors(neigh[1]))
                    # print "removed node", node
                    for n in neigh1:
                        self.g.add_edge(neigh[0],n)
                    self.g.remove_node(neigh[1])
                    self.g.remove_node(node)
                    self.n.remove(node)
                    del self.edge[neigh[1]]
                # show_graph(self.g,1)




    def divide_a(self, A, B, part):
        g1 = self.dg.copy()
        g2 = self.dg.copy()
        print part, B
        g1.remove_nodes_from(part[1])
        # show_graph(g1, 1)
        # print "g1", g1.nodes()
        g2.remove_nodes_from(part[0])
        # show_graph(g2, 1)
        # print "g2", g2.nodes()
        cuts = nx.get_edge_attributes(self.bd, 'label')
        print "cuts: ", cuts
        print "e:", self.de
        self.bd.add_node(self.bd.number_of_nodes(), label=list())
        self.bd.add_edge(self.bd.graph['root'], self.bd.number_of_nodes() - 1, label=list())
        side_edge = list()
        # show_graph(self.bd,1)
        if len(set(g1.nodes()).intersection(set(self.de))) == len(self.de):
            print "first if"
            # print set(g1.nodes()).intersection(set(self.de))
            for i in self.de:
                if len(g1.in_edges(i)) != len(self.dg.in_edges(i)) or len(g1.out_edges(i)) != len(self.dg.out_edges(i)):
                    # print i, g1.in_edges(i), self.dg.in_edges(i), g1.out_edges(i), self.dg.out_edges(i)
                    side_edge.append(i)
        elif len(set(g2.nodes()).intersection(set(self.de))) == len(self.de):
            print "second else"
            # print set(g2.nodes()).intersection(set(self.de))
            for i in self.de:
                if len(g2.in_edges(i)) != len(self.dg.in_edges(i)) or len(g2.out_edges(i)) != len(self.dg.out_edges(i)):
                    # print i, g2.in_edges(i), self.dg.in_edges(i), g2.out_edges(i), self.dg.out_edges(i)
                    side_edge.append(i)
        else:
            print "last else"
            side_edge = list(set(g1.nodes()).intersection(set(self.de)))
        print side_edge, self.de
        if len(side_edge) == 1:
            print "in side_edge==1",
            e = side_edge[0]
            v1 = self.dg.out_edges(e)[0][1]
            v2 = self.dg.out_edges(e)[1][1]
            print self.dg.in_edges(v1),
            print self.dg.in_edges(v2)
        for i in side_edge:
            print "edges", self.dg.out_edges(i)[0][1], self.dg.out_edges(i)[1][1],
            for e in [key for key, value in cuts.items() if
                      value == sorted([self.dg.out_edges(i)[0][1], self.dg.out_edges(i)[1][1]])]:
                if e[0] == self.bd.graph['root']:
                    v1 = e[1]
                else:
                    v1 = e[0]
                print v1
                self.bd.add_edge(v1, self.bd.number_of_nodes() - 1, label=self.bd.edge[e[0]][e[1]]['label'])
                self.bd.remove_edge(e[0], e[1])
        cut1 = set()
        cut2 = set()
        # print self.bd.edges(self.bd.number_of_nodes() - 1, data=True)
        # print self.bd.edges(self.bd.graph['root'], data=True)
        for e in self.bd.edges(self.bd.number_of_nodes() - 1, data=True):
            # print "in for1", cut1, e, self.bd.edge[e[0]][e[1]]['label']
            cut1.update(set(self.bd.edge[e[0]][e[1]]['label']))
        for e in self.bd.edges(self.bd.graph['root'], data=True):
            # print "in for2", cut2, e, self.bd.edge[e[0]][e[1]]['label']
            cut2.update(set(self.bd.edge[e[0]][e[1]]['label']))
        # print cut1, cut2
        self.bd.edge[self.bd.graph['root']][self.bd.number_of_nodes() - 1]['label'] = list(cut1.intersection(cut2))
        show_graph(self.bd, 1)

    def make_st(self, A, B):
        self.make_flow_network()
        # show_graph(self.dg)
        self.dg.add_node(0, nodetype=int)
        self.dg.add_node(self.dg.number_of_nodes(), nodetype=int)
        # show_graph(self.dg)
        for i in range(len(A)):
            anode = list(self.bd.nodes(data=True)[self.bd.edges(data=True)[A[i]][1]][1]['label'])
            bnode = list(self.bd.nodes(data=True)[self.bd.edges(data=True)[B[i]][1]][1]['label'])
            print "anode", anode, bnode
            while anode:
                v = anode.pop()
                self.dg.add_edge(0, v + 1, capacity=float("inf"))
            while bnode:
                v = bnode.pop()
                self.dg.add_edge(v + 1 + self.g.number_of_nodes(), self.dg.number_of_nodes() - 1, capacity=float("inf"))
        # show_graph(self.dg,1)
        cut_value, part = nx.minimum_cut(self.dg, 0, self.dg.number_of_nodes() - 1)
        # print part
        cog = self.dg.copy()
        # show_graph(dg)
        cog.remove_nodes_from(part[0])
        # show_graph(cog)
        print cog.number_of_edges(), len(B)
        self.divide_a(A, B, part)
        # if cog.number_of_edges() == len(B)+1:
        #
        # else:
        #     print "in else"
        #     self.divide_a(A, B,part)
        return self.bd

    def case_split(self):
        node = self.bd.graph['root']
        leaf = dict()
        for e in self.bd.edges(node):
            if (self.bd.degree(e[1]) != 1):
                leaf[e] = self.bd.edge[e[0]][e[1]]['label'].sort()
            else:
                v = e[1]
                mid = self.bd.edge[e[0]][e[1]]['label']
        one_list = list()
        for e in leaf.keys():
            if set(leaf[e]).issubset(mid):
                one_list.append(e)
        while one_list:
            curr_edge = one_list.pop()
            curr_node = int(self.bd.number_of_nodes())
            self.bd.add_node(curr_node, nodetype=int, label=list())
            lab = set
            for e in leaf.keys():
                if e[1] != curr_edge[1] and e[1] != v:
                    self.bd.add_edge(curr_node, e[1], label=leaf[e])
                    lab |= set(leaf[e])
                    self.bd.remove_edge(e[1], node)
            self.bd.add_edge(node, curr_node, label=list(lab))
            v = node
            node = curr_node
            mid = list(lab)
            self.bd.graph['root'] = curr_node
        if nx.is_biconnected(self.g):
            ga = nx.Graph()
            nodes = set()
            for e in self.bd.edges(node):
                if self.bd.degree(e[1]) == 1:
                    nodes |= set(self.bd.edge[e[0]][e[1]]['label'])
                    nodes |= {self.edge.keys()[self.edge.values().index(self.bd.edge[e[0]][e[1]]['label'].sort())]}
            for i in range(len(nodes)):
                nodes[i] = nodes[1] - 1
            ga = self.g.subgraph(list(nodes))
            for n in ga.nodes():
                if ga.degree(n) == 1 and n + 1 not in mid:
                    e = ga.edges(n)
                    alert = 0
                    curr_edge1 = list()
                    curr_edge1.append(n)
                    for e1 in ga.edges(e[1]):
                        curr_edge1.append(e1[1])
                        if e1[1] in mid:
                            alert = 1
                    if alert != 1:
                        curr_edge = leaf.keys()[leaf.values().index((curr_edge1.sort()))]
                        curr_node = int(self.bd.number_of_nodes())
                        self.bd.add_node(curr_node, nodetype=int, label=list())
                        lab = set
                        for e in leaf.keys():
                            if e[1] != curr_edge[1] and e[1] != v:
                                self.bd.add_edge(curr_node, e[1], label=leaf[e])
                                lab |= set(leaf[e])
                                self.bd.remove_edge(e[1], node)
                        self.bd.add_edge(node, curr_node, label=list(lab))
                        v = node
                        node = curr_node
                        mid = list(lab)
                        self.bd.graph['root'] = curr_node
        ga = nx.Graph()
        nodes = set()
        for e in self.bd.edges(node):
            if self.bd.degree(e[1]) == 1:
                nodes |= set(self.bd.edge[e[0]][e[1]]['label'])
                nodes |= {self.edge.keys()[self.edge.values().index(self.bd.edge[e[0]][e[1]]['label'].sort())]}
        for i in range(len(nodes)):
            nodes[i] = nodes[1] - 1
        ga = self.g.subgraph(list(nodes))
        if nx.is_connected(ga):
            return False
        else:
            comp = nx.connected_components(ga)


class Ga:
    def __init__(self, node, g, og):
        self.node = node
        self.g = nx.Graph()
        self.s = list()
        for i in range(len(g.edges(node))):
            self.g.add_edge(node, g.edges(node)[i][1], label=g.edges(0, data=True)[i][2]['label'])
            self.s |= g.edges(0, data=True)[i][2]['label']
        print self.s
        self.N = dict()
        for v1 in self.s:
            v = int(v1)
            self.N[v] = list()
        for v1 in self.s:
            v = int(v1)
            for e in og.edges(v):
                # print e
                self.N[v].append(e)
        # noinspection PyUnusedLocal
        self.l = [[0.0 for i1 in xrange(self.g.number_of_edges())] for j in xrange(self.g.number_of_edges())]
        for i in range(self.g.number_of_edges()):
            for j in range(self.g.number_of_edges()):
                for k in self.s:
                    if i == j and k in self.g.edges(data=True)[j][2]['label']:
                        self.l[i][j] += 1
                    elif i != j:
                        if k in self.g.edges(data=True)[j][2]['label'] and k in self.g.edges(data=True)[i][2]['label']:
                            self.l[i][j] += -1 / float(len(self.N[k]) - 1)
        for i in range(self.g.number_of_edges()):
            d = 0.0
            for j in range(self.g.number_of_edges()):
                d += self.l[i][j]
            if d != 0:
                print "some error"


class HyGa:
    def __init__(self, hg):
        self.node = hg.bd.graph['root']
        self.g = nx.Graph()
        self.hg = hg.g.copy()
        self.s = set()
        for i in range(len(hg.bd.edges(self.node))):
            self.g.add_edge(self.node, hg.bd.edges(self.node)[i][1],
                            label=hg.bd.edges(self.node, data=True)[i][2]['label'])
            self.s |= set(hg.bd.edges(self.node, data=True)[i][2]['label'])
        print "SELF.S", self.s
        # show_graph(self.g,1)
        self.N = dict()
        for v1 in self.s:
            v = int(v1)
            self.N[v] = list()
        for v1 in self.s:
            v = int(v1) - 1
            for e in hg.g.edges(v):
                # print e, v,hg.g.edges(v)
                self.N[v + 1].append(e)
        print self.N
        show_graph(hg.g, 1)
        # noinspection PyUnusedLocal
        self.l = [[0.0 for i1 in xrange(self.g.number_of_edges())] for j in xrange(self.g.number_of_edges())]
        for i in range(self.g.number_of_edges()):
            for j in range(self.g.number_of_edges()):
                for k in self.s:
                    if i == j and k in self.g.edges(data=True)[i][2]['label']:
                        self.l[i][j] += 1
                        # print k,self.g.edges(data=True)[i][2]['label'],self.g.edges(data=True)[j][2]['label']
                    elif i != j:
                        if k in self.g.edges(data=True)[i][2]['label'] and k in self.g.edges(data=True)[j][2]['label']:
                            self.l[i][j] += -1 / float(len(self.N[k]) - 1)
                            # print k,self.g.edges(data=True)[i][2]['label'],self.g.edges(data=True)[j][2]['label']
        for i in range(self.g.number_of_edges()):
            d = 0.0
            for j in range(self.g.number_of_edges()):
                d += self.l[i][j]
            if d != 0:
                print "some error", i, j


def read_hyper_graph(fname):
    with open(fname + '.edge', 'r') as solution_file:
        l3 = solution_file.readline()
        l3 = l3.replace('p edge', '')
        l3 = l3.replace('\n', '')
        l3 = l3.split()
        temp_nv = int(l3[0])
        ed = list()
        temp_ne = int(l3[1])
        for i1 in range(1, temp_ne + 1):
            l3 = solution_file.readline()
            if l3[0]!='e':
                break
            # print l3,
            l3 = l3.replace('e ', '')
            l3 = l3.split()
            temp_ed = list()
            for i in l3:
                temp_ed.append(int(i) - 1)
            ed.append(temp_ed)
            # temp_og.add_edge(int(l3[0]), int(l3[1]))
        if len(ed)!=temp_ne:
            print "edges missing:",temp_ne,len(ed)
    return ed, temp_nv


def show_graph(graph, layout, nolabel=0):
    """ show graph
    layout 1:graphviz,
    2:circular,
    3:spring,
    4:spectral,
    5: random,
    6: shell
    """

    m = graph.copy()
    if layout == 1:
        pos = graphviz_layout(m)
    elif layout == 2:
        pos = nx.circular_layout(m)
    elif layout == 3:
        pos = nx.spring_layout(m)
    elif layout == 4:
        pos = nx.spectral_layout(m)
    elif layout == 5:
        pos = nx.random_layout(m)
    elif layout == 6:
        pos = nx.shell_layout(m)
    if not nolabel:
        nx.draw_networkx_edge_labels(m, pos)
    nx.draw_networkx_labels(m, pos)
    nx.draw_networkx_nodes(m, pos)
    write_dot(m, "m1.dot")
    # os.system("dot -Tps m1.dot -o m1.ps")
    nx.draw(m, pos)
    plt.show()


def compute_fitness(fit, c=2):
    tc = 1
    sum1 = 0
    for i in range(len(fit)):
        sum1 += fit[i] * tc
        tc *= c
    return sum1


def write_graph(m, name):
    write_dot(m, name + '.dot')
    # os.system("dot -Tps " + name + ".dot -o " + name + ".ps")


def is_branch_decomposition(graph):
    set_degree = set(graph.degree().values())
    if set_degree != {1, 3}:
        return 0
    if nx.is_connected(graph) != True and nx.cycle_basis != []:
        return 0
    for node in graph.nodes():
        neigh = graph.neighbors(node)
        if len(neigh) == 3:
            cut1 = set(graph.edge[node][neigh[0]]['label'])
            cut2 = set(graph.edge[node][neigh[1]]['label'])
            cut3 = set(graph.edge[node][neigh[2]]['label'])
            if cut1.issubset(cut2 | cut3) == False or cut3.issubset(cut2 | cut1) == False or cut2.issubset(
                            cut1 | cut3) == False:
                print "cuts don't match", cut1, cut2, cut3
                return 0
    return 1


def reduction(g):
    while g.number_of_nodes()>=1 and min(g.degree().values()) <= 2:
        while g.number_of_nodes()>=1 and min(g.degree().values()) == 0:
            node = keywithminval(g.degree())
            # print "removed node", node
            g.remove_node(node)
        while g.number_of_nodes()>=1 and min(g.degree().values()) == 1:
            node = keywithminval(g.degree())
            # print "removed node", node
            g.remove_node(node)
        while g.number_of_nodes()>=1 and min(g.degree().values()) == 2:
            node = keywithminval(g.degree())
            neigh = list(g.neighbors(node))
            # print "removed node", node, neigh
            g.add_edge(neigh[0], neigh[1])
            g.remove_node(node)
    g=nx.convert_node_labels_to_integers(g,first_label=0)


def main(argv):
    global file_name, delta
    try:
        opts, args = getopt.getopt(argv, "hi:d:", ["ifile=", "delta="])
    except getopt.GetoptError:
        print './bwheu.py -i <input_file(without_extension)>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print './bwheu.py -i <input_file(without_extension)>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            file_name = arg
        elif opt in ("-d", "--delta"):
            delta = int(arg)
    print 'Input file is ', file_name


file_name = "c5"
delta = 0.3
if __name__ == "__main__":
    main(sys.argv[1:])


