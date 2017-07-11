#!/usr/bin/env python2.7
# coding=utf-8
import getopt
import logging
import math
import os
import random
import sys

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout, write_dot


# from datetime import datetime
# import matplotlib.pyplot as plt

# d657 krb200 krb100 lin105 pcb442


def dfs_component(u1, v1, max_dist1, max_cut, cut, cuts, max_cut_list, file_name, leaf_edges, internal_nodes,
                  cuts_at_cut_edges, vertices_in_cut_edges, g2, y, connecting_edges, g, u, v, print_in_file=0):
    """returns bounded size dfs result for local improvemnt"""
    dist = []
    # if i == 30:
    #     nx.write_edgelist(g, file_name + str(limit_on_size) + 'g.decomp')
    # write_graph(g, file_name + str(limit_on_size) + 'g')
    for x in xrange(g.number_of_nodes()):
        dist.append(-1)
    dist[u1] = 0
    dist[v1] = 0
    cut_list = []
    for x in xrange(g.number_of_nodes()):
        cut_list.append(max_cut)
    cut_list[u1] = cut[(u1, v1)]
    cut_list[v1] = cut[(u1, v1)]
    boundary_nodes = []
    dfs_list = [u1, v1]
    while dfs_list:
        v1 = dfs_list.pop(0)
        neigh = list(g.neighbors(v1))
        neigh1 = list(g.neighbors(v1))
        while neigh:
            w = neigh.pop()
            if dist[w] == -1:
                dist[w] = dist[v1] + 1
                if cut[ord1(v1, w)] < cut_list[w]:
                    cut_list[w] = cut[ord1(v1, w)]
                if cut_list[w] == max_cut and dist[v1] == 0:
                    dist[w] = 0
                    if cuts[ord1(v1, w)] in max_cut_list:
                        max_cut_list.remove(cuts[ord1(v1, w)])
                if cut_list[w] > cut_list[v1]:
                    boundary_nodes.append(v1)
                    for rem in neigh1:
                        if rem in neigh:
                            neigh.remove(rem)
                        if (dist[rem] == dist[v1] + 1) or dist[rem] == -1:
                            if rem in boundary_nodes:
                                boundary_nodes.remove(rem)
                            if rem in dfs_list:
                                dfs_list.remove(rem)
                    break
                if dist[w] == max_dist1 or (g.degree(w) == 1 and dist[w] <= max_dist1):
                    if w not in boundary_nodes:
                        boundary_nodes.append(w)
                elif w not in dfs_list:
                    if cut_list[w] <= cut_list[v1]:
                        dfs_list.append(w)
        internal_nodes.append(v1)
    nodes_to_delete = list(set(boundary_nodes) | set(internal_nodes))
    component_size = []
    for ele in [i2 for i2, value in enumerate(dist) if value == 0]:
        component_size.append(ele)
    y = g.subgraph(nodes_to_delete)
    leaf_edges = []
    connecting_edges = []
    cuts_at_cut_edges = []
    vertices_in_cut_edges = []
    for leaf in [i1 for i1, value in enumerate(dict(y.degree()).values()) if value == 2]:
        print y.degree().keys()[leaf]
    for leaf in [i1 for i1, value in enumerate(dict(y.degree()).values()) if value == 1]:
        leaves = dict(y.degree()).keys()[leaf]
        leaf_edges.append((leaves, list(y.neighbors(leaves))[0]))
        cuts_at_cut_edges.append(y.edge[leaves][list(y.neighbors(leaves))[0]]['label'])
        for a in y.edge[leaves][list(y.neighbors(leaves))[0]]['label']:
            if a not in vertices_in_cut_edges:
                vertices_in_cut_edges.append(a)
        if g.degree(leaves) > 1:
            connecting_edges.append((leaves, list(y.neighbors(leaves))[0]))
            if leaves in nodes_to_delete:
                nodes_to_delete.remove(leaves)
    if print_in_file == 1:
        y_file = open('component.csv', mode='a')
        y_file.write(
            '{0} {1} {2} {3} {4} {5} {6}\n'.format(file_name, str(u), str(v), str(max_cut), str(y.number_of_nodes()),
                                                   str(max_dist1), str(len(component_size))))
        y_file.close()
    g2.remove_nodes_from(nodes_to_delete)
    return u1, v1, max_dist1, max_cut, cut, cuts, max_cut_list, file_name, leaf_edges, internal_nodes, cuts_at_cut_edges, vertices_in_cut_edges, g2, y, connecting_edges, g, u, v
    # write_graph(y, file_name + str(limit_on_size) + 'y')
    # write_graph(g2, file_name + str(limit_on_size) + 'g2')


def ord1(x, w):
    if x < w:
        return x, w
    elif w < x:
        return w, x
    else:
        print "error"


def print_graph(width, file_name, limit_on_size, vertices_in_cut_edges, cuts_at_cut_edges, time_out, max_cut, i,
                original_graph):
    print file_name
    edge_file = open('{0}{1}.edge'.format(file_name, str(limit_on_size)), 'w')
    edge_file.write('p edge ' + str(len(vertices_in_cut_edges)) + ' ' + str(len(cuts_at_cut_edges)) + '\n')
    vertices_in_cut_edges.sort()
    for cut_edge in cuts_at_cut_edges:
        edge_file.write('e ')
        for vertex in cut_edge:
            for index in (index1 for index1, element in enumerate(vertices_in_cut_edges) if element == vertex):
                edge_file.write(str(index + 1) + ' ')
        edge_file.write('\n')
    edge_file.close()
    new_cut = sorted(cuts_at_cut_edges, key=len, reverse=True)
    print "new_cut[0]", cuts_at_cut_edges
    cmd_make = 'mkfifo ' + file_name + str(limit_on_size) + '.fifo'
    depth = int((int(len(cuts_at_cut_edges) / 2) - int(round(width / len(new_cut[0]), 0)) + int(
        round(math.log(int((width / len(new_cut[0])))), 0))) * 0.6)
    cmd = 'timeout {7} cat {1}{2}.edge | ./hybw2sat {0} {5}{6}.fifo {8} &'.format(str(width), file_name,
                                                                                  str(limit_on_size), file_name,
                                                                                  str(limit_on_size), file_name,
                                                                                  str(limit_on_size), str(time_out),
                                                                                  str(depth))
    print cmd
    cmd2 = './glucose -verb=0 -cpu-lim=' + str(time_out) + ' ' + file_name + str(
        limit_on_size) + '.fifo ' + file_name + str(limit_on_size) + str(width) + '.sol'
    print cmd2
    cmd_remove = 'rm ' + file_name + str(limit_on_size) + '.fifo'
    os.system(cmd_make)
    os.system(cmd)
    os.system(cmd2)
    os.system(cmd_remove)
    sys.stdout.flush()
    while max_cut > width >= len(new_cut[0]):
        try:
            fil = open(file_name + str(limit_on_size) + str(width) + '.sol')
            s = fil.read(1)
            print s[0], width
            fil.close()
        except Exception as ex1:
            print ex1
            s = 'I'
        while (s[0] != 'I' and s[0] != "U") and max_cut > width >= len(new_cut[0]):
            width -= 1
            cmd = 'timeout {7} cat {1}{2}.edge | ./hybw2sat {0} {5}{6}.fifo {8} &'.format(str(width), file_name,
                                                                                          str(limit_on_size),
                                                                                          file_name,
                                                                                          str(limit_on_size),
                                                                                          file_name,
                                                                                          str(limit_on_size),
                                                                                          str(time_out), str(depth))
            print cmd
            cmd2 = './glucose -verb=0 -cpu-lim=' + str(time_out) + ' ' + file_name + str(
                limit_on_size) + '.fifo ' + file_name + str(limit_on_size) + str(width) + '.sol'
            print cmd2
            os.system(cmd_make)
            os.system(cmd)
            os.system(cmd2)
            os.system(cmd_remove)
            sys.stdout.flush()
            try:
                fil = open(file_name + str(limit_on_size) + str(width) + '.sol')
                s = fil.read(1)
                print s[0], width
                fil.close()
            except Exception as ex1:
                print ex1
                s = 'I'
            if s == 'I' or s[0] == 'U' and max_cut > width:
                width += 1
                try:
                    fil = open(file_name + str(limit_on_size) + str(width) + '.sol')
                    s = fil.read(1)
                    print s[0], width
                    fil.close()
                except Exception as ex1:
                    print ex1
                    s = 'I'
                if (s != 'I' and s[0] != "U") and max_cut > width:
                    os.system(cmd_remove)
                    return 0, width, depth
                elif max_cut > width:
                    width += 1
                    try:
                        fil = open(file_name + str(limit_on_size) + str(width) + '.sol')
                        s = fil.read(1)
                        print s[0], width
                        fil.close()
                    except Exception as ex1:
                        print ex1
                        s = 'I'
                    if (s[0] != 'U' and s[0] != 'I') and max_cut > width:
                        os.system(cmd_remove)
                        return 0, width, depth
                    else:
                        return 1, width, depth
                else:
                    return 1, width, depth
            else:
                return 1, width, depth
        else:
            os.system(cmd_remove)
            return 1, width, depth
    else:
        os.system(cmd_remove)
        return 1, width, depth


# noinspection PyTypeChecker,PyUnusedLocal
def read_sol(u, v, depth, current_width, file_name, limit_on_size):
    print current_width
    solution_file = open(file_name + str(limit_on_size) + str(current_width) + '.sol')
    l3 = solution_file.read()
    l4 = l3.replace('\n', ' ')
    l3 = l4.replace('v', '')
    l4 = l3.split()
    # l4.pop()
    # l4.pop(0)
    # l4.pop(0)
    solution_file.close()
    (edge, nv, ne) = read_adjency(file_name + str(limit_on_size))
    ng = 0
    nsteps = depth
    arc = [[[0 for i1 in xrange(ne + 1)] for j in xrange(ne + 1)] for k in xrange(ne + 1)]
    ver = [[[0 for i1 in xrange(ne + 1)] for j in xrange(nv + 1)] for k in xrange(ne + 1)]
    led = [[0 for i1 in xrange(ne + 1)] for j in xrange(ne + 1)]
    ctr = [[[[0 for i1 in xrange(ne + 1)] for m in xrange(current_width + 1)] for j in xrange(nv + 1)] for k in
           xrange(ne + 1)]
    for i1 in range(1, nsteps + 1):
        for u1 in range(1, ne + 1):
            for v1 in range(u1 + 1, ne + 1):
                arc[u1][v1][i1] = ng + 1
                # print "arc[",u1,"][",v1,"][",i1,"]:",arc[u1][v1][i1]
                ng += 1
        for u1 in range(1, ne + 1):
            led[u1][i1] = ng + 1
            # print "led[",u1,"][",i1,"]",led[u1][i1]
            ng += 1
        for u1 in range(1, ne + 1):
            for v1 in range(1, nv + 1):
                ver[u1][v1][i1] = ng + 1
                # print "ver[",u1,"][",v1,"][",i1,"]:",ver[u1][v1][i1]
                ng += 1
        for u1 in range(1, ne + 1):
            for v1 in range(1, nv + 1):
                for j in range(1, current_width + 1):
                    ctr[u1][v1][j][i1] = ng + 1
                    # print "ctr[",u1,"][",v1,"][",j,"][",i1,"]:",ctr[u1][v1][j][i1]
                    ng += 1
    return nv, ne, nsteps, l4, edge, arc, led, ver, ctr


# returns a component list Independent
def make_component(nsteps, ne, l4, led, arc):
    c = []
    for i1 in range(1, nsteps + 1):
        a = []
        for i2 in xrange(ne + 1):
            a.append(0)
        for u1 in range(1, ne + 1):
            if int(l4[led[u1][i1] - 1]) > 0:
                for v1 in range(u1 + 1, ne + 1):
                    if int(l4[arc[u1][v1][i1] - 1]) > 0:
                        a[v1] = u1
        c.append(a)
    return c


def remove_extra_vertices(g1):
    for i3 in [i4 for i4, value in enumerate(dict(g1.degree()).values()) if value == 2]:
        la = []
        i2 = list(g1.neighbors(i3))
        la.append(g1.edge[i2[0]][i3]['label'])
        la.append(g1.edge[i2[1]][i3]['label'])
        if len(la[0]) < len(la[1]):
            laa = la[0]
        else:
            laa = la[1]
        g1.add_edge(i2[0], i2[1], label=laa, weight=len(laa))
        g1.remove_node(i3)


def add_edges(edge, nv, c, l4, ver, vertices_in_cut_edges, cuts_at_cut_edges, g1):  # if True:
    mark = []
    for x in range(g1.number_of_nodes()):
        mark.append(0)
    for i3 in range(g1.number_of_nodes()):
        if mark[i3] == 0:
            for i4 in range(i3 + 1, g1.number_of_nodes()):
                if i3 >= len(c[0]) - 1:
                    if set(g1.node[i3]['s']).issubset(set(g1.node[i4]['s'])) and mark[i3] == 0:
                        cut_e = []
                        for v1 in range(1, nv + 1):
                            if int(l4[ver[g1.node[i3]['lead']][v1][g1.node[i3]['level']] - 1]) > 0:
                                if not set(edge[v1]).issubset(g1.node[i3]['s']):
                                    cut_e.append(vertices_in_cut_edges[v1 - 1])
                        # print "degree of",i3,g1.degree()[i3]
                        if g1.degree(i3) > 0:
                            neigh = list(g1.neighbors(i3))
                            lab = []
                            for nei in neigh:
                                lab1 = g1.edge[i3][nei]['label'] + lab
                                lab = lab1
                        # print cut_e,i3,i4,list(set(lab)&set(cut_e)),lab
                        g1.add_edge(i3, i4, label=list(set(lab) & set(cut_e)), weight=len(list(set(lab) & set(cut_e))))
                        mark[i3] = 1
                else:
                    if set(g1.node[i3]['s']).issubset(set(g1.node[i4]['s'])) and mark[i3] == 0:
                        cut_e = []
                        for j in cuts_at_cut_edges[i3]:
                            cut_e.append(j)
                        g1.add_edge(i3, i4, label=cut_e, weight=len(cut_e))
                        mark[i3] = 1
    remove_extra_vertices(g1)


def make_graph(depth, file_name, limit_on_size, cuts_at_cut_edges, g1, current_width, u, v, vertices_in_cut_edges):  # if True:
    (nv, ne, nsteps, l4, edge, arc, led, ver, ctr) = read_sol(u,v,depth, current_width, file_name, limit_on_size)
    c = make_component(nsteps, ne, l4, led, arc)
    # print c
    g1.clear()
    nn = 0
    for i3 in range(1, len(c[0])):
        g1.add_node(nn, lead=i3, level=0, s=[i3])
        nn += 1
    for i2 in range(len(c)):
        if c[i2] != c[i2 - 1]:
            for i3 in (set(c[i2]) - {0}):
                g1.add_node(nn, lead=i3, level=i2 + 1, s=([i3] + [i4 for i4, value in enumerate(c[i2]) if value == i3]))
                nn += 1
    add_edges(edge, nv, c, l4, ver,vertices_in_cut_edges, cuts_at_cut_edges, g1)
    write_graph(g1, file_name + str(limit_on_size) + 'g1')
    # nx.write_dot(g1, "g1" + str(rnd) + ".dot")
    # os.system("dot -Tps g1.dot -o g1.ps")


# returns disjoint union of a1 and a2 Independent
def dis_joint(a1, a2):
    for no in a2.nodes():
        a1.add_node(no)
    for edg in a2.edges():
        a1.add_edge(edg[0], edg[1], label=a2.edge[edg[0]][edg[1]]['label'], weight=a2.edge[edg[0]][edg[1]]['weight'])
    return a1


# Removes extra nodes from a  Independent
def remove_extra_nodes(a):
    deg2 = []
    for i33 in [i4 for i4, value in enumerate(dict(a.degree()).values()) if value == 2]:
        deg2.append(dict(a.degree()).keys()[i33])
    for i3 in deg2:
        lab = []
        i4 = list(a.neighbors(i3))
        if len(i4) == 2:
            lab.append(a.edge[i4[0]][i3]['label'])
            lab.append(a.edge[i4[1]][i3]['label'])
            if len(lab[0]) < len(lab[1]):
                laa = lab[0]
            else:
                laa = lab[1]
            a.add_edge(i4[0], i4[1], label=laa, weight=len(laa))
            a.remove_node(i3)
    return a


# merges two graphs a2 and g2_copy according to the list ed2 Independent
def merges(a2, ed2, g2_copy, g_copy):
    print ed2
    a = dis_joint(g2_copy, a2)
    # write_graph(a2, file_name + str(limit_on_size) + 'a2')
    la = nx.get_edge_attributes(a2, 'label')
    #  print "labels:",la.values(),connecting_edges
    leaf = []
    leaf1 = -1
    for i1 in [i2 for i2, value in enumerate(dict(a2.degree()).values()) if value == 1]:
        leaf.append(dict(a2.degree()).keys()[i1])
    # print "leaf:",leaf
    for e in ed2:
        if e[0] in g2_copy.nodes():
            leaf1 = e[0]
        elif e[1] in g2_copy.nodes():
            leaf1 = e[1]
        else:
            print "error with leaf1"
        leaf2 = -1
        for key1 in [key2 for key2, value in enumerate(la.values()) if value == g_copy.edge[e[0]][e[1]]['label']]:
            if la.keys()[key1][0] in leaf:
                leaf2 = la.keys()[key1][0]
            elif la.keys()[key1][1] in leaf:
                leaf2 = la.keys()[key1][1]
            if leaf2 == -1:
                print "error with leaf2"
                continue
        # if leaf1!=-1 and leaf2!=-1:
        a.add_edge(leaf1, leaf2, label=g_copy.edge[e[0]][e[1]]['label'], weight=g_copy.edge[e[0]][e[1]]['weight'])
        print "adding edges", leaf1, leaf2, str(nx.cycle_basis(a))
    update_a = remove_extra_nodes(a)
    print "connected:" + str(nx.is_connected(update_a)) + "cycles" + str(nx.cycle_basis(update_a))
    # write_graph(a, file_name + str(limit_on_size) + 'a')
    return update_a


def merge_nodes(graph, nodes):
    """
    Merges the selected `nodes` of the graph G into one `new_node`,
    meaning that all the edges that pointed to or from one of these
    `nodes` will point to or from the `new_node`.
    attr_dict and **attr are defined as in `G.add_node`.
    :param graph:
    :param nodes:
    """
    m = min(nodes)
    # print "m", m , "nodes" ,nodes
    # print "edges", G.edges(nodes)
    for n1, n2 in graph.edges(nodes):
        # For all edges related to one of the nodes to merge,
        # make an edge going to or coming from the `new gene`.
        # print "n1", n1, "n2", n2
        if n1 not in nodes:
            graph.add_edge(m, n1)
        elif n2 not in nodes:
            graph.add_edge(m, n2)
    for n in nodes:  # remove the merged nodes
        if n != m:
            if n in graph.nodes():
                graph.remove_node(n)
                # print n


def get_minor(remaining_graph, real_graph, graph, cut_vertices, limit_on_size):
    local_minor = graph.copy()
    copy_vertices = list(cut_vertices)
    leaf_in_y = []
    leaf_in_g = []
    for i1 in [i2 for i2, value in enumerate(dict(remaining_graph.degree()).values()) if value == 1]:
        vertex = dict(remaining_graph.degree()).keys()[i1]
        leaf_in_y.append(vertex)
    for i1 in [i2 for i2, value in enumerate(dict(real_graph.degree()).values()) if value == 1]:
        vertex = dict(real_graph.degree()).keys()[i1]
        leaf_in_g.append(vertex)
    common_leaves = list(set(leaf_in_g).intersection(set(leaf_in_y)))
    contracted = dict()
    incident = list()
    while len(common_leaves) > 0 and local_minor.number_of_edges() > 50:
        vertex = random.choice(common_leaves)
        common_leaves.remove(vertex)
        neighbour = list(real_graph.neighbors(vertex))
        #         print "neigh:",neighbour,vertex,original_graph.neighbors(vertex)
        contract_list = real_graph.edge[vertex][neighbour[0]]['label']
        contract_list[0] += 1
        contract_list[1] += 1
        # print copy_vertices,contracted,contract_list,
        if contract_list[0] in contracted.keys():
            contract_list[0] = contracted[contract_list[0]]
        if contract_list[1] in contracted.keys():
            contract_list[1] = contracted[contract_list[1]]
        # print contract_list
        if contract_list[0] in copy_vertices and contract_list[1] in copy_vertices:
            incident.append(vertex)
            continue
        if max(contract_list) in copy_vertices:
            copy_vertices.remove(max(contract_list))
            copy_vertices.append(min(contract_list))
        print vertex, neighbour[0], "contract_list:", contract_list, real_graph.edge[vertex][neighbour[0]]['label']
        merge_nodes(local_minor, contract_list)
        contracted[max(contract_list)] = min(contract_list)
    while local_minor.number_of_edges() > limit_on_size:
        edge = random.choice(list(local_minor.edges()))
        merge_nodes(local_minor, edge)
    # write_graph(local_minor, 'test')
    print local_minor.nodes(), sorted(cut_vertices), sorted(copy_vertices)
    print local_minor.number_of_edges(), local_minor.edges()
    # write_graph(graph, 'original')
    nx.cycle_basis(local_minor)
    return local_minor


#     print len(leaf_in_G),leaf_in_G
#     print len(leaf_in_y),leaf_in_y
#     print len(common_leaves), common_leaves
#     write_graph(original_graph, 'original_graph')
#     write_graph(removed_graph, 'g2')
#     write_graph(local_graph, 'y')

def write_edge(graph, fname, limit_on_size):
    edge_file = open('{0}{1}_minor.edge'.format(fname, str(limit_on_size)), 'w')
    edge_file.write('p edge ' + str(graph.number_of_nodes()) + ' ' + str(graph.number_of_edges()) + '\n')
    #     print 'p edge ' + str(graph.number_of_nodes()) + ' ' + str(graph.number_of_edges())
    for cut_edge in graph.edges():
        edge_file.write('e ')
        for vertex in cut_edge:
            edge_file.write(str(vertex) + ' ')
        edge_file.write('\n')
    # print "e ",cut_edge[0],cut_edge[1]
    edge_file.close()


def solve(width, file_name, d, limit_on_size, original_graph, time_out):
    cmd_make = 'mkfifo ' + file_name + '.fifo'
    cmd = 'timeout {0} cat {1}.edge | ./hybw2sat {2} {3}.fifo &'.format(str(time_out), file_name,str(width),file_name)
    print cmd
    cmd2 = './glucose -verb=0 -cpu-lim=' + str(time_out) + ' ' + file_name + '.fifo ' + file_name + str(width) + '.sol'
    print cmd2
    cmd_remove = 'rm ' + file_name + '.fifo'
    os.system(cmd_make)
    os.system(cmd)
    os.system(cmd2)
    os.system(cmd_remove)
    sys.stdout.flush()
    while width >= 2:
        try:
            fil = open(file_name + str(width) + '.sol')
            s = fil.readline()
            print s[0]
            fil.close()
        except Exception as ex1:
            print ex1
            s = 'I'
        while s[0] != "U" and width >= 2:
            width -= 1
            cmd_make = 'mkfifo ' + file_name + '_minor.fifo'
            cmd = 'timeout {0} cat {1}.edge | ./hybw2sat {2} {3}.fifo &'.format(str(time_out), file_name,str(width),file_name)
            print cmd
            cmd2 = './glucose -verb=0 -cpu-lim=' + str(time_out) + ' ' + file_name + '.fifo ' + file_name + str(width) + '.sol'
            print cmd2
            cmd_remove = 'rm ' + file_name + '.fifo'
            os.system(cmd_make)
            os.system(cmd)
            os.system(cmd2)
            os.system(cmd_remove)
            sys.stdout.flush()
            try:
                fil = open(file_name + str(width) + '.sol')
                s = fil.readline()
                fil.close()
                print s[0]
            except Exception as ex1:
                print ex1
                s = 'I'
            while s[0] == 'U':
                width += 1
                try:
                    fil = open(file_name + str(width) + '.sol')
                    s = fil.readline()
                    print s[0]
                    fil.close()
                except Exception as ex1:
                    print ex1
                    s = 'I'
                if s[0] == 'S':
                    os.system(cmd_remove)
                    return 0, width
                else:
                    return 1, width
        else:
            os.system(cmd_remove)
            return 1, width
    else:
        os.system(cmd_remove)
        return 1, width


def get_width(graph, fname, mwidth, limit_on_size, original_graph, time_out, temp_path):
    # minor_cut = nx.get_edge_attributes(g, 'weight')
    # minor_scut = sorted(minor_cut, key=minor_cut.get, reverse=True)
    write_edge(graph, temp_path+fname, limit_on_size)
    solved, width = solve(mwidth, temp_path+fname, graph.number_of_edges(), limit_on_size, original_graph, time_out)
    return width


# Checks if graph is a branch decomposition Independent
def is_branch_decomposition(graph, initial_nodes):
    if graph.number_of_nodes() != initial_nodes:
        print "nodes do not match", graph.number_of_nodes(), initial_nodes
        return False
    set_degree = set(dict(graph.degree()).values())
    print "connected?", nx.is_connected(graph), "cycles", nx.cycle_basis(graph)
    if set_degree != {1, 3}:
        print "degrees not matched"
        return False
    if not nx.is_connected(graph):
        print "not connected"
        return False
    if nx.cycle_basis(graph):
        print "has cycles"
        return False
    for node in graph.nodes():
        neigh = list(graph.neighbors(node))
        if len(neigh) == 3:
            cut1 = set(graph.edge[node][neigh[0]])
            cut2 = set(graph.edge[node][neigh[1]])
            cut3 = set(graph.edge[node][neigh[2]])
            if cut1.issubset(cut2 | cut3) == False or cut3.issubset(cut2 | cut1) == False or cut2.issubset(
                            cut1 | cut3) == False:
                print "different cuts"
                return False
    return True


# Returns an adjency list from the name.edge file Independent
def read_adjency(name):
    solution_file = open(name + '.edge')
    l3 = solution_file.readline()
    l3 = l3.replace('p edge', '')
    l3 = l3.replace('\n', '')
    l3 = l3.split()
    nv = int(l3[0])
    ne = int(l3[1])
    edge = [[]]
    for i1 in range(1, nv + 1):
        edge.append([])
    for i1 in range(1, ne + 1):
        l3 = solution_file.readline()
        l3 = l3.replace('e ', '')
        l3 = l3.split()
        for i4 in l3:
            # noinspection PyTypeChecker
            edge[int(i4)].append(int(i1))
    solution_file.close()
    return edge, nv, ne


# Return an edgelist from name.edge file Independent
def read_edge(name):
    edge = nx.Graph()
    solution_file = open(name + '.edge')  # if True:
    l3 = solution_file.readline()
    l3 = l3.replace('p edge', '')
    l3 = l3.replace('\n', '')
    l3 = l3.split()
    ne = int(l3[1])
    for i1 in range(1, ne + 1):
        l3 = solution_file.readline()
        l3 = l3.replace('e ', '')
        l3 = l3.split()
        for i2 in l3:
            if int(i2) not in edge.nodes():
                edge.add_node(int(i2))
        edge.add_edge(int(l3[0]), int(l3[1]))
    solution_file.close()
    return edge


# def improve_decomposition(cut_list,sat_call):
# Displays the graph if matplotlib is called Independent
def show_graph(graph):
    m = graph.copy()
    pos = graphviz_layout(m)
    nx.draw_networkx_edge_labels(m, pos)
    nx.draw_networkx_nodes(m, pos)
    nx.draw_networkx_labels(m, pos)
    write_dot(m, "m.dot")
    os.system("dot -Tps m.dot -o m.ps")
    nx.draw(m, pos)
    # plt.show()


# Writes the graph in .dot format Independent
def write_graph(m, name):
    write_dot(m, name + '.dot')
    # os.system("dot -Tps " + name + ".dot -o " + name + ".ps")


def local_improve(file_name, limit_on_size, time_out, temp_path,second=0):
    g = nx.Graph()
    g1 = nx.Graph()
    y = nx.Graph()
    original_file = file_name
    file_name += '.decomp'
    print file_name, limit_on_size, time_out
    if second==0:
        try:
            g = nx.read_edgelist(file_name, edgetype=int, nodetype=int)
        except Exception as e1:
            logging.exception(e1)
            print "exception reading file"
            pass
    if second==1:
        try:
            g = nx.read_edgelist(temp_path+file_name, edgetype=int, nodetype=int)
        except Exception as e1:
            logging.exception(e1)
            print "exception reading file"
            pass

    # file_name = '/dev/shm/{0}'.format(file_name)
    g3 = g.copy()
    cut = nx.get_edge_attributes(g, 'weight')
    cuts = nx.get_edge_attributes(g, 'label')
    scut = sorted(cut, key=cut.get, reverse=True)
    i = 0
    max_cut = cut[(scut[0][0], scut[0][1])]
    possible = 0
    max_dist = 6
    rnd = 3
    maxc = max_cut
    max_cut_list = []
    initial_number_of_vertices = g.number_of_nodes()
    print "Is it a branch decomposition?", is_branch_decomposition(g, initial_number_of_vertices), nx.is_connected(g)
    # from dfsvar import *
    for cu in [cu1 for cu1, val in enumerate(cut.values()) if val == max_cut]:
        max_cut_list.append(cuts.values()[cu])
    # print_local_sizes()
    original_graph = read_edge(original_file)
    print file_name
    file_name = temp_path + file_name
    print file_name
    minor_width = list()
    if int(original_graph.number_of_edges()) < int(limit_on_size):
        temp_width = get_width(original_graph, original_file, maxc, limit_on_size, original_graph, time_out,temp_path)
        print "current branch width:", temp_width, maxc, temp_width
        sys.exit(0)
    while max_cut_list:
        # print_local_sizes()#if True:
        print "max_cut_list:", len(max_cut_list)
        max_cut_element = max_cut_list.pop()
        if max_cut_element not in cuts.values():
            print "continue"
            continue
        i = cuts.values().index(max_cut_element)
        u = cuts.keys()[i][0]
        v = cuts.keys()[i][1]
        print "i", i, cuts[(u, v)], rnd
        # rnd=rnd+1
        if cut[(u, v)] < maxc:
            print "current branch width:", cut[(scut[0][0], scut[0][1])], "original branch width:", maxc
            sys.stdout.flush()
        if cut[(u, v)] >= max_cut:  # if True:
            g2 = g.copy()
            vertices_in_cut_edges = []
            cuts_at_cut_edges = []
            leaf_edges = []
            connecting_edges = []
            internal_nodes = []
            unsat = 0
            no_sat_call = 0
            make_g = 1
            current_depth = 0
            print "extract_g"
            print "number of nodes before function", g.number_of_nodes()
            u1, v1, max_dist1, max_cut, cut, cuts, max_cut_list, file_name, leaf_edges, internal_nodes, cuts_at_cut_edges, vertices_in_cut_edges, g2, y, connecting_edges, g, u, v = dfs_component(
                u, v, max_dist, max_cut, cut, cuts, max_cut_list, file_name, leaf_edges, internal_nodes,
                cuts_at_cut_edges, vertices_in_cut_edges, g2, y, connecting_edges, g, u, v)
            print "local graph", len(cuts_at_cut_edges)
            if len(cuts_at_cut_edges) > limit_on_size:
                print "local graph too big"
                break
            current_width = max_cut - 1
            current_cut = cuts[(u, v)]
            print "minor"
            for ele_l1 in cuts_at_cut_edges:
                if len(ele_l1) == max_cut:
                    no_sat_call = 1
                    print "no sat call"
                    current_width = max_cut
                    break
            sys.stdout.flush()
            if no_sat_call == 0:
                print "print_graph", current_width, cuts_at_cut_edges
                (unsat, current_width, current_depth) = print_graph(current_width, file_name, limit_on_size,
                                                                    vertices_in_cut_edges, cuts_at_cut_edges, time_out,
                                                                    max_cut, i, original_graph)
                print unsat
            if unsat == 1 and rnd == 1:
                current_width += 1
                (unsat, current_width, current_depth) = print_graph(current_width, file_name, limit_on_size,
                                                                    vertices_in_cut_edges, cuts_at_cut_edges, time_out,
                                                                    max_cut, i, original_graph)
                print unsat
                make_g = 0
            if unsat == 0:
                print "make_graph", current_width
                make_graph(current_depth, file_name, limit_on_size, cuts_at_cut_edges, g1, current_width, u, v,
                           vertices_in_cut_edges)
                a3 = nx.Graph()
                h1 = nx.convert_node_labels_to_integers(g1, first_label=g.number_of_nodes())
                # show_graph(h1)
                print "merges"
                h2 = merges(h1, connecting_edges, g2, g)
                print g.number_of_nodes(), h2.number_of_nodes()
                h2 = nx.convert_node_labels_to_integers(h2)
                g = h2.copy()
                g2 = g.copy()
                cut = nx.get_edge_attributes(g, 'weight')
                cuts = nx.get_edge_attributes(g, 'label')
                scut = sorted(cut, key=cut.get, reverse=True)
                print len(cuts), len(cut), len(scut)
                print "Is it a branch decomposition?", is_branch_decomposition(g, initial_number_of_vertices)
        if cut[(scut[0][0], scut[0][1])] < max_cut:
            max_cut = cut[(scut[0][0], scut[0][1])]
            max_cut_list = []
            for cu in [cu1 for cu1, val in enumerate(cut.values()) if val == max_cut]:
                max_cut_list.append(cuts.values()[cu])
            print "updated ", max_cut  # , max_cut_list
            # print_local_sizes()
            rnd = 2
        if max_cut_list == [] and rnd > 0:
            print "in the round-1"
            rnd -= 1
            for cu in [cu1 for cu1, val in enumerate(cut.values()) if val == max_cut]:
                max_cut_list.append(cuts.values()[cu])
            print "round-1 round", rnd, max_cut,  # max_cut_list,
        print "@end"  # , max_cut_list
    print "Is it a branch decomposition?", is_branch_decomposition(g, initial_number_of_vertices)
    print "final branch width:", cut[scut[0][0], scut[0][1]], maxc

def main(argv):
    temp_path = '/home/neha/temp/'
    file_name = 'eil51'
    limit_on_size = 60
    time_out = 600
    folder='/home/neha/Dropbox/python/'
    try:
        opts, args = getopt.getopt(argv, "hi:l:t:o:f:", ["ifile=", "lim=","timeout=", "temp=","folder="])
    except getopt.GetoptError:
        print 'test.py -i <inputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <input_file(without_extension)> -l <limit_on_size> -t <timeout_for_SAT_call>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            file_name = arg
        elif opt in ("-l", "--lim"):
            limit_on_size = int(arg)
            print 'limit', limit_on_size
        elif opt in ("-t", "--timeout"):
            time_out = int(arg)
        elif opt in ("-p", "--temp"):
            temp_path = arg
        elif opt in ("-f","--folder"):
            folder=arg
            print 'timeout', time_out
    print 'Input file is ', file_name
    # os.chdir('/home/staff/neha/Dropbox/python/decomp')
    # os.chdir('/home/neha/bwsat/bwsat/test/edge/decomp/')
    # os.chdir('/home/neha/bwsat/bwsat/test/Delunay/decomp/')
    os.chdir(folder)
    print "start"
    # file_name = 'zeroin'
    local_improve(file_name,limit_on_size,time_out,temp_path)



if __name__ == "__main__":
    main(sys.argv[1:])
# cut = nx.get_edge_attributes(g, 'weight')
# cuts = nx.get_edge_attributes(g, 'label')
# scut = sorted(cut, key=cut.get, reverse=True)
# max_cut = cut[(scut[0][0], scut[0][1])]
# max_cut_list = []
# for cu in [cu1 for cu1, val in enumerate(cut.values()) if val == max_cut]:
#     max_cut_list.append(cuts.values()[cu])
# while max_cut_list:
#     # print_local_sizes()#if True:
#     print "max_cut_list:", len(max_cut_list)
#     max_cut_element = max_cut_list.pop()
#     g2 = g.copy()
#     vertices_in_cut_edges = []
#     cuts_at_cut_edges = []
#     connecting_edges = []
#     unsat = 0
#     no_sat_call = 0
#     print "extract_g"
#     print "number of nodes before function", g.number_of_nodes()
#     dfs_component(u, v, max_dist)
#     minor = get_minor(g2, g, original_graph, vertices_in_cut_edges)
#     print "number of edges in minor", minor.number_of_edges()
#     # sys.exit(0)
#     if minor.number_of_nodes() > 2:
#         temp_width = get_width(minor, temp_path + original_file, max_cut)
#         minor_width.append(temp_width)
# # write_graph(g, file_name)
# sorted(minor_width, reverse=True)
# print "@exit max_cut_list:", max_cut_list,
# print "Is it a branch decomposition?", is_branch_decomposition(g, initial_number_of_vertices)
# print "lower bound: ", minor_width[0]

# rl1304.decomp fl417.decomp
# def dfs_ed(u1, v1, max_dist1):
#     dist = []
#     for x in xrange(g.number_of_nodes()):
#         dist.append(-1)
#     dist[u1] = 0
#     dist[v1] = 0
#     cut_list = []
#     for x in xrange(g.number_of_nodes()):
#         cut_list.append(max_cut)
#     cut_list[u1] = cut[(u1, v1)]
#     cut_list[v1] = cut[(u1, v1)]
#     dfs_list = [u1, v1]
#     boundary_nodes = []
#     while dfs_list:
#         v1 = dfs_list.pop(0)
#         neigh = g.neighbors(v1)
#         for w in neigh:
#             if dist[w] == -1:
#                 dist[w] = dist[v1] + 1
#                 if cut[ord1(v1, w)] < cut_list[w]:
#                     cut_list[w] = cut[ord1(v1, w)]
#                 if dist[w] == max_dist1 or (g.degree()[w] == 1 and dist[w] <= max_dist1):
#                     if cut_list[w] > cut_list[v1]:
#                         for rem in neigh:
#                             if (dist[rem] == dist[v1] + 1) or dist[rem] == -1:
#                                 boundary_nodes.remove(rem)
#                                 break
#                     if w not in boundary_nodes:
#                         boundary_nodes.append(w)
#                 elif w not in dfs_list:
#                     if cut_list[w] <= cut_list[v1]:
#                         dfs_list.append(w)
#         internal_nodes.append(v1)
#     nodes_to_delete = list(set(boundary_nodes) | set(internal_nodes))
#     y = g.subgraph(nodes_to_delete)
#     leaf_edges = []
#     connecting_edges = []
#     cuts_at_cut_edges = []
#     vertices_in_cut_edges = []
#     for leaf in [i1 for i1, value in enumerate(y.degree().values()) if value == 1]:
#         leaves = y.degree().keys()[leaf]
#         leaf_edges.append((leaves, y.neighbors(leaves)[0]))
#         cuts_at_cut_edges.append(y.edge[leaves][y.neighbors(leaves)[0]]['label'])
#         for a in y.edge[leaves][y.neighbors(leaves)[0]]['label']:
#             if a not in vertices_in_cut_edges:
#                 vertices_in_cut_edges.append(a)
#         if g.degree()[leaves] > 1:
#             connecting_edges.append((leaves, y.neighbors(leaves)[0]))
#             if leaves in nodes_to_delete:
#                 nodes_to_delete.remove(leaves)
#     g2.remove_nodes_from(nodes_to_delete)

# def dfs_component_budget(u1, v1, budget, print_in_file):  # if True:
#     max_dist1 = 5
#     dist = []
#     for x in xrange(g.number_of_nodes()):
#         dist.append(-1)
#     dist[u1] = 0
#     dist[v1] = 0
#     cut_list = []
#     for x in xrange(g.number_of_nodes()):
#         cut_list.append(max_cut)
#     cut_list[u1] = cut[(u1, v1)]
#     cut_list[v1] = cut[(u1, v1)]
#     boundary_nodes = []
#     dfs_list = [u1, v1]
#     visited = []
#     while budget > len(list(set(boundary_nodes) | set(internal_nodes))):
#         print "budget", budget, "size of y:", len(list(set(boundary_nodes) | set(internal_nodes)))
#         while dfs_list:
#             v1 = dfs_list.pop(0)
#             neigh = g.neighbors(v1)
#             for w in neigh:
#                 # print w, v1, dist[w]
#                 if dist[w] == -1:
#                     dist[w] = dist[v1] + 1
#                     if cut[ord1(v1, w)] < cut_list[w]:
#                         cut_list[w] = cut[ord1(v1, w)]
#                     if cut_list[w] == max_cut and dist[v1] == 0:
#                         dist[w] = 0
#                         if cuts[ord1(v1, w)] in max_cut_list:
#                             max_cut_list.remove(cuts[ord1(v1, w)])
#                     if dist[w] == max_dist1 or (g.degree()[w] == 1 and dist[w] <= max_dist1):
#                         if cut_list[w] > cut_list[v1]:
#                             for rem in neigh:
#                                 if (dist[rem] == dist[v1] + 1) or dist[rem] == -1:
#                                     if rem in boundary_nodes:
#                                         boundary_nodes.remove(rem)
#                                     break
#                         if w not in boundary_nodes:
#                             boundary_nodes.append(w)
#                     elif w not in dfs_list:
#                         if cut_list[w] <= cut_list[v1]:
#                             dfs_list.append(w)
#             internal_nodes.append(v1)
#         max_of_dist = max(dist)
#         for ele in [i2 for i2, value in enumerate(dist) if value == max_of_dist]:
#             # print ele
#             if ele not in dfs_list and ele not in visited:
#                 dfs_list.append(ele)
#                 visited.append(ele)
#         if not dfs_list:
#             break
#         max_dist1 += 1
#     nodes_to_delete = list(set(boundary_nodes) | set(internal_nodes))
#     component_size = []
#     for ele in [i2 for i2, value in enumerate(dist) if value == 0]:
#         component_size.append(ele)
#     y = g.subgraph(nodes_to_delete)
#     leaf_edges = []
#     connecting_edges = []
#     cuts_at_cut_edges = []
#     vertices_in_cut_edges = []
#     for leaf in [i1 for i1, value in enumerate(y.degree().values()) if value == 1]:
#         leaves = y.degree().keys()[leaf]
#         leaf_edges.append((leaves, y.neighbors(leaves)[0]))
#         cuts_at_cut_edges.append(y.edge[leaves][y.neighbors(leaves)[0]]['label'])
#         for a in y.edge[leaves][y.neighbors(leaves)[0]]['label']:
#             if a not in vertices_in_cut_edges:
#                 vertices_in_cut_edges.append(a)
#         if g.degree()[leaves] > 1:
#             connecting_edges.append((leaves, y.neighbors(leaves)[0]))
#             if leaves in nodes_to_delete:
#                 nodes_to_delete.remove(leaves)
#     print y.number_of_nodes()
#     if print_in_file == 1:
#         y_file = open('component_budget.csv', mode='a')
#         y_file.write(
#               '{0} {1} {2} {3} {4} {5} {6}\n'.format(file_name, str(u), str(v), str(max_cut), str(y.number_of_nodes()),
#                                                        str(max_dist1), str(len(component_size))))
#         y_file.close()
#     g2.remove_nodes_from(nodes_to_delete)
#
# def print_local_sizes():
#     print "printing local sizes"
#     max_cut_list_duplicate = list(max_cut_list)
#     while max_cut_list:
#         max_cut_element = max_cut_list.pop()
#         if max_cut_element not in cuts.values():
#             print "continue"
#             continue
#         i = cuts.values().index(max_cut_element)
#         u = cuts.keys()[i][0]
#         v = cuts.keys()[i][1]
#         if cut[(u, v)] >= max_cut:  # if True:
#             g2 = g.copy()
#             vertices_in_cut_edges = []
#             cuts_at_cut_edges = []
#             leaf_edges = []
#             connecting_edges = []
#             internal_nodes = []
#             print "dfs component"
#             dfs_component(u, v, max_dist, 1)
#             g2 = g.copy()
#             vertices_in_cut_edges = []
#             cuts_at_cut_edges = []
#             leaf_edges = []
#             connecting_edges = []
#             internal_nodes = []
#             print "dfs component budget"
#             dfs_component_budget(u, v, 2 ** 5, 1)
#         else:
#             break
#     max_cut_list = list(max_cut_list_duplicate)

# def get_graph(graph):
#     start_time=datetime.now()
#     edges_in_graph=[[]]
#     for i1 in [i2 for i2, value in enumerate(dict(graph.degree()).values()) if value == 1]:
#         vertex=dict(graph.degree()).keys()[i1]
#         neighbour=graph.neighbors(vertex)
#         edges_in_graph.append(graph.edge[vertex][neighbour[0]]['label'])
#     edges_in_graph.pop(0)
#     print datetime.now()-start_time
#     return edges_in_graph
