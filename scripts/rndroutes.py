# -*- coding: utf-8 -*-
import os
import sys

import random
import xml.etree.ElementTree as ET


class Net:
    def __init__(self):
        self.edges = {}

    def add_edge(self, edge):
        self.edges[edge.id] = edge

    def has_edge(self, edge_id):
        return self.edges.has_key(edge_id)

    def get_edge(self, edge_id):
        return self.edges[edge_id]

    def get_edges(self):
        return self.edges.values()

    def add_link(self, from_edge_id, to_edge_id):
        self.edges[from_edge_id].add_link(self.edges[to_edge_id])


class Edge:
    def __init__(self, id):
        self.id = id
        self.links = []

    def add_link(self, to_edge):
        self.links.append(to_edge)

    def get_links(self):
        return list(self.links)


class Route:
    def __init__(self, id):
        self.id = id
        self.edges = []

    def add_edge(self, edge):
        self.edges.append(edge)

    def to_xml(self):
    	return '<route id="{}" edges="{}"/>'.format(self.id, " ".join(self.edges))


def load_net(file):
    ''' Loads the net file

    :type file: string
    :param file: the path to the net file (SUMO config xml)
    :param r: how many past intervals to include in the input data
    :param d: how far future interval to predict
    '''

    #############
    # LOAD DATA #
    #############
    net = Net()
    tree = ET.parse(file)
    e_root = tree.getroot()

    print("loading edges")
    e_edges = e_root.findall('.//edge')
    for e_edge in e_edges:
        try:
            eid = e_edge.get('id')
            if not eid.startswith(':'):
                print(" edge {}".format(eid))
                net.add_edge(Edge(eid))
        except Exception as e:
            print("ERR: {}".format(e))

    print("loading connections")
    e_conns = e_root.findall('.//connection')
    for e_conn in e_conns:
        fid = e_conn.get('from')
        tid = e_conn.get('to')
        print(" {} --> {}".format(tid, fid))
        if net.has_edge(fid) and net.has_edge(tid):
            print(" {} --> {}".format(tid, fid))
            net.add_link(fid, tid)

    return net


def generate_routes(net, num=1):
    routes = []
    edges = net.get_edges()
    random.shuffle(edges)

    for i in xrange(num):
        edge = edges[i % len(edges)]
        route = Route('r_{}'.format(i))
        route.add_edge(edge)

        # edge つなげてく
        while True:
            links = edge.get_links()
            if 0 <= len(links):
                break
            edge = random.shuffle(links)[0]
            route.add_edge(edge)

        routes.append(route)

    return routes


if __name__ == '__main__':
    net = load_net('../data/net.net.xml')
    routes = generate_routes(net, num=10)
    for route in routes:
    	print(route.to_xml())