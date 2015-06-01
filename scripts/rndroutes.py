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
        self.lanes = []
        self.links = []

    def add_lane(self, lane):
        self.lanes.append(lane)

    def add_link(self, to_edge):
        self.links.append(to_edge)

    def get_links(self):
        return list(self.links)

    def get_capacity(self):
        return len(self.lanes) # FIXME: return correct capacity

class Lane:
    def __init__(self, id, edge, index, speed, length):
        self.id = id
        self.edge = edge
        self.index = index
        self.speed = speed
        self.length = length

class Route:
    def __init__(self, id):
        self.id = id
        self.edges = []

    def add_edge(self, edge):
        self.edges.append(edge)

    def to_xml(self):
        return '<route id="{}" edges="{}"/>'.format(self.id, " ".join(e.id for e in self.edges))

class Flow:
    def __init__(self, id, route, begin, end, period):
        self.id = id
        self.route = route
        self.begin = begin
        self.end = end
        self.period = period

    def to_xml(self):
        return '<flow id="{}" route="{}" begin="{:.0f}" end="{:.0f}" period="{:.0f}"/>'.format(self.id, self.route.id, self.begin, self.end, self.period)

class Runtime:
    def __init__(self):
        self.net = None
        self.routes = []
        self.flows = []
        self.beginTime = 0
        self.endTime = 0

    def load_net(self, file):
        self.net = load_net(file)

    def set_time(self, beginTime, endTime):
        self.beginTime = beginTime
        self.endTime = endTime

    def generate_routes(self, num, max_hop):
        self.routes = generate_routes(self.net, num, max_hop)

    def generate_flows(self, num=1):
        self.flows = []
        for route in self.routes:
            self.flows.extend(generate_flows(route, num=num))
        self.flows.sort(key=lambda x:x.begin)

    def to_xml(self):
        window = self.endTime - self.beginTime
        xml = ''
        xml += '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">'
        xml += "\n"
        for route in self.routes:
            xml += "\t"
            xml += route.to_xml()
            xml += "\n"
        xml += "\n"
        for flow in self.flows:
            orig_begin = flow.begin
            orig_end = flow.end
            flow.begin = flow.begin * window
            flow.end = flow.end * window
            xml += "\t"
            xml += flow.to_xml()
            flow.begin = orig_begin
            flow.end = orig_end
            xml += "\n"
        xml += "</routes>"
        return xml


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
    for e_edge in e_root.findall('.//edge'):
        try:
            eid = e_edge.get('id')
            if not eid.startswith(':'):
                print(" edge {}".format(eid))
                edge = Edge(eid)
                net.add_edge(edge)
                for e_lane in e_edge.findall('.//lane'):
                    lid = e_lane.get('id')
                    lindex = e_lane.get('index')
                    lspeed = e_lane.get('speed')
                    llength = e_lane.get('length')
                    lane = Lane(lid, edge, lindex, lspeed, llength)
                    edge.add_lane(lane)
        except Exception as e:
            print("ERR: {}".format(e))

    print("loading connections")
    for e_conn in e_root.findall('.//connection'):
        fid = e_conn.get('from')
        tid = e_conn.get('to')
        if net.has_edge(fid) and net.has_edge(tid):
            print(" {} --> {}".format(tid, fid))
            net.add_link(fid, tid)

    return net

def generate_routes(net, num=1, max_hop=10):
    routes = []
    edges = net.get_edges()
    random.shuffle(edges)

    for i in xrange(num):
        edge = edges[i % len(edges)]
        route = Route('r_{}'.format(i))
        route.add_edge(edge)
        hop = 0

        # edge つなげてく
        while True:
            links = edge.get_links()
            if len(links) <= 0 or max_hop < hop:
                break
            random.shuffle(links)
            edge = links[0]
            route.add_edge(edge)
            hop += 1

        routes.append(route)

    return routes

def generate_flows(route, num=1, min_time=0, max_time=1, mu=0.5, sigma=0.5):
    flows = []
    window = max_time - min_time

    for i in xrange(num):
        fid = "f_{}_{}".format(route.id, i)
        z = random.normalvariate(mu, sigma)
        begin = min(max(0, z - (sigma/2.0)), 1)
        end = min(max(0, z + (sigma/2.0)), 1)
        if begin == end:
            if begin < 0.5:
                begin = 0
                end = min(max(0, sigma/2.0), 1)
            else:
                begin = min(max(0, sigma/2.0), 1)
                end = 1
        period = max(10, random.normalvariate(0.5, 1.0) * 100)
        flow = Flow(fid, route, begin*window, end*window, period)
        flows.append(flow)

    return flows


if __name__ == '__main__':
    runtime = Runtime()
    runtime.load_net('../data/net.net.xml')
    runtime.set_time(0, 18000)
    runtime.generate_routes(10, 5)
    runtime.generate_flows(2)
    xml = runtime.to_xml()
    print(xml)