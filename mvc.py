from collections import defaultdict
import math
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, LpMinimize, value

class Graph:
 
    def __init__(self, vertices):
        self.V = vertices
        self.graph = defaultdict(list)
 
    def addEdge(self, u, v):
        self.graph[u].append(v)
 
    def getVertexCover(self):
        visited = [False] * (self.V)
         
        for u in range(self.V):
            if not visited[u]:
                 
                for v in self.graph[u]:
                    if not visited[v]:
                        visited[v] = True
                        visited[u] = True
                        break
 
        mvc_set = []
        for j in range(self.V):
            if visited[j]:
                mvc_set.append(j)
        
        return mvc_set

class Vertex:
    def __init__(self, node):
        self.id = node
        self.adjacent = {}

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.keys()  

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]

class WeightedGraph:
    def __init__(self, vertices):
        self.vert_dict = {}
        self.num_vertices = 0
        for vertex in vertices:
            self.add_vertex(vertex)

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node):
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def add_edge(self, frm, to, cost = 0):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

    def get_vertices(self):
        return self.vert_dict.keys()