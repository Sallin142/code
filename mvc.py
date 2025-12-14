"""
Minimum Vertex Cover Implementation

reference:  
https://pages.cs.wisc.edu/~shuchi/courses/787-F09/scribe-notes/lec9.pdf [1] 
https://www2.cs.sfu.ca/~hangma/pub/ijcai19.pdf [2]

"""

from collections import defaultdict
from pulp import LpProblem, LpMinimize, LpVariable, value


class UnweightedGraph:
    # simple graph for computing minimum vertex cover
    # Given a graph G and any maximal matching M on G, we can find a vertex cover for G of size at most 2|M|. [1]
    
    def __init__(self, num_vertices):
    
        # initialize graph with specified number of vertices
        self.num_vertices = num_vertices
        self.adjacency_list = defaultdict( list)
    
    def add_edge(self, vertex_a, vertex_b):
        
        # add an undirected edge between two vertices
        self.adjacency_list[vertex_a].append(vertex_b)
    
    def get_minimum_vertex_cover(self):
        """
        Algorithm: Maximal matching-based approximation
        - finds a maximal matching
        - includes both endpoints of each matched edge
        - guarantees size ≤ 2 * optimal
        """
        marked = [False] * self.num_vertices
        cover = []
        
        # find maximal matching
        for u in range(self.num_vertices ):
            if marked[u ]:
                continue
            
            # try to match u with an unmarked neighbor
            for v in self.adjacency_list[u]:
                if not marked[v]:
                    marked[u] =True
                    marked[v] = True
                    cover.append(u)
                    cover.append(v)
                    break
        return cover

# Vertex in a weighted graph [2]
class WeightedGraphVertex:
    
    def __init__(self, vertex_id ):

        self.vertex_id = vertex_id
        self.neighbors = {}
    
    # add a neighbor with specified edge weight
    def add_neighbor(self, neighbor_vertex , weight = 0):
        self.neighbors[neighbor_vertex ] = weight
    
    def get_neighbors(self):
        return self.neighbors.keys()
    
    def get_edge_weight(self, neighbor_vertex):
        return self.neighbors.get( neighbor_vertex , 0)
    
    def __str__(self):
        neighbor_ids = [v.vertex_id for v in self.neighbors ]
        return f"Vertex {self.vertex_id}: neighbors = {neighbor_ids}"


class WeightedGraph:

    # weighted graph for computing weighted minimum vertex cover
    
    def __init__(self, initial_vertices=None):

        self.vertices = {}
        if initial_vertices:
            for vertex_id in initial_vertices:
                self.add_vertex( vertex_id)
    
    def add_vertex(self, vertex_id):
        if vertex_id not in self.vertices:
            self.vertices[vertex_id] = WeightedGraphVertex(vertex_id)
    
    def add_edge(self, vertex_a, vertex_b, weight=0):
        # check both vertices exist
        self.add_vertex(vertex_a)
        self.add_vertex(vertex_b)
        
        # add bidirectional edge
        v1 = self.vertices[vertex_a]
        v2 = self.vertices[vertex_b ]
        v1.add_neighbor(v2, weight)
        v2.add_neighbor(v1, weight)
    
    def get_vertex(self, vertex_id):
        return self.vertices.get(vertex_id )
    
    def get_all_vertex_ids(self):
        return self.vertices.keys()
    
    def __iter__(self):
        return iter(self.vertices.values())
    
    def __len__(self ):
        return len(self.vertices)

Graph = UnweightedGraph
UnweightedGraph.addEdge = UnweightedGraph.add_edge
UnweightedGraph.getVertexCover =UnweightedGraph.get_minimum_vertex_cover