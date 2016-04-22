# code based on "On the Identification of Causal Effects", Tian & Pearl
import graphviz as gv
import subprocess
import os

def add_value(dictionary,key,value = None):
    """ adds the value to list of values for relevent key """
    if key in dictionary:
        if value is not None:
            dictionary[key].append(value)
    else:
        if value is None:
            dictionary[key] = []
        else:
            dictionary[key]=[value]

def intersection(lst1,lst2):
    return [x for x in lst1 if x in lst2]

def math_sum(variables,exclude):
    sum_over  = variables - exclude
    if len(sum_over) > 0:
        return "\sum_{%s}" % ",".join(sum_over)
    return ""

def math_sum2(variables,exclude):
    """ takes in two sets and returns a tex expression summing over variables \ exclude """
    sum_over = variables - exclude
    if len(sum_over) > 0:
        return "\sum_{%s}"%",".join(sum_over)
    return ""

def math_condition(left,*right):
    result = "P("+left
    first = True
    for cond in right:
        print(cond)
        if len(cond) > 0:
            if first:
                result+="|"
                first = False
            else:
                result+=","
            result+=",".join(cond)
    result +=")"
    return result
            
    
def edge_to_string(edge):
    return edge[0]+"-"+edge[1]

def edge_lists_to_string(dg,ug):
    return ",".join([edge_to_string(edge) for edge in dg])+":"+",".join([edge_to_string(edge) for edge in ug])

def read_undirected(file):
    """ reads the output of listg -t (see below). Creates an edge list for each graph
    Graph 1, order 3.
  0 : 2;
  1 : 2;
  2 : ;

Graph 2, order 3.
  0 : 1 2;
  1 : 2;
  2 : ;
  """
    graphs = []
    edges = []
    for line in file.readlines():
        if line.startswith("Graph"): 
            if len(edges) > 0:
                graphs.append(edges)
            edges = []
        elif ":" in line:

            line = [x.strip() for x in line.strip(";\n").split(":")]
            source = line[0]
            dest = line[1].split(" ")
            for d in dest:
                if len(d) > 0:
                    edges.append((source,d))
    graphs.append(edges)
    return graphs


def read_directed(file):
    """ reads the T files producted by directg. Creates an edge list for each graph"""
    graphs = []
    for line in file.readlines(): # one graph per line
        edges = []
        line = line.strip("\n").split(" ")
        n_edges = int(line[1])
        for s in range(2,2*n_edges+1,2):
            edge = (line[s],line[s+1])
            edges.append((edge))
        graphs.append(edges)
    return graphs 
         
class Graph():
    def __init__(self, graph = None):
        if graph:
            self.children = graph.children.copy()
            self.parents = graph.parents.copy()
            self.siblings = graph.siblings.copy()
        else:
            self.children = {} # each node in graph points to a list of children
            self.parents = {} # each node in the graph points to a list of parents
            self.siblings = {} # each node in the graph points to a list of siblings (ie nodes connected with a bidirectional link

    @classmethod
    def from_edge_lists(cls,directed_edges,undirected_edges):
        g = cls()
        g.load_edge_lists(directed_edges,undirected_edges)
        return g

    def load_edge_lists(self,directed_edges,undirected_edges):
        for from_node,to_node in directed_edges:
            self.add_edge(from_node,to_node)
            
        for a,b in undirected_edges:
            self.add_bidirected_edge(a,b)
        

    def add_edge(self,from_node, to_node):
        add_value(self.children,from_node,to_node)
        add_value(self.children,to_node)
        add_value(self.parents, to_node, from_node)
        add_value(self.parents,from_node)
        add_value(self.siblings,to_node)
        add_value(self.siblings,from_node)
        self.verify_structure()  

    def add_bidirected_edge(self, a,b):
        add_value(self.siblings,a,b)
        add_value(self.siblings,b,a)
        add_value(self.children,a)
        add_value(self.children,b)
        add_value(self.parents,a)
        add_value(self.parents,b)
        self.verify_structure() 

    def verify_structure(self):
        assert(self.children.keys() == self.parents.keys() == self.siblings.keys()) # all nodes should be in all dicts.
                        
    def nodes(self):
        return set(self.children.keys())

    def topological_sort(self):
        sorted_nodes = []
        unsorted_graph = self.parents.copy()
        # run while until the unsorted graph is empty
        while len(unsorted_graph) > 0:
            acyclic = False
            # go through all the node/edges pairs in the unsorted graph. Note .items() takes a copy
            for node,edges in list(unsorted_graph.items()):
                # if a set of edges doesn't contain any nodes that are still in the unsorted graph, remove that pair and append to sorted graph
                no_unprocessed = True
                for edge in edges:
                    if edge in unsorted_graph:
                        no_unprocessed = False
                if no_unprocessed:
                    acyclic = True
                    del unsorted_graph[node]
                    sorted_nodes.append(node)
            if not acyclic:
                raise RuntimeError("a cyclic dependency occured")
        return sorted_nodes

    def ancestors(self,nodes):
        """ find all the ancestors of a set of nodes (including the nodes themselves)"""
        ancestors = set(nodes)
        expanded = True
        while expanded:
            parents = set()
            found = len(ancestors)
            for node in nodes:
                parents.update(self.parents[node])
            ancestors.update(parents)
            expanded = len(ancestors) > found
            nodes = parents
        return ancestors

    def parentsOf(self,nodes):
        """ returns the set of the parents of all the specified nodes"""
        result = set([])
        for node in nodes:
            result.update(self.parents[node])
        return result

    def subgraph(self,nodes):
        """ returns a new graph containing only those nodes in the specified list and any edges connecting them """
        sg = Graph()
        for node in nodes:
            sg.children[node] = intersection(self.children[node],nodes)
            sg.parents[node] = intersection(self.parents[node],nodes)
            sg.siblings[node] = intersection(self.siblings[node],nodes)
        sg.verify_structure()
        return sg

    def c_subgraph(self,nodes):
        """ returns a new graph containing specified nodes (S), the parents of the specified nodes (W) and links connecting nodes in S or from W to S.
            The resulting graph will have no edges (bi-directed or otherwise entering into nodes in W"""
        sg = self.subgraph(nodes)
        for p in self.parentsOf(nodes):
            sg.parents[p] = []
            sg.siblings[p] = []
            sg.children[p] = intersection(self.children[p],nodes)
        sg.verify_structure()
        return sg
                             
    def districts(self):
        """ returns list of sets of nodes mutually connected bi-directionally """
        assigned = set([]) # record which nodes have already been assigned a district
        result = []
        for node in self.siblings:
            if node not in assigned:
                district = set([node])
                self._find_district(node,district)
                assigned.update(district)
                result.append(district)
        return result
            
    # re-write this so it returns a list rather than modifying the input one ...
    def _find_district(self,node,district):
        # find all the things bi-directionally connected to this one
        for dest in self.siblings[node]:
            if dest not in district:
                district.add(dest)
                self._find_district(dest,district)

    def c_factor(self,component):
        result = ""
        order = self.topological_sort()
        for variable in component:
            cond = order[0:order.index(variable)]
            result += math_condition(variable,cond)
        return result
        
    def to_graphviz(self):
        g = gv.Digraph(format='svg')
        for node, children in self.children.items():
            for child in children:
                g.edge(node,child)
            for sibling in self.siblings.get(node):
                if sibling > node:
                    g.edge(node,sibling,color="blue",dir="none") # bidirected edges are not supported but we can display them without an arrow
        return g
        

#TODO save as graphml files
def generate_all_causal_graphs(nauty_directory,num_nodes):
    os.makedirs("graphs",exist_ok=True)
    cd = os.getcwd()
    os.chdir(nauty_directory)
    simple = cd+"/"+"graphs/s{0}".format(num_nodes)
    simple_connected = cd+"/"+"graphs/c{0}".format(num_nodes)
    directed_connected =  cd+"/"+"graphs/d{0}".format(num_nodes)
    causal = cd+"/"+"graphs/causal{0}".format(num_nodes)
    subprocess.call(["./geng",str(num_nodes),simple])
    subprocess.call(["./listg","-t",simple,simple])
    subprocess.call(["./geng","-c" ,str(num_nodes),simple_connected])
    subprocess.call(["./directg","-o","-T",simple_connected,directed_connected])
    with open(directed_connected, 'r') as d, open(simple, 'r') as u, open(causal,'w') as o:
        graphs = []
        directed_graphs = read_directed(d)
        # TODO delete cyclic graphs here would be more efficient
        undirected_graphs = read_undirected(u)
        for dg in directed_graphs:
            for ug in undirected_graphs:
               o.write(edge_lists_to_string(dg,ug))
               o.write("\n")
               graphs.append(Graph.from_edge_lists(dg,ug))
        print (len(graphs), "causal graphs generated")
        return graphs
    

        
def identify(y,x,P,G):
    """ Determine if P(y|do(x)) is identifiable. x is a set of variables on which to intervene, y is the target set, G is a graph."""
    V = G.nodes() # the set of nodes in the graph

    if len(x) == 0: # if the intervention set is empty
        return math_sum(V,y)+P # just sum over all the non-target variables

    ancestors = G.ancestors(y)
    if len(V - ancestors) > 0: # If there are some nodes in the graph that are not ancestors of y, then they are irrelevent. Marginalize them out
        return identify(y,x.intersection(ancestors),math_sum(V,ancestors)+P,G.subgraph(ancestors))

    GVX = G.c_subgraph(V-x) # G[V\X]
    
    W = (V-x)-GVX.ancestors(y) # intervention on these nodes does not effect Y given intervention on X. 
    if len(W) > 0:
        return identify(y,x.union(W),P,G) 

    components = G.subgraph(V-x).districts() #D(G[V\X])

    if len(components) > 1: # problem decomposes
        result = math_sum(V,y.union(x))
        for c in components:
            result += "("+identify(c,V-c,P,G)+")"
        return result

    else:
        S = V-x
        dg = G.districts() #D(G)
        
        if len(dg) == 1:
            raise IdentifyException(V,V.intersection(S))
        
        if S in dg: # S is a component of G.
            return math_sum(S,y)+G.c_factor(S)

        # find the component of G of which S is a subset. # will throw an exception if no such subset is found.
        S_prime = next(c for c in dg if S.issubset(c))
        P = ""
        order = G.topological_sort()
        for variable in S_prime:
            prior = set(order[0:order.index(variable)])
            cond1 = ",".join(prior.intersection(S_prime))
            cond2 = ",".join(prior - S_prime)
            P += math_condition(variable,cond1,cond2)
        return identify(y,x.intersection(S_prime),P,G.subgraph(S_prime))
        

class IdentifyException(Exception):
    def __init__(self, set1,set2):
        self.set1 = set1
        self.set2 = set2
    def __str__(self):
        return "Hedge:"+str(self.set1)+","+str(self.set2)
       

        
## identifyable (Pearl p92)
#gi1 = Graph.from_edge_lists([("X","Y")],[])
#gi2 = Graph.from_edge_lists([("X","Y"),("X","Z"),("Z","Y")],[("Z","Y")])
#gi3 = Graph.from_edge_lists([("X","Y"),("Z","X"),("Z","Y")],[("Z","Y")])
#gi4 = Graph.from_edge_lists([("X","Y"),("Z","X"),("Z","Y")],[("Z","X")])
#gi5 = Graph.from_edge_lists([("X","Z"),("Z","Y")],[("X","Y")])
#gi6 = Graph.from_edge_lists([("X","Z1"),("X","Y"),("Z1","Y"),("Z1","Z2"),("Z2","Y")],[("X","Z2")])
#gi7 = Graph.from_edge_lists([("X","Z1"),("Z1","Y"),("Z2","X"),("Z2","Z1"),("Z2","Z3"),("Z3","Y")],[("X","Y"),("X","Z2"),("Z2","Y"),("X","Z3")])
#ident_graphs = [gi1,gi2,gi3,gi4,gi5,gi6,gi7]
#
#
## not identifiable (Pearl p94)
#gna = Graph.from_edge_lists([("X","Y")],[("X","Y")])
#gnb = Graph.from_edge_lists([("X","Z"),("Z","Y")],[("X","Z")])
#gnc = Graph.from_edge_lists([("X","Y"),("X","Z"),("Z","Y")],[("X","Z")])
#gnd = Graph.from_edge_lists([("X","Y"),("Z","Y")],[("X","Z"),("Z","Y")])
#gne = Graph.from_edge_lists([("X","Y"),("Z","X")],[("X","Z"),("Z","Y")])
#gnf = Graph.from_edge_lists([("X","Z"),("Z","Y")],[("X","Y"),("Z","Y")])
#gng = Graph.from_edge_lists([("X","Z1"),("Z1","Y"),("Z2","Y")],[("X","Z2"),("Z1","Z2")])
#gnh = Graph.from_edge_lists([("X","W"),("W","Y"),("Z","X")],[("X","Z"),("X","Y"),("Z","W"),("Z","Y")])
#non_ident_graphs = [gna,gnb,gnc,gnd,gne,gnf,gng,gnh] 
#  
## some other graphs.
#g_non1 = Graph.from_edge_lists([("X","Y")],[("X","Y")])
#g3 = Graph.from_edge_lists([("X","Z1"),("X","Y"),("Z1","Y"),("Z1","Z2"),("Z2","Y")],[("X","Z2"),("Z1","Y")])
#g4 = Graph.from_edge_lists([("X1","X2"),("X2","X3"),("X3","X4"),("X4","Y")],[("X1","X3"),("X3","Y"),("X2","X4")])
#g5 = Graph.from_edge_lists([("X","Z1"),("Z1","Z2"),("Z2","Y")],[("X","Z2")])
#
#gfig6a = Graph.from_edge_lists([("X","Z"),("Z","Y"),("W1","W2"),("W2","X")],[("X","W1"),("W1","Z")])
#
#
#result = identify(set(['Y']),set(['X']),"P("+",".join(gi1.nodes())+")",gi7)
#print(result)



##components = gfig6a.districts()
##factors = gfig6a.c_factors(components)
##for f in factors:
##    print_cfactor(f)

#print(gfig6a.topological_sort())

##print("IDENTIFIABLE")
##for graph in ident_graphs:
##    print("RESULT:",graph.query("X","Y"))
##
##print("NON_IDENTIFIABLE")
##for graph in non_ident_graphs:
##    print("RESULT",graph.query("X","Y"))


#print("sorted",g4.topological_sort())

#g = Graph([('Z','X'),('Z','Y'),('X','W'),('W','Y')],[('Z','Y')])
#g = Graph([],[('A','B'),('B','E'),('A','C'),('A','D'),('C','D'),('F','G'),('G','H'),('G','I')])                

#print(g4.districts())
#print (g4.ancestors(["X3"]))
#print (g3.ancestors(["Z1"]))




    

