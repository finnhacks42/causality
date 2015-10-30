# code based on "On the Identification of Causal Effects", Tian & Pearl
import matplotlib.pyplot as plt
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
        
        assert(self.children.keys() == self.parents.keys() == self.siblings.keys()) # all nodes should be in all dicts.

    def add_bidirected_edge(self, a,b):
        add_value(self.siblings,a,b)
        add_value(self.siblings,b,a)
        add_value(self.children,a)
        add_value(self.children,b)
        add_value(self.parents,a)
        add_value(self.parents,b)
        assert(self.children.keys() == self.parents.keys() == self.siblings.keys()) # all nodes should be in all dicts.      
                        
    def nodes(self):
        return list(self.children.keys())

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
        """ find all the ancestors of a set of nodes"""
        ancestors = set()
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

    def An(self,nodes):
        result = self.ancestors(nodes)
        result.update(nodes)
        return result

    def subgraph(self,nodes):
        """ returns a new graph containing only those nodes in the specified list and any edges connecting them """
        sg = Graph()
        for node in nodes:
            sg.children[node] = intersection(self.children[node],nodes)
            sg.parents[node] = intersection(self.parents[node],nodes)
            sg.siblings[node] = intersection(self.siblings[node],nodes)
        return sg
            
        
                       
    def districts(self):
        """ returns sets of nodes mutually connected bi-directionally """
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
                

    def indentify2(self,y,x,P,G):
        V = G.nodes()
        if len(x) == 0:
            return math_sum(V,y)+P


        


    def identify(self,C,T,Q):
        """ C is a district (ie a set of nodes), T is another district, Q relates to calculating the expression (None for now)"""
        assert(C <= T <= set(self.nodes()))
        print("Identify",C,T)
        G_T = self.subgraph(T) # assert that this graph has only one component ...
        A = G_T.An(C)
        print ("A:",A)
        if A == C:
            return prepended(Q,("sum",T-C))
        if A == T:
            return None # fail
        if C < A < T:
            G_A = self.subgraph(A)
            components = G_A.districts()
            T2 = next(c for c in components if C <= c) # I think I should assert that T2 is not empty here ...
            Q2 = prepended(Q,("sum",T-A))
            return self.identify(C,T2,Q2)
            
        else:
            raise RuntimeError("Unexpected state in identify")

    def get_parents(self,variables):
        result = set([])
        for v in variables:
            result.update(self.parents[v])
        return result        

    def c_factors(self,components):
        """ compute the c-factors (Q_i) for each component """
        order = self.topological_sort()
        factors = []
        for c in components:
            c_factor = []
            for variable in c:
                v_i = order[0:order.index(variable)+1]
                g_v_i = self.subgraph(v_i)
                T_i = next(s for s in g_v_i.districts() if variable in s)
                cond = self.get_parents(T_i)
                if variable in cond:
                    cond.remove(variable)
                term = (variable,cond)
                c_factor.append(term)
            factors.append(c_factor)
        return factors
                
                

    def find_component_containing(self,components,variable):
        for indx,s in enumerate(components):
            if variable in s:
                return (indx,s)
        return None
        
       

    def query(self,X,Y):
        """ tests if P(Y|do(X)) is identifiable. X and Y are single variables"""
        c_components = self.districts() # 1) find the components
        c_factors = self.c_factors(c_components)
        print("c-factors",c_factors)
        indx,S_x = self.find_component_containing(c_components,X)
        print ("S_x",S_x)
        Q_s_x = c_factors[indx] #[('f',c_factors[indx])] # this will be a list of terms
        g_x = self.subgraph([v for v in self.nodes() if v != X]) # subgraph of this graph with only X removed
        D = g_x.An(Y)
        print("D",D)
        D_x = D.intersection(S_x)
        print("D_x",D_x)
        g_dx = self.subgraph(D_x)
        Dterms = []
        for j,D_xj in enumerate(g_dx.districts()):
            print("D_x_",j,D_xj)
            # Compute Q[D_xj] from Q[S_x] by calling identify(D_xj,S_x,Q[S_x])
            Q_D_xj = self.identify(D_xj,S_x,Q_s_x) # should be a list of terms with some sums out the front...
            if not Q_D_xj:
                return None # not identifiable
            Dterms.append(Q_D_xj)

        del c_components[indx]
        del c_factors[indx]
        Si_terms = []
        for comp,factor in zip(c_components,c_factors):
            Si_terms.append((comp-D,factor))
        return (D-set(Y),Dterms,Si_terms)

def prepended(lst,value):
    result = lst[:]
    result.insert(0,value)
    return result


def to_math(result):
    text = ''
    if len(result[0]) > 0:
        text +="\sum_{%s}"%",".join(result[0])

    for qdxj in result[1]: #qdxj will be a number of sums followed by a list of factors
        text +="("
        for term in qdxj:
            if term[0] == "sum":
                if len(term[1]) > 0:
                    text +="\sum_{%s}"%",".join(term[1])
            else:
                text += factor_tostring(term)
        
    for term in result[2]:
        text +="("
        if len(term[0]) > 0:
            text +="\sum_{%s}"%",".join(term[0])
        for f in term[1]:
            text += factor_tostring(f)
        text += ")"
        
    return text

def factor_tostring(factor):
    print(factor)
    text = ""
    text += "P("+factor[0]
    if len(factor[1]) > 0:
        text+="|"+",".join(factor[1])
    text += ")"
    return text

def display(math):
    ax = plt.axes([0,0,0.01,0.02]) #left,bottom,width,height
    ax.set_xticks([])
    ax.set_yticks([])
    plt.text(5,5,'$%s$' %math,size=20)
    plt.show()
        
# identifyable (Pearl p92)
gi1 = Graph.from_edge_lists([("X","Y")],[])
gi2 = Graph.from_edge_lists([("X","Y"),("X","Z"),("Z","Y")],[("Z","Y")])
gi3 = Graph.from_edge_lists([("X","Y"),("Z","X"),("Z","Y")],[("Z","Y")])
gi4 = Graph.from_edge_lists([("X","Y"),("Z","X"),("Z","Y")],[("Z","X")])
gi5 = Graph.from_edge_lists([("X","Z"),("Z","Y")],[("X","Y")])
gi6 = Graph.from_edge_lists([("X","Z1"),("X","Y"),("Z1","Y"),("Z1","Z2"),("Z2","Y")],[("X","Z2")])
gi7 = Graph.from_edge_lists([("X","Z1"),("Z1","Y"),("Z2","X"),("Z2","Z1"),("Z2","Z3"),("Z3","Y")],[("X","Y"),("X","Z2"),("Z2","Y"),("X","Z3")])
ident_graphs = [gi1,gi2,gi3,gi4,gi5,gi6,gi7]


# not identifiable (Pearl p94)
gna = Graph.from_edge_lists([("X","Y")],[("X","Y")])
gnb = Graph.from_edge_lists([("X","Z"),("Z","Y")],[("X","Z")])
gnc = Graph.from_edge_lists([("X","Y"),("X","Z"),("Z","Y")],[("X","Z")])
gnd = Graph.from_edge_lists([("X","Y"),("Z","Y")],[("X","Z"),("Z","Y")])
gne = Graph.from_edge_lists([("X","Y"),("Z","X")],[("X","Z"),("Z","Y")])
gnf = Graph.from_edge_lists([("X","Z"),("Z","Y")],[("X","Y"),("Z","Y")])
gng = Graph.from_edge_lists([("X","Z1"),("Z1","Y"),("Z2","Y")],[("X","Z2"),("Z1","Z2")])
gnh = Graph.from_edge_lists([("X","W"),("W","Y"),("Z","X")],[("X","Z"),("X","Y"),("Z","W"),("Z","Y")])
non_ident_graphs = [gna,gnb,gnc,gnd,gne,gnf,gng,gnh] 
  
# some other graphs.
g_non1 = Graph.from_edge_lists([("X","Y")],[("X","Y")])
g3 = Graph.from_edge_lists([("X","Z1"),("X","Y"),("Z1","Y"),("Z1","Z2"),("Z2","Y")],[("X","Z2"),("Z1","Y")])
g4 = Graph.from_edge_lists([("X1","X2"),("X2","X3"),("X3","X4"),("X4","Y")],[("X1","X3"),("X3","Y"),("X2","X4")])
g5 = Graph.from_edge_lists([("X","Z1"),("Z1","Z2"),("Z2","Y")],[("X","Z2")])

gfig6a = Graph.from_edge_lists([("X","Z"),("Z","Y"),("W1","W2"),("W2","X")],[("X","W1"),("W1","Z")])

result = gi1.query("X","Y") # seems wrong way round so far ...

math = to_math(result)
display(math)
    



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




    

