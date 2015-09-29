


# district - start at a node, explore outwards until all found.
        
# a recursive algorithm is the obvious approach, is there an alternative?

bi_edges = [('A','B'),('B','C'),('D','C')]

def add_value(dictionary,key,value):
    """ adds the value to list of values for relevent key """
    if key in dictionary:
        dictionary[key].append(value)
    else:
        dictionary[key]=[value]
    

class Graph():
    def __init__(self,directed_edges,undirected_edges):
        self.directed={}
        self.undirected={}
        nodes = set([])
        for from_node,to_node in directed_edges:
            add_value(self.directed,from_node,to_node)
            nodes.add(from_node)
            nodes.add(to_node)
        for a,b in undirected_edges:
            add_value(self.undirected,a,b)
            add_value(self.undirected,b,a)
            nodes.add(a)
            nodes.add(b)
        self.nodelist = sorted(list(nodes))
                        
    def nodes(self):
        return self.nodelist

    def districts(self):
        """ returns sets of nodes mutually connected bi-directionally """
        assigned = set([]) # record which nodes have already been assigned a district
        result = []
        for node in self.undirected:
            if node not in assigned:
                district = [node]
                self._find_district(node,district)
                assigned.update(district)
                result.append(district)
        return result
            
    # re-write this so it returns a list rather than modifying the input one ...
    def _find_district(self,node,district):
        # find all the things bi-directionally connected to this one
        for dest in self.undirected[node]:
            if dest not in district:
                district.append(dest)
                self._find_district(dest,district)

                    

g = Graph([('Z','X'),('Z','Y'),('X','W'),('W','Y')],[('Z','Y')])
#g = Graph([],[('A','B'),('B','E'),('A','C'),('A','D'),('C','D'),('F','G'),('G','H'),('G','I')])                

print(g.districts())

    

