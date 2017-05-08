''' BayesNetNode.py


S. Tanimoto, 11 Jan 2009

This file provides:

-- Node class definitions.
'''

import Tkinter

class Node:

    name_node_hash = {}
    def __init__(self, name, desc, x, y, poss_vals = ['True', 'False']):
        self.name = name
        self.desc = desc
        self.x = x
        self.y = y
        Node.name_node_hash[name] = self
        self.parents = []
        self.children = []
        self.possible_values = poss_vals
        self.p = {} # Hash, so that number of parents can change.
        self.current_prob = {} # current probabilities, one for each
            # possible_value.
        default_prob = 1.0 / len(poss_vals)
        for pv in poss_vals:
            self.current_prob[pv] = default_prob
            self.p[self.name+"="+pv] = default_prob

    def get_prior(self, poss_val):
        return self.p[self.name+"="+poss_val]
        
    def set_prior(self, poss_val, prob):
        self.p[self.name+"="+poss_val] = prob
        self.current_prob[poss_val] = prob
        
    def add_parent(self, parent_name):
        p = Node.name_node_hash[parent_name]
        self.parents.append(p)
        return p
                
    def add_child(self, child_name):
        c = Node.name_node_hash[child_name]
        self.children.append(c)
        return c        
        
class BayesNet:
    def __init__(self, name):
        self.name = name
        self.nodes = []
        self.input_nodes = []

    def add_node(self, node):
        self.nodes.append(node)
        from InputNode import Input_Node
        if isinstance(node, Input_Node):
            self.input_nodes.append(node)

    def get_input_nodes(self):
        return self.input_nodes


