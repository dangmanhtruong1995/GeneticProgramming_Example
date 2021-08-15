import os
from pdb import set_trace
import numpy as np
from copy import deepcopy as deepcopy
import sys
import random
import math
import pandas as pd
import matplotlib.pyplot as plt

from print_binary_tree import TreeNode, printHeapTree, printBTree

# References:
# [1] William B. Langdon, Riccardo Poli, Nicholas F. McPhee, John R. Koza.
# Genetic Programming: An Introduction and Tutorial, with a Survey of Techniques and Applications
# Computational Intelligence: A Compendium (2008),  Studies in Computational Intelligence, vol 115. Springer, Berlin, Heidelberg

MAXIMUM_VALUE = 999999

def is_number(val):
    if isinstance(val, int) or isinstance(val, float):
        return True
    else:
        return False

class Node:
    def __init__(self, data=None):
        self.data = data
        self.left = None
        self.right = None
        self.children = 0 # Number of children, used for easy random selection
        self.parent = None
        self.parent_type = None

    def add_left(self, node):
        # assert (self.left is None), \
            # 'add_left requires that left node must be None first'
        self.left = node
        self.left.parent = self
        self.parent_type = 'left'

    def add_right(self, node):
        # assert (self.right is None), \
            # 'add_right requires that right node must be None first'
        self.right = node
        self.right.parent = self
        self.parent_type = 'right'

    def remove_left(self):
        if (self.left is not None):
            self.children -= (self.left.children + 1)
            self.left.parent = None
            self.left = None
       
    def remove_right(self):
        if (self.right is not None):
            self.children -= (self.right.children + 1)
            self.right.parent = None
            self.right = None

    def is_leaf(self):
        if (self.left is None) and (self.right is None):
            return True
        else:
            return False

    def print_tree(self):
        printBTree(self,lambda n:(str(n.data),n.left,n.right))
        print('')

   
    def num_of_nodes(self):
        if self.left is None:
            n_node_left = 0
        else:
            n_node_left = self.left.num_of_nodes()
        if self.right is None:
            n_node_right = 0
        else:
            n_node_right = self.right.num_of_nodes()
        return n_node_left + n_node_right + 1

    def update_children(self):
        if self.left is None:
            n_children_left = 0
        else:
            self.left.update_children()
            n_children_left = self.left.children + 1
        if self.right is None:
            n_children_right = 0
        else:
            self.right.update_children()
            n_children_right = self.right.children + 1
        self.children = n_children_left + n_children_right

def init_tree(func_list, term_list, max_depth, method):
    n_func = len(func_list)
    n_term = len(term_list)
    n_all = n_func + n_term
    rand_int = np.random.randint(1, n_all+1, size=(1)).tolist()[0]
    if (max_depth == 0) or ((method=='grow') and (rand_int > n_func)):
        r1 = np.random.randint(0, n_term, size=(1)).tolist()[0]
        expr = Node(term_list[r1])
        return expr
    else:
        r1 = np.random.randint(0, n_func, size=(1)).tolist()[0]
        func = Node(func_list[r1])
        arg1 = init_tree(func_list, term_list, max_depth-1, method)
        arg2 = init_tree(func_list, term_list, max_depth-1, method)
        func.add_left(arg1)
        func.add_right(arg2)
        return func

def random_node_util(root, count):
    # https://www.geeksforgeeks.org/select-random-node-tree-equal-probability/
    if root is None:
        return 0
    if root.left is None:
        left_children = 0
    else:
        left_children = root.left.children + 1
    if (count == left_children):
        return root
    elif (count < left_children):
        return random_node_util(root.left, count)
    return random_node_util(root.right, count - left_children - 1)

def random_node(root):
    root.update_children()
    count = np.random.randint(0, root.children+1, size=(1)).tolist()[0]
    # print(count)
    # set_trace()
    return random_node_util(root, count)

def random_node_prob(root, p_leaf):
    # p_leaf: probability to choose leaf    
    assert (0 <= p_leaf) and (p_leaf <= 1), 'p_leaf must be in range [0,1]'
    rand_prob = random.uniform(0, 1)
    if rand_prob < p_leaf:
        # Only choose leaf
        # print('Only choose leaf')
        while True:
            node = random_node(root)
            if node.is_leaf() is True:
                break
    else:
        # Only choose node
        # print('Only choose node')
        if root.num_of_nodes() == 1:
            return root # Special case, else the while loop will never end
        count_idx = 0
        while True:            
            node = random_node(root)
            # set_trace()
            count_idx += 1
            if count_idx > 100:
                set_trace()
            if node.is_leaf() is False:
                break
    return node

def expr_eval(func_list, term_list, tree, var_dict):
    data = tree.data
    if data in term_list:
        if is_number(data):
            return data
        else:
            return var_dict[data]
    else:
        left_val = expr_eval(func_list, term_list, tree.left, var_dict)
        right_val = expr_eval(func_list, term_list, tree.right, var_dict)
        if data == '+':
            return left_val + right_val
        elif data == '-':
            return left_val - right_val
        elif data == '*':
            return left_val * right_val
        elif data == '/':
            if right_val != 0:
                return left_val / (right_val * 1.0)
            else:
                return 0
        elif data == 'sqrt':
            if left_val > 0:
                return math.sqrt(left_val)
            else:
                return 0
        elif data == '^':
            if right_val >= 0:
                return np.power(left_val, right_val)
            else:
                return 1
        elif data == 'sin':
            return np.sin(left_val * right_val)
        elif data == 'cos':
            return np.cos(left_val * right_val)
    pass

class FitnessEvaluator:
    def __init__(self, gp_helper, file_name):
        df = pd.read_csv(file_name)        
        self.gp_helper = gp_helper
        self.df = df
    
    def fitness(self, cand):
        gp_helper = self.gp_helper
        df = self.df
        func_list = gp_helper.func_list
        term_list = gp_helper.term_list
        n_func = gp_helper.n_func
        n_term = gp_helper.n_term
        n_all = n_func + n_term
        
        n_inst = len(df)
        f_avg = 0
        for idx in range(n_inst):
            row = df.iloc[idx, :]
            try:
                f_val = np.abs(expr_eval(func_list, term_list, cand, row) - row['target'])
            except:
                set_trace()            
            if math.isnan(f_val):
                f_val = MAXIMUM_VALUE
            if f_val > MAXIMUM_VALUE:
                f_val = MAXIMUM_VALUE
            f_avg += f_val 

            ld = 0.001
            f_avg += ld*cand.num_of_nodes() # Favors small trees over large trees
            
        if f_avg < 0:
            set_trace()
        f_avg /= (1.0 * n_inst)
        return f_avg
        

class GPHelper:
    def __init__(self, func_list, term_list, p_gauss_mutate):
        n_func = len(func_list)
        n_term = len(term_list)
        self.func_list = func_list
        self.term_list = term_list
        self.n_func = n_func
        self.n_term = n_term
        self.p_gauss_mutate = p_gauss_mutate

    def point_mutation(self, cand):
        # assert isinstance(cand, Node), 'cand must be of class Node'

        func_list = self.func_list
        term_list = self.term_list
        n_func = self.n_func
        n_term = self.n_term
        n_all = n_func + n_term
        p_gauss_mutate = self.p_gauss_mutate

        new_cand = deepcopy(cand)
        node = random_node(new_cand)
        rand_int = np.random.randint(0, n_all, size=(1)).tolist()[0]
        if node.data in term_list:
            # Mutate into new terminal
            if is_number(node.data):
                rand_prob = random.uniform(0, 1)
                if rand_prob < p_gauss_mutate:                
                    rand_num = np.random.normal()
                    node.data += rand_num
                    term_list.append(node.data)
                    n_term += 1
                    self.term_list = term_list
                    self.n_term = n_term
                else:
                    rand_int = np.random.randint(0, n_term, size=(1)).tolist()[0]
                    node.data = term_list[rand_int]
            else:
                rand_int = np.random.randint(0, n_term, size=(1)).tolist()[0]
                node.data = term_list[rand_int]                
        else:
            # Mutate into new function
            rand_int = np.random.randint(0, n_func, size=(1)).tolist()[0]
            node.data = func_list[rand_int]
        return new_cand

    def subtree_mutation(self, cand):
        # assert isinstance(cand, Node), 'cand must be of class Node'

        func_list = self.func_list
        term_list = self.term_list
        n_func = self.n_func
        n_term = self.n_term
        n_all = n_func + n_term

        max_depth = 3 # Small mutation tree
        method = 'grow'

        new_cand = deepcopy(cand)
        if new_cand.num_of_nodes() == 1:
            random_tree = init_tree(func_list, term_list, max_depth, method)
            rand_prob = random.uniform(0, 1)
            if rand_prob < 0.5:
                new_cand.add_left(random_tree)
            else:
                new_cand.add_right(random_tree)
            return new_cand
        else:
            count_idx = 0
            while True:
                # print('in WHILE')
                count_idx += 1
                node = random_node(new_cand)
                if count_idx > 100:
                    set_trace()
                if node.parent is not None:
                    # print('BREAK')
                    break
            
            random_tree = init_tree(func_list, term_list, max_depth, method)
            if node.parent.left is node:
                node.parent.add_left(random_tree)
            else:
                node.parent.add_right(random_tree)
            return new_cand

    def random_removal(self, cand):
        func_list = self.func_list
        term_list = self.term_list
        n_func = self.n_func
        n_term = self.n_term
        n_all = n_func + n_term
        
        new_cand = deepcopy(cand)
        if new_cand.num_of_nodes() == 1:
            return new_cand
        else:
            count_idx = 0
            while True:
                # print('in WHILE')
                count_idx += 1
                node = random_node(new_cand)
                if count_idx > 100:
                    set_trace()
                if node.parent is not None:
                    # print('BREAK')
                    break
            parent = node.parent
            rand_int = np.random.randint(0, n_term, size=(1)).tolist()[0]
            new_node = Node(term_list[rand_int])
            if parent.left is node:
                parent.add_left(new_node)
            else:
                parent.add_right(new_node)
            return new_cand

    def subtree_crossover(self, cand_1, cand_2):
        assert isinstance(cand_1, Node), 'cand_1 must be of class Node'
        assert isinstance(cand_2, Node), 'cand_2 must be of class Node'

        func_list = self.func_list
        term_list = self.term_list
        n_func = self.n_func
        n_term = self.n_term
        n_all = n_func + n_term

        new_cand_1 = deepcopy(cand_1)
        new_cand_2 = deepcopy(cand_2)

        if (new_cand_1.num_of_nodes() == 1) or (new_cand_2.num_of_nodes() == 1):
            # Special case, leave it for now
            # TODO: Write more 
            return (new_cand_1, new_cand_2)

        p_leaf = 0.1
        count_idx = 0
        while True:
            node_1 = random_node_prob(new_cand_1, p_leaf)
            count_idx += 1
            if count_idx > 100:
                set_trace()
            if node_1.parent is not None:
                break
        count_idx = 0        
        while True:
            node_2 = random_node_prob(new_cand_2, p_leaf)
            count_idx += 1
            if count_idx > 100:
                set_trace()
            if node_2.parent is not None:
                break

        node_1_parent = node_1.parent
        node_2_parent = node_2.parent
        if node_1_parent.left is node_1:
            node_1_parent.left = node_2
            node_2.parent = node_1_parent
        else:
            node_1_parent.right = node_2
            node_2.parent = node_1_parent
        if node_2_parent.left is node_2:
            node_2_parent.left = node_1
            node_1.parent = node_2_parent
        else:
            node_2_parent.right = node_1
            node_1.parent = node_2_parent
        return (new_cand_1, new_cand_2)
    # set_trace()

def rws(fitness_list):
    # Roulette wheel selection
    n_cand = np.size(fitness_list)
    idx = np.random.choice(np.arange(np.size(fitness_list)), replace=True,
        p=fitness_list/fitness_list.sum())
    return idx

class GPOptimizer:
    def __init__(self, n_cand, max_depth, func_list, term_list,
            p_crossover, p_subtree_mutate, p_gauss_mutate):
        gp_helper = GPHelper(func_list, term_list, p_gauss_mutate)
        # Ramped-half-and-half initialization:
        # first half uses Grow method, last half uses Full method
        cand_list = []
        method = 'grow'
        for cand_idx in range(0, int(n_cand/2.0)):
            cand = init_tree(func_list, term_list, max_depth, method)
            cand_list.append(cand)
        method = 'full'
        for cand_idx in range(int(n_cand/2.0), n_cand):
            cand = init_tree(func_list, term_list, max_depth, method)
            cand_list.append(cand)
        
        self.gp_helper = gp_helper
        self.cand_list = cand_list
        self.p_crossover = p_crossover
        self.p_subtree_mutate = p_subtree_mutate

    def optimize(self, func, n_gen):
        gp_helper = self.gp_helper
        cand_list = self.cand_list
        p_crossover = self.p_crossover
        p_subtree_mutate = self.p_subtree_mutate
        n_cand = len(cand_list)        

        fitness_list = np.zeros((n_cand))
        # set_trace()
        for cand_idx in range(n_cand):
            cand = cand_list[cand_idx]
            fitness_list[cand_idx] = func.fitness(cand)

        best_global_f = MAXIMUM_VALUE
        best_global = None
        best_list = np.zeros((n_gen))
        avg_list = np.zeros((n_gen))
        for gen_idx in range(n_gen):
            print('Generation %d' % (gen_idx))

            # Crossover
            print('Crossover')
            for cand_idx in range(n_cand):
                rand_prob = random.uniform(0, 1)
                if rand_prob < p_crossover:
                    fitness_list_inv = np.array([1 / (f_val+0.001) for f_val in fitness_list.tolist()])
                    try:
                        cand_idx_1 = rws(fitness_list_inv)
                    except:
                        set_trace()
                    cand_idx_2 = rws(fitness_list_inv)
                    cand_1 = deepcopy(cand_list[cand_idx_1])
                    cand_2 = deepcopy(cand_list[cand_idx_2])
                    (new_cand_1, new_cand_2) = gp_helper.subtree_crossover(cand_1, cand_2)
                    
                    f_1 = func.fitness(new_cand_1)
                    f_2 = func.fitness(new_cand_2)
                    
                    if f_1 < fitness_list[cand_idx_1]:                    
                        cand_list[cand_idx_1] = new_cand_1
                        fitness_list[cand_idx_1] = f_1
                    if f_2 < fitness_list[cand_idx_2]:                    
                        cand_list[cand_idx_2] = new_cand_2
                        fitness_list[cand_idx_2] = f_2
             
            print('Subtree mutation')
            for idx in range(n_cand):
                rand_prob = random.uniform(0, 1)
                if rand_prob < p_subtree_mutate:
                    fitness_list_inv = np.array([1 / (f_val+0.001) for f_val in fitness_list.tolist()])
                    cand_idx = rws(fitness_list_inv)
                    cand = deepcopy(cand_list[cand_idx])
                    new_cand = gp_helper.subtree_mutation(cand)
                    f_1 = func.fitness(new_cand)
                    if f_1 < fitness_list[cand_idx]:
                        cand_list[cand_idx] = new_cand
                        fitness_list[cand_idx] = f_1

            print('Point mutation')
            for idx in range(n_cand):
                rand_prob = random.uniform(0, 1)
                if rand_prob < p_subtree_mutate:
                    fitness_list_inv = np.array([1 / (f_val+0.001) for f_val in fitness_list.tolist()])
                    cand_idx = rws(fitness_list_inv)
                    cand = deepcopy(cand_list[cand_idx])
                    new_cand = gp_helper.point_mutation(cand)
                    f_1 = func.fitness(new_cand)
                    if f_1 < fitness_list[cand_idx]:
                        cand_list[cand_idx] = new_cand
                        fitness_list[cand_idx] = f_1
                  
            print('Random removal')
            for idx in range(n_cand):
                rand_prob = random.uniform(0, 1)
                if rand_prob < p_subtree_mutate:
                    fitness_list_inv = np.array([1 / (f_val+0.001) for f_val in fitness_list.tolist()])
                    cand_idx = rws(fitness_list_inv)
                    cand = deepcopy(cand_list[cand_idx])
                    new_cand = gp_helper.random_removal(cand)
                    f_1 = func.fitness(new_cand)
                    if f_1 < fitness_list[cand_idx]:
                        cand_list[cand_idx] = new_cand
                        fitness_list[cand_idx] = f_1        
        
            best_idx = 0
            best = fitness_list[0]
            avg = 0
            for idx in range(n_cand):
                fitness = fitness_list[idx]
                if best > fitness:
                    best = fitness
                    best_idx = idx
                avg += fitness
            avg /= (1.0 * n_cand)
            best_list[gen_idx] = best
            avg_list[gen_idx] = avg
            
            if best_global_f > best:
                best_global_f = best
                best_global = cand_list[best_idx]

            
            # cand_list[best_idx].print_tree()
            # print(fitness_list[best_idx])
            best_global.print_tree()
            print(func.fitness(best_global))
            # if cand_list[best_idx].num_of_nodes() == 1:
                # set_trace()
        
        # opt_cand = cand_list[best_idx]
        return (best_global, best_list, avg_list)

        
def main():
    # code_list = [
        # '+',
        # '-',
        # '*',
        # '/',
        # 'sqrt_',
        # '^',
        # 'sin_',
        # 'cos_', 
        # 3,
        # 'x',
        # 2,
        # 1,
        # 8,
    # ]
    
    func_list = [
        '+',
        '-',
        '*',
        '/',
        # 'sqrt',
        # '^',
        # 'sin',
        # 'cos', 
    ]
    term_list = [
        0,
        3,
        'x',
        'y',
        'z',
        2,
        1,
        8,
        4,
        5,
        6,
        7,
        8,
        9,
    ]
    max_depth = 3
    n_cand = 100
    method = 'grow'
    p_crossover = 0.9
    p_subtree_mutate = 0.1
    p_gauss_mutate = 0.5
    n_gen = 1000
    file_name = 'data.txt'
    
    tree = Node('+')
    tree.add_left(Node('*'))
    tree.add_right(Node('+'))
    tree.left.add_left(Node(3))
    tree.left.add_right(Node('^'))
    tree.left.right.add_left(Node('x'))
    tree.left.right.add_right(Node(2))
    tree.right.add_left(Node('z'))
    tree.right.add_right(Node('*'))
    tree.right.right.add_left(Node(8))
    tree.right.right.add_right(Node('y'))
    
    # tree.update_children()

 
    # set_trace()
    # tree.print_tree()
    

    # method = 'full'
    # rand_tree = init_tree(func_list, term_list, max_depth, method)    
    # rand_tree.print_tree()
    # print(rand_tree.num_of_nodes())
    
    # node = random_node(tree)
    # tree.print_tree()
    # set_trace()
    # print(node.data)
    
    gp_optimizer = GPOptimizer(n_cand, max_depth, func_list, term_list,
        p_crossover, p_subtree_mutate, p_gauss_mutate)
    gp_helper = GPHelper(func_list, term_list, p_gauss_mutate)
    
    # new_tree = gp_helper.point_mutation(tree)
    # tree.print_tree()
    # new_tree.print_tree()
    # print(func_list)
    # print(term_list)

    # new_tree = gp_helper.subtree_mutation(tree)
    # tree.print_tree()
    # new_tree.print_tree()    

    # p_leaf = 0.1
    # node = random_node_prob(tree, p_leaf)
    # tree.print_tree()
    # print(node.data)
    # set_trace()
    
    # max_depth = 3
    # tree_2 = init_tree(func_list, term_list, max_depth, 'grow')
    # print('init end')
    # tree.print_tree()
    
    # tree_2.print_tree()      
    # (new_tree, new_tree_2) = gp_helper.subtree_crossover(tree, tree_2)
    # new_tree.print_tree()
    # new_tree_2.print_tree()

    # var_dict = {
        # 'x': 26,
        # 'y': 35,
        # 'z': 1,
    # }
    # tree.print_tree()
    # result = expr_eval(func_list, term_list, tree, var_dict)    
    # print(result)
    
    # fitness_evaluator = FitnessEvaluator(gp_helper, file_name)
    # f_avg = fitness_evaluator.fitness(tree)
    # print(f_avg)
    
    # fitness_list = np.array([0.7, 0.2, 0.1])
    # idx = rws(fitness_list)
    # print(fitness_list[idx])

    fitness_evaluator = FitnessEvaluator(gp_helper, file_name)
    (opt_cand, best_list, avg_list) = gp_optimizer.optimize(fitness_evaluator, n_gen)
    opt_cand.print_tree()
    set_trace()
    plt.plot(best_list)
    plt.title('Best')
    plt.show()
    set_trace()
    plt.plot(avg_list)
    plt.title('Avg')
    plt.show()

if __name__ == '__main__':
    main()
    