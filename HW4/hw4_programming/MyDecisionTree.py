import numpy as np

class Tree_node:
    """
    Data structure for nodes in the decision-tree
    """
    def __init__(self,):
        self.is_leaf = False # whether or not the current node is a leaf node
        self.feature = None # index of the selected feature (for non-leaf node)
        self.label = -1 # class label (for leaf node)
        self.left_child = None # left child node
        self.right_child = None # right child node

class Decision_tree:
    """
    Decision tree with binary features
    """
    def __init__(self,min_entropy):
        self.min_entropy = min_entropy
        self.root = None

    def fit(self,train_x,train_y):
        # construct the decision-tree with recursion
        self.root = self.generate_tree(train_x,train_y)

    def predict(self,test_x):
        # iterate through all samples
        prediction = np.zeros([len(test_x),]).astype('int') # placeholder
        
        for i in range(len(test_x)):
            # traverse the decision-tree based on the features of the current sample
            tree = self.root
            while tree.is_leaf == False:
                if test_x[i,tree.feature] == 0:
                    tree = tree.left_child
                else:
                    tree = tree.right_child
            prediction[i] = tree.label

        return prediction

    def generate_tree(self,data,label):
        # initialize the current tree node
        cur_node = Tree_node()

        # compute the node entropy
        node_entropy = self.compute_node_entropy(label)

        # determine if the current node is a leaf node
        if node_entropy < self.min_entropy:
            # determine the class label for leaf node
            cur_node.is_leaf = True
            
            n_idx = []
            for i in range(10):
                n_idx.append(sum(1 for j in range(len(label)) if label[j]== i))
                
            cur_node.label = n_idx.index(max(n_idx))
            return cur_node

        # select the feature that will best split the current non-leaf node
        selected_feature = self.select_feature(data,label)
        cur_node.feature = selected_feature

        # split the data based on the selected feature and start the next level of recursion
        left_x,right_x = [],[]
        left_y,right_y = [],[]
        for n in range(len(data)):
            if data[n,selected_feature] == 0:
                left_x.append(data[n])
                left_y.append(label[n])
            else:
                right_x.append(data[n])
                right_y.append(label[n])

        left_x = np.array(left_x)
        right_x = np.array(right_x)
        cur_node.left_child = self.generate_tree(left_x,left_y)
        cur_node.right_child = self.generate_tree(right_x,right_y)


        return cur_node

    def select_feature(self,data,label):
        # iterate through all features and compute their corresponding entropy
        best_feat = 0
        feature_entropy = []

        for i in range(len(data[0])):# 64 features

            # compute the entropy of splitting based on the selected features
            left,right= [],[]
            left_y, right_y = [],[]
            for n in range(len(data)):
                if data[n,i] == 0:
                    left.append(label[n])
                else:
                    right.append(label[n])

            #left,right = np.array(left), np.array(right)

            feature_entropy.append(self.compute_split_entropy(left, right))
            

        # select the feature with minimum entropy
        best_feat = feature_entropy.index(min(feature_entropy))

        return best_feat

    def compute_split_entropy(self,left_y,right_y):
        # compute the entropy of a potential split, left_y and right_y are labels for the two branches
        split_entropy = 0 # placeholder
        
        if len(left_y) > 0 :
            left_entropy = self.compute_node_entropy(left_y)
        else:
            left_entropy = 0
        
        if len(right_y) > 0 :
            right_entropy = self.compute_node_entropy(right_y)
        else:
            right_entropy = 0
        
        split_entropy = (len(left_y)/(len(left_y)+len(right_y))) * left_entropy
        split_entropy += (len(right_y)/(len(left_y)+len(right_y))) * right_entropy
        

        return split_entropy

    def compute_node_entropy(self,label):
        # compute the entropy of a tree node (add 1e-15 inside the log2 when computing the entropy to prevent numerical issue)
        node_entropy = 0 # placeholder
        #(sum -p*log2(p+1e-15)
        
        N = len(label)
        n_idx,p = [],[]
        for i in range(10):
            n_idx.append(sum(1 for j in range(len(label)) if label[j]== i))
            p.append(n_idx[i]/N)
            node_entropy += -(p[i]*np.log2(p[i] + 1e-15))

        return node_entropy
