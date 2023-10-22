import numpy as np
from numpy.random import default_rng
from treelib import Node, Tree
#function to read data

def get_data(path):
    """ Read in the dataset from the specified filepath

    Args:
        path (str): The filepath to the dataset file

    Returns:
        tuple: returns a tuple of (x, y, classes), each being a numpy array. 
                - x is a numpy array with shape (N, K), 
                    where N is the number of instances
                    K is the number of features/attributes
                - y is a numpy array with shape (N, ), and should be integers from 1 to 4
                    representing the room number
    """
    x=[]
    y=[]
    
    for line in open(path):
        if line.strip() != '':
            row = line.strip().split()
            #maybe something to make sure data is 8 elements long DONE
            assert len(row)==8 , "Error : Input data does not have 8 elements"
            x_row = list(map(float,row[:-1]))
            x.append(x_row)
            y.append(float(row[-1]))
    x=np.array(x)
    y=np.array(y)
    return (x,y)

def split_dataset(x, y, test_proportion, random_generator=default_rng()):
    """ Split dataset into training and test sets, according to the given 
        test set proportion.
    
    Args:
        x (np.ndarray): Instances, numpy array with shape (N,K)
        y (np.ndarray): Class labels, numpy array with shape (N,)
        test_proprotion (float): the desired proportion of test examples 
                                 (0.0-1.0)
        random_generator (np.random.Generator): A random generator

    Returns:
        tuple: returns a tuple of (x_train, x_test, y_train, y_test) 
               - x_train (np.ndarray): Training instances shape (N_train, K)
               - x_test (np.ndarray): Test instances shape (N_test, K)
               - y_train (np.ndarray): Training labels, shape (N_train, )
               - y_test (np.ndarray): Test labels, shape (N_train, )
    """

    shuffled_indices = random_generator.permutation(len(x))
    n_test = round(len(x) * test_proportion)
    n_train = len(x) - n_test
    x_train = x[shuffled_indices[:n_train]]
    y_train = y[shuffled_indices[:n_train]]
    x_test = x[shuffled_indices[n_train:]]
    y_test = y[shuffled_indices[n_train:]]
    return (x_train, x_test, y_train, y_test)

# change data to x,y
def calc_entropy(x,y):
    """ Calculates the entropy of the data set given
    
    Args:
        x (np.ndarray): Instances, numpy array with shape (N,K)
        y (np.ndarray): Class labels, numpy array with shape (N,)

    Returns:
        float:
            Entropy of the data set
    """
    num_samples=len(x)
    entropy_array=[] 
    distinct_rooms, counts = np.unique(y, return_counts=True)
    entropy_array=[weighted_info_per_symbol(count,num_samples) for count in counts]
    return sum(entropy_array)

def weighted_info_per_symbol(numerator,denominator):
    return (-numerator/denominator)*np.log2(numerator/denominator)

def find_split(x, y):
    """ finds optimal split in dataset and returns the left, right datasets along with splitting attribute and value respectively

    Args:
        x (np.ndarray): Instances, numpy array with shape (N,K)
        y (np.ndarray): Class labels, numpy array with shape (N,)

    Returns:
        split (dictionary): contains l_dataset_x, l_dataset_y, r_dataset_x, r_dataset_y, value, attribute
        where
        l_dataset_x (np.array): Instances, numpy array with shape (M, K) 
        l_dataset_y (np.array): Class labels, numpy array with shape (M, )
        r_dataset_x (np.array): Instances, numpy array with shape (N-M, K)
        r_dataset_x (np.array): Class labels, numpy array with shape (N-M, )
        value (int): splitting value of continuos 
        attribute (int):
    """

    """"
    ___________________________________
    | x1 | x2 | x3 | x4 | x5 | x6 | x7 |
    ___________________________________
    | . | .  | .  | .  |  . | .  |  . |
    ___________________________________ 
    
    Loop through each column of array x. Every column is a different attribute. Should loop 7 times for every router
        Sort rows of column
            For each sorted column, loop through every row starting from 1 to n-1
                Compute information gain for current data split
                    If greater than current information gain
                        Update index, attribute and max information gain variables
                    Otherwise
                        Continue 
    Return (attribute, index/value)
    
    
    1) calc entropy of x,y
    2) repeat for x= 1...7
    3) extract col x
    4) sort col x
    5) repeat for i = 1...1999
    6) calc entropy for x[col_x<col_x[i]] , y [col_x<col_x[i]]
    7) calc IG if its greater than current IG update attribute (x) and value col_x[i]
    
    
    """
    
    max_IG = 0
    entropy=calc_entropy(x,y)
    attribute = 0 
    value = 0 
    for router in range(np.shape(x)[1]):
        col_x = x[:, router]
        
        s_col_x = x[x[:, router].argsort()][:,router]
        # col_x is a sorted column in x
        
        for row in range(1,np.shape(x)[0]-1):
            remainder_left = calc_entropy(x[col_x<s_col_x[row]],y[col_x<s_col_x[row]]) * (row/len(y))
            remiander_right = calc_entropy(x[col_x>=s_col_x[row]],y[col_x>=s_col_x[row]]) * (1-(row/len(y)))
            tmp_IG = entropy - (remainder_left + remiander_right)
            if tmp_IG > max_IG:
                max_IG = tmp_IG
                attribute = router
                value = col_x[row]
                
                
    split={"l_dataset_x":x[x[:,attribute]<value], "l_dataset_y":y[x[:,attribute]<value],"r_dataset_x":x[x[:,attribute]>=value],"r_dataset_y":y[x[:,attribute]>=value],"value": value,"attribute":attribute}
    # print(split)
    return split

def decision_tree_learning(x,y,depth=0):
    """ creates a decision tree recursively
    
    Args:
        x (np.ndarray): Instances, numpy array with shape (N,K)
        y (np.ndarray): Class labels, numpy array with shape (N,)
        depth (int): depth of the decision tree

    Returns:
        root (dictionary):
            contains left_node,right_node,split_feature,split_value
        or 
        root (int)
            contains final decision
        
        depth (int): max depth of the decision tree
    """
    node={}
    if len(np.unique(y))==1:
        # building the leaft
        return ({"l_branch":None , "r_branch":None , "split_value":None , "split_attribute":None, "Final_Descision": np.unique(y)[0],"leaf":True}, depth)
    else:
        split = find_split(x, y) # return left and right datasets and attribute and attribute val for the node
        if np.shape(split["l_dataset_x"])[0] and np.shape(split["l_dataset_y"])[0] and np.shape(split["r_dataset_x"])[0] and np.shape(split["r_dataset_y"])[0]:
            node["l_branch"], l_depth = decision_tree_learning(split["l_dataset_x"], split["l_dataset_y"], depth+1)
            node["r_branch"], r_depth = decision_tree_learning(split["r_dataset_x"], split["r_dataset_y"], depth+1)
            node["split_value"], node["split_attribute"] = split["value"], split["attribute"]
            node["leaf"] = False
            return(node, max(l_depth,r_depth))
        print(split)
        return ({"l_branch":None , "r_branch":None , "split_value":None , "split_attribute":None, "Final_Descision": None,"leaf":True}, depth) 
# use relative file path so its the same for everyone
dx,dy=get_data("./data/wifi_db/clean_dataset.txt")         
            
root,maxD = decision_tree_learning(dx,dy)
      
def print_tree(root, level=0, prefix="Root: "):
        if root['leaf']:
            ret = "\t" * level + prefix +'Final Decision: ' + str(root["Final_Descision"]) + "\n"
        else: ret = "\t" * level + prefix + " split val : " + str(root['split_value']) + " split attribute : " + str(root['split_attribute'])  + "\n"
        if root['l_branch']:
            ret += print_tree(root['l_branch'], level + 1, "L--- ")
        if root['r_branch']:
            ret += print_tree(root['r_branch'], level + 1, "R--- ")
        return ret
        
tree= print_tree(root)
print(tree)


