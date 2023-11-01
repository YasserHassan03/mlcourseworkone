import numpy as np
import decimal as dec
from numpy.random import default_rng
from treelib import Node, Tree
import matplotlib
import matplotlib.pyplot as plt
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



def split_dataset(x, y, folds = 10,  random_generator=default_rng()):
    """ 
    shuffles the data and splits into 10 folds
    Args:
        x (np.ndarray): Instances, numpy array with shape (N,K)
        y (np.ndarray): Class labels, numpy array with shape (N,)
        folds (int): number of folds
        random_generator (np.random.Generator): random generator object
    Returns:
        tuple: returns a tuple of (x_data, y_data), each being a numpy array. 
                - x_data is a list of numpy arrays with shape (N/10, K), 
                    where N is the number of instances
                    K is the number of features/attributes
                - y_data is a list of numpy arrays with shape (N/10, ), and should be integers from 1 to 4
                    representing the room number
    
    """
    test_proportion=1/folds
    x_data=[]
    y_data=[]
    shuffled_indices = random_generator.permutation(len(x))
    length_of_fold = round(len(x) * test_proportion)
    
    x_shuffled=x[shuffled_indices]
    y_shuffled=y[shuffled_indices]

    for i in range(folds):
        x_data.append(x_shuffled[i*length_of_fold:(i+1)*length_of_fold])
        y_data.append(y_shuffled[i*length_of_fold:(i+1)*length_of_fold])
    # print(x_data, y_data)
    return (x_data,y_data)
    
def calc_entropy(y):
    """ Calculates the entropy of the data set given
    
    Args:
        x (np.ndarray): Instances, numpy array with shape (N,K)
        y (np.ndarray): Class labels, numpy array with shape (N,)

    Returns:
        float:
            Entropy of the data set
    """
    num_samples=len(y)
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
    max_IG = 0
    entropy=calc_entropy(y)
    attribute = 0 
    value = 0 
    split_index = 0
    assert np.shape(x)[1]==7
    for router in range(7):
        col_x = x[:, router]
        s_col_x = x[x[:, router].argsort()][:,router]
        s_y= y[x[:, router].argsort()]
        for row in range(s_y.shape[0]-1):
            if s_y[row] != s_y[row+1]:
                mid = (s_col_x[row] + s_col_x[row+1]) / 2
                l_ds = s_y[:row+1]
                r_ds = s_y[row+1:]    
                assert r_ds.shape[0] + l_ds.shape[0] == x.shape[0]
                remainder_left = calc_entropy(l_ds) * (l_ds.shape[0]/len(y))
                remainder_right = calc_entropy(r_ds) * (r_ds.shape[0]/len(y))
                tmp_IG = round(entropy, 10) - round(remainder_left + remainder_right, 10)
                assert tmp_IG >= 0    
                if tmp_IG > max_IG:
                    max_IG = tmp_IG
                    attribute = router
                    value = mid
                    split_index = row+1
    sorted_ds = x[np.argsort(x[:, attribute])]
    sorted_labels = y[np.argsort(x[:, attribute])]
    split={"l_dataset_x":sorted_ds[:split_index], "l_dataset_y":sorted_labels[:split_index],"r_dataset_x":sorted_ds[split_index:],"r_dataset_y":sorted_labels[split_index:],"value": value,"attribute":attribute}
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
    if np.unique(y).shape[0] == 1:
        return ({"l_branch":None , "r_branch":None , "split_value":None , "split_attribute":None, "Final_Decision": np.unique(y)[0],"leaf":True}, depth)
    else:
        split = find_split(x, y)
        ret_l_depth = depth
        ret_r_depth = depth
        node = {}
        if (split["l_dataset_x"].shape[0] > 0):
            node["l_branch"], ret_l_depth = decision_tree_learning(split["l_dataset_x"], split["l_dataset_y"], depth+1)
        else:
            node["l_branch"]=None
        if (split["r_dataset_x"].shape[0] > 0):
            node["r_branch"], ret_r_depth = decision_tree_learning(split["r_dataset_x"], split["r_dataset_y"], depth+1)
        else:
            node["r_branch"]=None
        node["split_value"], node["split_attribute"] = split["value"], split["attribute"]            
        node["leaf"] = False
        return(node, max(ret_l_depth,ret_r_depth))

      
def print_tree(root, level=0, prefix="Root: "):
        if 'leaf' in root:
            if root['leaf']:
                ret = "\t" * level + prefix +'Final Decision: ' + str(root["Final_Decision"]) + "\n"
                return ret
        ret = "\t" * level + prefix + " split val : " + str(root['split_value']) + " split attribute : " + str(root['split_attribute'])  + "\n"
        if 'l_branch'in root:
            ret += print_tree(root['l_branch'], level + 1, "L--- ")
        if 'r_branch' in root:
            ret += print_tree(root['r_branch'], level + 1, "R--- ")
        return ret
        
#function to recursively plot the decision tree
def plot_decision_tree(node, parent_pos, branch, depth=0):
    """ Plots decision tree on matlab canvas recursively
    Args:
        node (dictionary): contains left_node,right_node,split_feature,split_value
        parent_pos (tuple): contains x,y coordinates of parent node
        branch (int): -1 for left branch, 1 for right branch
    Returns:
        None
    """
    if node is None:
        return
    scale=1000/(1.9**depth)
    if depth>6:
        scale=1000/(1.8**(depth-3))
    if depth>10:
        scale=1000/(1.7**(depth-3))
    if depth>11:
        scale=1000/(1.7**(depth-5))
    x = parent_pos[0] + branch * scale
    if depth==0:
        y = parent_pos[1]
    else:
        y = parent_pos[1]-1
    if branch == -1:
        plt.plot([parent_pos[0], x], [parent_pos[1], y], 'b-', zorder=1)
    else:
        plt.plot([parent_pos[0], x], [parent_pos[1], y], 'r-', zorder=1)

    if node['leaf']:
        plt.scatter(x, y, s=120, c='green', edgecolor='black', zorder=2)
        plt.text(x, y, str(int(node['Final_Decision'])), ha='center', va='center', fontsize=10, color='white')
    else:
        plt.scatter(x, y, color='white')
        plt.text(x, y, ('x' +str(node['split_attribute'])+'<'+str(node['split_value'])), ha='center', va='center', fontsize=5, color='black',bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    if 'l_branch' in node:
        plot_decision_tree(node['l_branch'], (x, y), -1, depth + 1)
    if 'r_branch' in node:
        plot_decision_tree(node['r_branch'], (x, y), 1, depth + 1)
    #print('x' +str(node['split_attribute'])+'<'+str(node['split_value']))
    #print(x,y)
    #print(depth)


def predict_room(test_attributes,node):
    '''
    predict the room number for a given test data by traversing the decision tree
    Args:
        test_attributes (np.ndarray): Instances, numpy array with shape (N,K)
        node (dictionary): contains left_node,right_node,split_feature,split_value
    returns:
        room number (int)
    '''
    
    if node['leaf']:
        return node['Final_Decision']
    else:
        if (test_attributes[(node['split_attribute'])] < node["split_value"]):
            return predict_room(test_attributes, node['l_branch'])
        else:
            return predict_room(test_attributes, node['r_branch'])

def predict_rooms(test_data, node):
    '''
    get predictions for all the test data
    Args:
        test_data (np.ndarray): Instances, numpy array with shape (N,K)
        node (dictionary): contains left_node,right_node,split_feature,split_value
    returns:    
        np.ndarray: room number predictions for all the test data of shape (N, )
    '''
    return np.apply_along_axis(predict_room, 1, test_data, node)

#globally scoped

confusion_matrix = np.zeros((4,4))

def populate_matrix(room_preds, actual_rooms): 
    '''
    populates confusion matrix
    Args:
        room_preds (np.ndarray): room number predictions for all the test data of shape (N, )
        actual_rooms (np.ndarray): Class labels, numpy array with shape (N,)
    returns:
        None
    '''
    for i in range(len(room_preds)):
        confusion_matrix[int(actual_rooms[i]-1)][int(room_preds[i]-1)] += 1
    

def cross_validation(x, y, folds=10):
    '''
    splits the data into 10 folds and performs cross validation
    Args:
        x (np.ndarray): Instances, numpy array with shape (N,K)
        y (np.ndarray): Class labels, numpy array with shape (N,)
        folds (int): number of folds
    returns:
        None
    '''
      
    x_folds, y_folds= split_dataset(x,y)
    for i in range(folds):
        x_train = np.concatenate(x_folds[:i] + x_folds[i+1:],axis=0)
        y_train = np.concatenate(y_folds[:i] + y_folds[i+1:], axis=0)
        x_test = x_folds[i]
        y_test = y_folds[i]
        root,maxD = decision_tree_learning(x_train,y_train)
        room_preds=predict_rooms(x_test,root)
        populate_matrix(room_preds,y_test)

def accuracy():
    '''
    Calculates accuracy of model from the confusion matric
    Args:
        None
    Returns:
        Accuracy
    '''
    return (np.trace(confusion_matrix)/confusion_matrix.sum())*100

def precision_recall():
    '''
    calculates the precision and the recall of the training data for each class from the confusion matrix
    Args:
        None
    Return:
        tuple of lists : (Precision for each class, Recall for each class)
    '''
    
    true_positives=np.diag(confusion_matrix)
    precision_per_class=[]
    recall_per_class=[]
    for i in range(len(true_positives)):
        precision_per_class.append(true_positives[i]/np.sum(confusion_matrix,0)[i])
        recall_per_class.append(true_positives[i]/np.sum(confusion_matrix,1)[i])
    return (np.array(precision_per_class), np.array(recall_per_class))

def F1(precision_array,recall_array):
    '''
    Calculates the F1 score for the dataset
    Args:
        precision_array (np.ndarray): Precision for K classes, numpy array with shape (K,)
        recall_array (np.ndarray): Recall for K classes, numpy array with shape (K,)
    Returns:
            f1_per_class (np.ndarray): f1 score for K classes, numpy array with shape (K,)
    '''  
    f1_per_class = (2*precision_array*recall_array)/(precision_array+recall_array)
    return(f1_per_class)


def main():
    x_data,y_data=get_data("./data/wifi_db/noisy_dataset.txt")                     
    cross_validation(x_data,y_data)
    print(accuracy())
    precision_array,recall_array = precision_recall()
    print(precision_array,recall_array)
    print(F1(precision_array,recall_array ))
    root,maxD = decision_tree_learning(x_data,y_data) 
    print(confusion_matrix)
    #print("prediction")
    #room_preds=(predict_rooms(x_folds[3],root))
    #print("accuracy=" +str(calc_accuracy(room_preds,y_folds[3])))

    #tree=print_tree(root)
    plot_decision_tree(root, (0, 0), 0)
    plt.tight_layout()
    plt.axis('off')
    plt.show()




if __name__ == '__main__':
    main()
    
