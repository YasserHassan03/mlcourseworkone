import numpy as np
from numpy.random import default_rng
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
        if line.strip() !='':
            row = line.strip().split()
            #maybe something to make sure data is 8 elements long DONE
            assert len(row)==8 , "Error : Input data does not have 8 elements"
            x_row = list(map(float,row[:-1]))
            x.append(x_row)
            y.append(float(row[-1]))
    x=np.array(x)
    y=np.array(y)
    return (x,y)


# use relative file path so its the same for everyone
dx,dy=get_data("./CW1 60012/wifi_db/clean_dataset.txt")

#function to split data set
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

# TODO Finish this function

def find_split(data):
    max_IG=[]
    #sort data by each attribute val
    for router in range(7):
        sorted_router=data[data[:, router].argsort()]
        #need to iterate through sorted column in array to find optimum split for each attribute (store it in smthn)
    #choose best split out of all attributes
    print(sorted_router)
    
# print(findSplit(data))



#def drawTree():
