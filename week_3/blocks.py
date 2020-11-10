import numpy as np


# In[105]:


def linear_forward(x_input, W, b):
    """Perform the mapping of the input
    # Arguments
        x_input: input of the linear function - np.array of size `(n_objects, n_in)`
        W: np.array of size `(n_in, n_out)`
        b: np.array of size `(n_out,)`
    # Output
        the output of the linear function 
        np.array of size `(n_objects, n_out)`
    """
    output = np.dot(x_input,W) + b
    
    return output


def linear_grad_W(x_input, grad_output, W, b):
    """Calculate the partial derivative of 
        the loss with respect to W parameter of the function
        dL / dW = (dL / dh) * (dh / dW)
    # Arguments
        x_input: input of a dense layer - np.array of size `(n_objects, n_in)`
        grad_output: partial derivative of the loss functions with 
            respect to the ouput of the dense layer (dL / dh)
            np.array of size `(n_objects, n_out)`
        W: np.array of size `(n_in, n_out)`
        b: np.array of size `(n_out,)`
    # Output
        the partial derivative of the loss 
        with respect to W parameter of the function
        np.array of size `(n_in, n_out)`
    """
    grad_W = np.dot(x_input.T,grad_output)
    
    return grad_W


def linear_grad_b(x_input, grad_output, W, b):
    """Calculate the partial derivative of 
        the loss with respect to b parameter of the function
        dL / db = (dL / dh) * (dh / db)
    # Arguments
        x_input: input of a dense layer - np.array of size `(n_objects, n_in)`
        grad_output: partial derivative of the loss functions with 
            respect to the ouput of the linear function (dL / dh)
            np.array of size `(n_objects, n_out)`
        W: np.array of size `(n_in, n_out)`
        b: np.array of size `(n_out,)`
    # Output
        the partial derivative of the loss 
        with respect to b parameter of the linear function
        np.array of size `(n_out,)`
    """
    grad_b = np.sum(grad_output)
    
    return grad_b


def sigmoid_forward(x_input):
    """sigmoid nonlinearity
    # Arguments
        x_input: np.array of size `(n_objects, n_in)`
    # Output
        the output of relu layer
        np.array of size `(n_objects, n_in)`
    """
    output = 1/(1+np.e**(-x_input))
    return output



# In[131]:


def sigmoid_grad_input(x_input, grad_output):
    """sigmoid nonlinearity gradient. 
        Calculate the partial derivative of the loss 
        with respect to the input of the layer
    # Arguments
        x_input: np.array of size `(n_objects, n_in)`
        grad_output: np.array of size `(n_objects, n_in)` 
            dL / df
    # Output
        the partial derivative of the loss 
        with respect to the input of the function
        np.array of size `(n_objects, n_in)` 
        dL / dh
    """
    f_h = (1/(1+np.e**(-x_input)))*(1-(1/(1+np.e**(-x_input))))
    grad_input = f_h * grad_output
    return grad_input


# In[133]:




def nll_forward(target_pred, target_true):
    """Compute the value of NLL
        for a given prediction and the ground truth
    # Arguments
        target_pred: predictions - np.array of size `(n_objects, 1)`
        target_true: ground truth - np.array of size `(n_objects, 1)`
    # Output
        the value of NLL for a given prediction and the ground truth
        scalar
    """
    output = -(1/len(target_pred))*np.sum(target_true*np.log(target_pred) + (1-target_true)*np.log(1-target_pred))  
    return output



def nll_grad_input(target_pred, target_true):
    """Compute the partial derivative of NLL
        with respect to its input
    # Arguments
        target_pred: predictions - np.array of size `(n_objects, 1)`
        target_true: ground truth - np.array of size `(n_objects, 1)`
    # Output
        the partial derivative 
        of NLL with respect to its input
        np.array of size `(n_objects, 1)`
    """
    grad_input = (1/len(target_pred))*((target_pred-target_true)/(target_pred*(1-target_pred)))  
    return grad_input


# In[126]:




def tree_gini_index(Y_left, Y_right, classes):
    """Compute the Gini Index.
    # Arguments
        Y_left: class labels of the data left set
            np.array of size `(n_objects, 1)`
        Y_right: class labels of the data right set
            np.array of size `(n_objects, 1)`
        classes: list of all class values
    # Output
        gini: scalar `float`
    """
    gini = 0
    all_inst = len(Y_left)+len(Y_right)

    gini += (len(Y_left)/all_inst)*(1-((len(Y_left[Y_left == classes[0]])/len(Y_left))**2 + 
                                       (len(Y_left[Y_left == classes[1]])/len(Y_left))**2))
    gini += (len(Y_right)/all_inst)*(1-((len(Y_right[Y_right == classes[0]])/len(Y_right))**2 + 
                                       (len(Y_right[Y_right == classes[1]])/len(Y_right))**2))
    
    return gini



def tree_split_data_left(X, Y, feature_index, split_value):
    """Split the data `X` and `Y`, at the feature indexed by `feature_index`.
    If the value is less than `split_value` then return it as part of the left group.
    
    # Arguments
        X: np.array of size `(n_objects, n_in)`
        Y: np.array of size `(n_objects, 1)`
        feature_index: index of the feature to split at 
        split_value: value to split between
    # Output
        (XY_left): np.array of size `(n_objects_left, n_in + 1)`
    """
    X_left, Y_left = None, None
    X_left = X[X[:,feature_index] < split_value]
    Y_left = Y[X[:,feature_index] < split_value]

    XY_left = np.concatenate([X_left, Y_left], axis=-1)
    return XY_left


def tree_split_data_right(X, Y, feature_index, split_value):
    """Split the data `X` and `Y`, at the feature indexed by `feature_index`.
    If the value is greater or equal than `split_value` then return it as part of the right group.
    
    # Arguments
        X: np.array of size `(n_objects, n_in)`
        Y: np.array of size `(n_objects, 1)`
        feature_index: index of the feature to split at
        split_value: value to split between
    # Output
        (XY_left): np.array of size `(n_objects_left, n_in + 1)`
    """
    X_right, Y_right = None, None
    
    X_right = X[X[:,feature_index] >= split_value]
    Y_right = Y[X[:,feature_index] >= split_value]
    
    XY_right = np.concatenate([X_right, Y_right], axis=-1)
    return XY_right




def tree_best_split(X, Y):
    class_values = list(set(Y.flatten().tolist()))
    r_index, r_value, r_score =  float("inf"),  float("inf"), float("inf")
    r_XY_left, r_XY_right = (X,Y), (X,Y)
    for feature_index in range(X.shape[1]):
        for row in X:
            XY_left = tree_split_data_left(X, Y, feature_index, row[feature_index])
            XY_right = tree_split_data_right(X, Y, feature_index, row[feature_index])
            XY_left, XY_right = (XY_left[:,:-1], XY_left[:,-1:]), (XY_right[:,:-1], XY_right[:,-1:])
            gini = tree_gini_index(XY_left[1], XY_right[1], class_values)
            if gini < r_score:
                r_index, r_value, r_score = feature_index, row[feature_index], gini
                r_XY_left, r_XY_right = XY_left, XY_right
    return {'index':r_index, 'value':r_value, 'XY_left': r_XY_left, 'XY_right':r_XY_right}



def tree_to_terminal(Y):
    """The most frequent class label, out of the data points belonging to the leaf node,
    is selected as the predicted class.
    
    # Arguments
        Y: np.array of size `(n_objects)`
        
    # Output
        label: most frequent label of `Y.dtype`
    """
    (values,counts) = np.unique(Y,return_counts=True)
    ind = np.argmax(counts)
    label = values[ind]
    
    return label


def tree_recursive_split(X, Y, node, max_depth, min_size, depth):
    XY_left, XY_right = node['XY_left'], node['XY_right']
    del(node['XY_left'])
    del(node['XY_right'])
    # check for a no split
    if XY_left[0].size <= 0 or XY_right[0].size <= 0:
        node['left_child'] = node['right_child'] = tree_to_terminal(np.concatenate((XY_left[1], XY_right[1])))
        return
    # check for max depth
    if depth >= max_depth:
        node['left_child'], node['right_child'] = tree_to_terminal(XY_left[1]), tree_to_terminal(XY_right[1])
        return
    # process left child
    if XY_left[0].shape[0] <= min_size:
        node['left_child'] = tree_to_terminal(XY_left[1])
    else:
        node['left_child'] = tree_best_split(*XY_left)
        tree_recursive_split(X, Y, node['left_child'], max_depth, min_size, depth+1)
    # process right child
    if XY_right[0].shape[0] <= min_size:
        node['right_child'] = tree_to_terminal(XY_right[1])
    else:
        node['right_child'] = tree_best_split(*XY_right)
        tree_recursive_split(X, Y, node['right_child'], max_depth, min_size, depth+1)




