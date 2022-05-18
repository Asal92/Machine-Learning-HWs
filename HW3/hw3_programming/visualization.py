from matplotlib import pyplot as plt
import numpy as np

# recommended color for different digits
color_mapping = {0:'red',1:'green',2:'blue',3:'yellow',4:'magenta',5:'orangered',
                6:'cyan',7:'purple',8:'gold',9:'pink'}


def plot2d(data,label,split='train'):
    # 2d scatter plot of the hidden features

    hidden_1 = data[:,0] # first hidden layer
    hidden_2 = data[:,1] # second hidden layer

    fig, ax = plt.subplots()
    for value in np.unique(label):
        ind = np.where(label == value)
        ax.scatter(hidden_1[ind], hidden_2[ind], c = color_mapping[value], label = value, s = 100)
        
    ax.legend()
    ax.set_xlabel("Hidden Value 1")
    ax.set_ylabel("Hidden Value 2")
    
    fig.savefig(split+' Plot2d.png')
    plt.title(split + " dataset 2d plot")
    plt.show()


def plot3d(data,label,split='train'):
    # 3d scatter plot of the hidden features
    hidden_1 = data[:,0] # first hidden layer
    hidden_2 = data[:,1] # second hidden layer
    hidden_3 = data[:,2] # Third hidden layer

    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    for value in np.unique(label):
        ind = np.where(label == value)
        ax.scatter(hidden_1[ind], hidden_2[ind], hidden_3[ind], c = color_mapping[value], label = value)
    ax.legend()
    ax.set_xlabel("Hidden Value 1")
    ax.set_ylabel("Hidden Value 2")
    ax.set_zlabel("Hidden Value 3")
    
    fig.savefig(split+' Plot3d.png')
    plt.title(split + " dataset 3d plot")
    plt.show()
