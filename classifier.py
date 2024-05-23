# classifier.py
# Lin Li/26-dec-2021
#
# python pacman.py --p ClassifierAgent 
# -l smallGrid

import numpy as np

class Classifier:
    # Constructor. This gets run when we first invoke pacman.py
    def __init__(self, maxDepth = 20, minLeaf = 2):
        # create an instance of the classifier
        self.maxDepth = maxDepth
        self.minLeaf = minLeaf
        self.RandomForest = None
        self.listIndexes = None
        pass

    # to reset the fit of the classifier
    def reset(self):
        pass
    
    # fit the classifier to the provided data
    def fit(self, data, target):
        data = np.array(data)
        target = np.array(target)
        ## train a random forest
        # The list_indexes list is used to store the feature indexes used for each decision tree
        # in the random forest
        self.RandomForest, self.listIndexes = self.fitRandomForest(data, target, 10, 5, 17, 0.4)
        pass


    ## Uses the saved random forest and feature indices to predict a move.
    def predict(self, data, legal):
        return self.predictRandomForest(data, self.RandomForest, self.listIndexes, legal)
        
    ## Calculate the gini index per feature in the training data, 
    ## which is used to calculate the optimal split for the decision tree
    def gini(self, xTrain, yTrain): 
        giniList = np.array([])
        #iterate over all features
        for i in range(len(xTrain[0])): 
            gini0 = 0
            # iterate over unique feature values
            for j in set(xTrain.flatten()): 
                gini1 = 0
                #get index where attribute == feature value
                index = np.where(xTrain[:,i] == j)[0] 
                #calculate proportion of unique feature value in dataset
                prop = len(index)/len(xTrain) 
                #only calculate when index not empty
                if index.size > 0:  
                    #iterate over unqiue classes and calculate gini index 
                    for k in set(yTrain): 
                        gini1 += (len(np.where(yTrain[index] == k)[0])/len(index))**2
                # sum gini index weighted by proportion
                gini0 += prop*(1 - gini1) 
            giniList = np.append(giniList, gini0)
        return giniList
    
    ## Uses the decision tree to predict Pacman's next action
    def predictDecisionTree(self, xTree, features, legal):
        # turn the provided feature vector into numpy array
        features = np.array(features)
        # use the classifier to predict a move based on the feature vector
        node = xTree
        lastNode = node

        counter = 0

        while type(node) == dict:
            counter += 1
            split = node['split']

            #count iterations and stop if exceeded depth by factor of 1.5
            if counter >= 30:
                return np.random.randint(4)

            # First checks if after next split there will be a class prediction. 
            # If that is the case, then check wether the class that would be selected 
            # corresponds to a non-legal action; 
            # if that is the case, select the other one or move up, if both are illegal.

            # Check if selecting left will lead to class prediction, and whether moved up 
            # in previous iteration, if the case don't let move up again.
            if features[split] == 0 and type(node['left 0']) != dict: 
                # Check if left will lead to illegal action and the other alternative is 
                # also a class prediction
                if self.convertNumberToMove(node['left 0']) not in legal and type(node['right 1']) != dict: 
                    # If both alternatives are class predictions, and not in legal select
                    # the "sibling node" to the current one.
                    if self.convertNumberToMove(node['right 1']) not in legal and lastNode['right 1'] == node: 
                        node = lastNode['left 0']
                    elif self.convertNumberToMove(node['right 1']) not in legal and lastNode['left 0'] == node:
                        node = lastNode['right 1']
                # If selecting left is class prediction and either right is not class 
                # prediction or will result in legal move, select right.
                elif self.convertNumberToMove(node['left 0']) not in legal: 
                    lastNode = node
                    node = node['right 1']

                else:
                    #if left will lead to legal move select that node
                    node = node['left 0'] 

            # check if selecting left will lead to class prediciton
            elif features[split] == 1 and type(node['right 1']) != dict:  
                if self.convertNumberToMove(node['right 1']) not in legal and type(node['left 0']) != dict:
                    if self.convertNumberToMove(node['left 0']) not in legal and lastNode['right 1'] == node:
                        node = lastNode['left 0']
                    elif self.convertNumberToMove(node['left 0']) not in legal and lastNode['left 0'] == node:
                        node = lastNode['right 1']

                elif self.convertNumberToMove(node['right 1']) not in legal:
                    lastNode = node
                    node = node['left 0']
                else:
                    node = node['right 1']

            elif features[split] == 0:
                lastNode = node
                node = node['left 0']


            elif features[split] == 1:
                lastNode = node
                node = node['right 1']

        return node

    ##  Recursively grows a decision tree given the training data and labels.
    def growDecisionTree(self, xTrain, yTrain, depth, parentLeaf, splitMask):
        # calculate unique classes and their frequency
        classes, counts = np.unique(yTrain, return_counts = True)  
        # calculate probability of each class
        counts = counts / len(yTrain)  

        # return parent leaf value if no more examples left
        if classes.size == 0:
            return parentLeaf
        # choose leaf value based on frequency
        leafValue = np.random.choice(classes, 1, True, counts)[0]  

        # return leaf-value if max depth is reached, there is only 1 class left, or 
        # all attributes have been used
        if depth > self.maxDepth or classes.size == 1 or not splitMask.any():
            return leafValue

        # return leaf-value if minLeaf size is reached
        if len(xTrain) <= self.minLeaf:
            return leafValue
        splitMask = np.copy(splitMask)

        # compute gini index for each feature and select the lowest
        giniList = self.gini(xTrain, yTrain)
        # exclude already chosen splits
        giniList[np.where(splitMask == False)[0]] = 1  
        split = np.argmin(giniList)
        # keep track of splits
        splitMask[split] = False  

        # split the data into left node == 0 and right node == 1
        leftIndex = np.where(xTrain[:, split] == 0)
        rightIndex = np.where(xTrain[:, split] == 1)

        # apply growDecisionTree to left node and right node
        xLeft = self.growDecisionTree(xTrain[leftIndex], yTrain[leftIndex], depth + 1, leafValue, splitMask)
        splitMask = np.copy(splitMask)
        xRight = self.growDecisionTree(xTrain[rightIndex], yTrain[rightIndex], depth + 1, leafValue, splitMask)
        splitMask = np.copy(splitMask)

        node = {
            "split": split,
            "left 0": xLeft,
            "right 1": xRight
        }
        return node

    def fitRandomForest(self, x, y, nTrees, minFeatures, maxFeatures, nSampleSize):
        randomForestTrees = []
        listIndexes = []
        for i in range(nTrees):
            # select a subsample of training data randomly with replacement
            stoch = np.array(range(len(x)))
            np.random.shuffle(stoch)
            stochShuffled = round(len(y) * nSampleSize)
            xTrainFold = x[stoch[:stochShuffled]]
            yTrainFold = y[stoch[:stochShuffled]]

            # select features randomly with replacement
            stoch = np.array(range(len(x[0])))
            np.random.shuffle(stoch)

            # Choose features randomly between minimum and maximum feature limits
            nFeatures = np.random.randint(minFeatures, maxFeatures)
            listIndexes.append(list(stoch[:nFeatures]))
            xTrainFold = xTrainFold[:, stoch[:nFeatures]]

            # Creates decision tree
            foldTree = self.growDecisionTree(xTrainFold, yTrainFold, 0, 0, np.ones(xTrainFold.shape[1], dtype=bool))

            # Adds decision tree to the list (random forest)
            randomForestTrees.append(foldTree)

        return randomForestTrees, listIndexes

    ## uses the saved random forest model and feature indices to predict 
    ## the next move of Pacman
    def predictRandomForest(self, x, randomForestTrees, listIndexes, legal):
        yPredictionList = []
        for i in range(len(randomForestTrees)):           
            newX = []
            # Get the features from x which are used in creating the current decision tree 
            # in random forest 
            for j in listIndexes[i]:
                newX.append(x[j])

            # Make prediction based on the specific decision tree
            yPredictionList.append(self.predictDecisionTree(randomForestTrees[i], newX, legal))
            
        # return the direction with the maximum count
        return max(set(yPredictionList), key=yPredictionList.count)

    def convertNumberToMove(self, number):
        # return string values for direction based on numeric input for direction
        if number == 0:
            return 'North'
        elif number == 1:
            return 'East'
        elif number == 2:
            return 'South'
        elif number == 3:
            return 'West'
    
    
        
