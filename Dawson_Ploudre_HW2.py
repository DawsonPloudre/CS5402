import numpy as np
from collections import Counter


class DecisionTree():
    def __init__(self):
        pass
        
    def _entropy(self, probabilities: list) -> float:
        """
        Calculate Entropy H(X) for a given distribution P(X).
        """
        return sum([-p * np.log2(p) for p in probabilities if p>0])
    
    def _class_probabilities(self, labels: list) -> list:
        """
        Given a list of labels, return the probability for each class P(Y). 
        """
        total_count = len(labels)
        return [label_count / total_count for label_count in Counter(labels).values()]

    def _data_entropy(self, labels: list) -> float:
        """
        Calculate the Entropy H(Y) for a given list of labels.
        """
        return self._entropy(self._class_probabilities(labels))
    
   
    def _split_data(self, data: np.ndarray, feature_idx: int, feature_val: float) -> tuple: 
        """
        Split data into two sub-groups [group1, goup2] based on attribute [feature_idx] and value [feature_val]
        if  sample[feature_idx] < feature_val:
            group1 <- sample
        else:
            group2 <- sample
        """
        mask_below_threshold = data[:, feature_idx] < feature_val
        group1 = data[mask_below_threshold]
        group2 = data[~mask_below_threshold]
        return group1, group2


    def _partition_entropy(self, g1_labels: list, g2_labels:list) -> float:
        """
        Calculate the entropy for current partition. H(Y|X=feature_val)
         H(Y|X=feature_val) = weight_1 * H(group1) + weight_2 * H(group2) 
        """
        total_count = len(g1_labels) + len(g2_labels)
        return self._data_entropy(g1_labels)*(len(g1_labels)/total_count) + self._data_entropy(g2_labels)*(len(g2_labels)/total_count)


    def find_best_cutpoint(self, data: np.ndarray, attribute_list: list) -> tuple:
        """
        Find the best cutpoint for a given attribute.
        For a continuous attribute in [attribute_list]:
            1. Sort the feature values from lowest to highest;
            2. Iterate through the successive midpoints of the sorted feature values;
            3. Track the information gain in step-2 
        return  [feature_idx], [midpoint] that yields the highest [information gain].

        Parameters:
            data: [np.ndarray], a collection of samples.
            attribute_list: [list],  at the current tree node, the list of attributes to consider. For example, [0, 2] means we should consider the first and the third attributes.
        Return:
            feature_idx:  The index of the attribute to use at the current tree node.
            cutpoit:      the cutpoint for splitting the data.
            information_gain:  the information gain at this node.
        """
        DataLabels = data[:, -1]
        
        #This should give us the entropy of whole set
        initial_entropy = self._data_entropy(DataLabels)
        
        #Just Initializing
        best_feature_idx = None
        best_cutpoint = None
        best_info_gain = -float('inf')

        for feature_idx in attribute_list:
            feature_values = sorted(np.unique(data[:, feature_idx]))
            
            #IF only 1 unique val, we shouldn't check it
            if len(feature_values) <= 1:
                continue
            
            # Calculating midpoints
            midpoints = [(feature_values[i] + feature_values[i+1]) / 2 for i in range(len(feature_values)-1)]
            
            for midpoint in midpoints:
                group1, group2 = self._split_data(data, feature_idx, midpoint)
                
                #Continue if either group is empty
                if len(group1) == 0 or len(group2) == 0:
                    continue

                g1_labels = group1[:, -1]
                g2_labels = group2[:, -1]
                
                #Calculate the entropy for this partition
                cond_entropy = self._partition_entropy(g1_labels, g2_labels)
                
                #Should give us the info gain
                info_gain = initial_entropy - cond_entropy
                
                #IF the new info gain is higher, we can make it the best
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature_idx = feature_idx
                    best_cutpoint = midpoint
        
        return best_feature_idx, best_cutpoint, best_info_gain
     
    
def main():
    train_data = np.load('./train_data.npy', allow_pickle=True)  # [112, 5], each sample contains 4 features with the last column as label.
   
    my_dt = DecisionTree()
    feature_idx, feature_val, information_gain = my_dt.find_best_cutpoint(train_data, [0, 1, 2, 3])
    print("The {0}th feature should be used at the tree root with the spliting value: {1}. The infomation gain by this particition is {2}".format(feature_idx+1, feature_val, information_gain))



if __name__=="__main__":
        main()