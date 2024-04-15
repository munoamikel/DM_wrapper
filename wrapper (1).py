#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 17:19:20 2023

@author: 10101317
"""

from weka.core.converters import Loader
from weka.core.classes import Random, main
from weka.classifiers import Classifier, Evaluation
import weka.core.jvm as jvm
from weka.core.dataset import Attribute, Instance, Instances
from weka.filters import Filter
from weka.core.classes import from_commandline
import matplotlib.pyplot as plt



jvm.start(system_cp=True, packages=True, max_heap_size="512m", system_info=True)


# Initialize ARFF loader
arff_loader = Loader(classname="weka.core.converters.ArffLoader")

# Data directory and ARFF file names
data_directory = "/users/1010317/Desktop/BILDU/3.maila/DM/irteera/"
arff = "originalEdge.arff"

# Load ARFF file and set the class index
data = arff_loader.load_file(data_directory + arff, class_index="last")
data.class_is_last()

# Bagging(10) classifier with 1-NN for arff1
cmdline = 'weka.classifiers.meta.Bagging -P 100 -S 1 -num-slots 1 -I 10 -W weka.classifiers.lazy.IBk -- -K 1 -W 0 -A "weka.core.neighboursearch.LinearNNSearch -A weka.core.EuclideanDistance"'
classifier = from_commandline(cmdline, classname="weka.classifiers.Classifier")

#Bagging(15) classifier with 1-NN for arff2
cmdline = 'weka.classifiers.meta.Bagging -P 100 -S 1 -num-slots 1 -I 15 -W weka.classifiers.lazy.IBk -- -K 1 -W 0 -A "weka.core.neighboursearch.LinearNNSearch -A weka.core.EuclideanDistance"'
classifier2 = from_commandline(cmdline, classname="weka.classifiers.Classifier")



# Lists and variables for attribute selection 
selected_attribute_indices = set()
best_attribute_index = 1
attr_str = ""


current_accuracy = 0
previous_accuracy = 0


# Lists to store accuracy values for plotting
accuracy_history = []


############################################################################################################################################################################
while (best_attribute_index != -1): # While there are still attributes to select
    best_attribute_index = -1
    for index, current_attribute in enumerate(data.attributes()):
        attribute_selection_string = ""

        
        if index + 1 == (data.class_index + 1):
            break

        # Choose an attribute not used before
        if not (index + 1 in selected_attribute_indices):
            for selected_index in selected_attribute_indices:
                attribute_selection_string = attribute_selection_string + str(selected_index) + ","
            
            if len(selected_attribute_indices) == 0:
                attribute_selection_string = str(index + 1) + "," + str(data.class_index + 1)
            else:
                attribute_selection_string = attribute_selection_string + str(index + 1) + "," + str(data.class_index + 1)

            # Calculate and test classifier with selected attributes
            remove_filter = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-V","-R", attribute_selection_string])
            remove_filter.inputformat(data)

            filtered_data = remove_filter.filter(data)
            filtered_data.class_is_last()


            classifier.build_classifier(filtered_data)
            evaluation = Evaluation(filtered_data)
            evaluation.test_model(classifier, filtered_data)
            current_accuracy = evaluation.percent_correct

            if (current_accuracy > previous_accuracy):
                previous_accuracy = current_accuracy
                best_attribute = current_attribute
                best_attribute_index = index

        

    if (best_attribute_index != -1):
        selected_attribute_indices.add(best_attribute_index + 1)
        accuracy_history.append(current_accuracy)


history = []
for selected_index in selected_attribute_indices:
    history.append(data.attribute(selected_index - 1))



############################################################################################################################################################################


#Create output file and store the selected attributes
output_file = open("selected_attributes.txt", "w")
output_file.write("Selected attributes:\n")
for attribute in history:
    output_file.write(attribute.name + "\n")
output_file.write("\n\nAccuracy history:\n")
for index, accuracy in enumerate(accuracy_history):
    output_file.write("Number of selected attributes: " + str(index + 1) + ", accuracy: " + str(accuracy) + "\n")
output_file.close()






# Plotting the accuracy history
plt.plot(range(1, len(accuracy_history) + 1), accuracy_history, marker='o')
plt.title('Accuracy During Attribute Selection')
plt.xlabel('Number of Selected Attributes')
plt.ylabel('Accuracy (%)')
plt.show()

#save the plot
plt.savefig('accuracy_history.png')




jvm.stop()