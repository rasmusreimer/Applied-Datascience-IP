"""
Takes the two trained NN Classifiers,NN_final_model_NN, NN_final_model_MI and the XGB classifier XGB_final and runs inference on the test set located in the data folder, AppML_InitialProject_test_classification.
Saves the predicted probabilities to the corresponding index from the test set in a csv file, as well as a list of used input features for the model.
saves the three seperate solutions to files in the format of:
Solution n: Classification_RasmusRimer_ModelName.csv, Classification_RasmusReimer_ModelName_VariableList.csv 
saves to the path of the submission folder Electron_Project/Submission
"""

