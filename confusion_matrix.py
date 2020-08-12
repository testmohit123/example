import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import numpy as np
# Set random seed
seed = 42

################################
########## DATA PREP ###########
################################

# Load in the data
df = pd.read_csv("wine_quality.csv")

# Split into train and test sections
X = df.drop('quality',axis = 1)
y = df.quality
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

#################################
########## MODELLING ############
#################################

# Fit a model on the train section
regr = RandomForestClassifier(max_depth=5, random_state=seed)
regr.fit(X_train, y_train)

# Report training set score
predictions = regr.predict(X_test)
train_score = accuracy_score(predictions,y_test) * 100

# Write scores to a file
with open("metrics.txt", 'w') as outfile:
        outfile.write("Accuracy of the Model is: %2.1f%%\n" % train_score)
        


##########################################
############ PLOT CONFUSION MATRIX #############
##########################################

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(regr,X_test,y_test)
plt.show()
plt.tight_layout()
plt.savefig("confusion_matrix.png",dpi=120)
plt.close()
