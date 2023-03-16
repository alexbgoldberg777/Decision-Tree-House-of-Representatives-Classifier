# Decision-Tree-House-of-Representatives-Classifier
This is a decision tree algorithm used to classify a House of Representatives Member as Democrat or Republican based on their votes for key issues. Decision-Tree.py includes the training algorithm. Running this file will create 50 training and test set splits of the dataset, train the algorithm on the training set, and compute the accuracy of using the trained tree on the test set. The accuracies of these 50 trials are shown on a plot in the Plots folder.

The dataset can be found at: https://archive.ics.uci.edu/ml/datasets/congressional+voting+records
Within the included version of the dataset, the column value key is as follows: a 0 means the representative of that row did not vote on that issue, a 1 means a nay vote on that issue, and a 2 means a yea vote. In the label column, a 0 denotes a Democrat while a 1 denotes a Republican.
