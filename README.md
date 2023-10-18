# Flower Classification and Recommendation

## Details
* In this project, we will tackle the problem of classifying flower types and recommending similar flower images based on the provided image.
* For flower classification, we used supervised learning approach, specifically Convolutional Neural Network (CNN). We trained the model using AlexNet and ResNet50, then compare the model performance for evaluation.
* For flower recommendation, we used unsupervised learning approach (clustering). We trained the KNN model and DBSCAN model, then compare the model performance for evaluation.
* There is a notebook file and a python file for each task.
* Further details are available in the project report.

## How To Use
### Task 1: Flower Classification
1. Install the following packages before running the jupyter notebook file and the python script: pandas, numpy, matplotlib, seaborn, sklearn, tensorflow, PIL
2. To run the model via python script, use command prompt to navigate to the folder containing the file predict.py
3. Type the command: python predict.py -i "image_path"
4. Wait for the program to process the image, and print out the flower type of the input image.

### Task 2: Flower Recommendation
1. Install the following packages before running the jupyter notebook file and the python script: pandas, numpy, matplotlib, seaborn, sklearn, tensorflow, PIL, opencv-python, plotly
2. To run the model via python script, use command prompt to navigate to the folder containing the file recommend.py
3. Type the command: python recommend.py -i "image_path"
4. Wait for the program to process the image, and recommend 10 images from the dataset.

## References
[1] Y. Liu, F. Tang, D. Zhou, Y. Meng, and W. Dong, ‘Flower classification via convolutional
neural network’, in 2016 IEEE International Conference on Functional-Structural Plant
Growth Modeling, Simulation, Visualization and Applications (FSPMA), 2016, pp. 110–116.<br/>
[2] Z. S. Younus et al., “Content-based image retrieval using PSO and k-means clustering
algorithm,” Arabian journal of geosciences, vol. 8, no. 8, pp. 6211–6224, 2015, doi:
10.1007/s12517-014-1584-7.<br/>
[3] S. Saxena, “Introduction to the architecture of Alexnet,” Analytics Vidhya,
https://www.analyticsvidhya.com/blog/2021/03/introduction-to-the-architecture-of-alexnet
(accessed May 13, 2023).<br/>
[4] P. Dwivedi, “Understanding and coding a ResNet in Keras,” Medium,
https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33
(accessed May 13, 2023).<br/>
[5] A. GUPTA, “PCA vs T-Sne,” Kaggle, https://www.kaggle.com/code/agsam23/pca-vs-t-sne
(accessed May 19, 2023).<br/>
[6] DeepAI, “K-means,” DeepAI,
https://deepai.org/machine-learning-glossary-and-terms/k-means (accessed May 19, 2023).<br/>
[7] A. Gupta, “Elbow method for optimal value of K in kmeans,” GeeksforGeeks,
https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/ (accessed
May 19, 2023).<br/>
[8] Ginni, “What is DBSCAN,” Tutorials Point, https://www.tutorialspoint.com/what-is-dbscan
(accessed May 19, 2023).<br/>
[9] H. Belyadi and A. Haghighat, “Silhouette coefficient,” Silhouette Coefficient - an overview |
ScienceDirect Topics,
https://www.sciencedirect.com/topics/computer-science/silhouette-coefficient (accessed May
19, 2023)<br/>
