# Data-Science-Projects
Learning data science through building projects. #100DayOfMLCode


Day 001: This might be too soon but have been working on an ANN project. Have implemented the core algorithms in @AndrewYNg 's class but this is the first time I have applied it in a real world problem. Still polishing the code.

Day002: Polished up my notebook on peak detection using a neural network. With the lack of real labeled data, I worked around the problem using simulated peak data. This made my machine learning problem a lot easier.

Day003: Taking a deep dive into principal component analysis. Never taken linear algebra before so I am pretty confused about eigenvectors. Starting with basic linear algebra and working my way to gaining a fundamental understanding of PCA. 

Day004 and 005: Progress has been slow last two days. Still going through the linear algebra essentials. @3blue1brown has a great video series walking you through the graphical intuition of linear algebra.

Day006: finally watched through the linear algebra series by @3blue1brown. Moving on to understanding single value decomposition and eigendecomposition.

Day007 and 008: Got a general understanding of the math behind PCA. Now starting a mini project to denoise an image using PCA 

Day009. Used PCA to denoise an image. It worked somewhat but it's not as good as a boxcar average. Still trying. 

Day010 and 011. PCA to denoise an image. Borrowed some ideas from this paper, which uses local pixel grouping and block matching to build a training set. Also messed around with Kmeans. Not happy with the results yet.

Day 012 and 013: Have been working on image denoising using PCA for several days. Still doesn't work well. Going to polish it up and upload to github anyways. Moving on to other topics and revisit this later.

Day 014: polished up my notebook on image denoising using PCA. Going to branch out to learn something else. Probably enroll in a data science course.

Day 015: Game plan: Need to develop the breadth and depth of knowledge at the same time. Also need to showcase skills through projects. Eventually, will develop a web app for real world data problems. Now, started a new project on stock market clustering.

Day 016: Obtained the daily stock movements of dozens of companies in last two years using pandas_datareader. Clustered with KMeans and PCA to visualize the data. Interesting to see the similarities between groups like auto and oil or McDonalds and pepsi.

Day 017: Part 1. Started reading this DL book ch 1, 2. Review on the maths behind ANN: tensor operations, gradient descent, and backprop. Always wondered how backprop was derived. It uses chain rule to calc the gradients of weights of previous layers.

Day 017: Part 2. Started exploring the pima-indian dataset. Checked for skewness and correlations using pandas.plotting.scatter_matrix. Getting rid of data that don't make sense. Following a tutorial on using keras to detect the onset of diabetes.

Day 018: Part 1. Statistics. This book seems quite useful. Reviewed on central limit theorem. Since I was exploring the pima-indians data, I also learned about bivariate (Pearson) correlation. Plotted a scattering matrix for the dataset. 

Day 019: Part 1. Read an article on chinese NLP. Would be interesting to work on as it's difficult and I speak it. Don't know too much about NLP so it might kick my butt. But will try when I'm ready.

Day 018 & 019. Part 2. Still working on diabetes onset detection using #keras. Learned about hyperparameter (learning rate, dropout, epoch, and batch size) tuning using a grid search method. And there is so much more to learn! 

Day 020 & 021. Part 2. Finishing diabetes onset detection. The dataset is too imbalanced so I had to resample the data. Used RandomizedSearchCV for auto hyperparameter tuning. Finally got acceptable precision and recall, ~80% for both classes for the test set. 

Day 021. Part 1. Important to learn from the work of others. Going through a git repo in detail: https://github.com/awbirdsall/pyvap …. Learning the practices of managing and distribution open source software. This sample project is also useful: https://github.com/pypa/sampleproject

Day 022: Part 1. Coding skill Thursday. Trying to understand how http://setup.py  and .travis.yml work. Also looked at ideas to showcase my coding skill, see: https://techbeacon.com/app-dev-testing/what-do-job-seeking-developers-need-their-github …. Part 2. Polished and pushed diabetes detection project to github

ay 023: Supervised learning on DNA sequences. Classified whether a DNA sequence is a promoter of a gene or not (Dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/ …) Realized you can train and evaluate many different algorithms at the same time and see which gives the best results. 

Day 024: Part 2. Tutorial on NLP (https://www.eduonix.com/learn-machine-learning-by-building-projects …). Very new to this sub-field. Learning the common techniques like tokenizing, stemming, chunking, etc, using the nltk module. Very excited to learn more!

Day 024: Part 1. Deep learning Saturday. Continue reading "Deep Learning with Python". Ch 3. Review on activation function, loss funcs (binary & categorical crossentropy & mean squared error). Plotting loss func vs epoch for validation set to detect overfitting

Day 025 &26: NLP continues. Trying to understand how to use regular expression.

Day 027: NLP continues. Learned tagging words to their part of speech. Used Regexp to parse words. Learned about name entity recognition.

Day 028: NLP. Learned how to train a SVM to predict if a movie review is positive or negative based on the most frequent words in the review. I thought it would do a poor job, but got a 83.6 % accuracy. Not bad!

Day 029: Part 1: Learning how to package my python code using pip. Got stuck trying to upload to pypi. Kind of lost. Part 2: NLP. Polished my notebook on predicting movie review sentiment. Also correctly predicted sentiments of reviews I found online. 

Day 031: Part 1. Read chapter 4 on "Deep Learning with Python". On the issue of overfitting and various methods to overcome it (e.g. dropout, Kfold, regularization).

Day 030 & 031: Part 2: Working on a computer vision problem using a all convolutional neural network with keras (dataset: http://www.cs.toronto.edu/~kriz/cifar.html …, paper: https://arxiv.org/pdf/1412.6806.pdf …). Using google colab for the free GPU power. Still training and I am going to bed.

Day 032: my CNN model can't get >90% on the test set. Training for more epochs. Hopefully doesn't overfit.

Day 033 & 034: my conv net is at 89% now. Still trying to hunt for the next few percents. Hyperparameter tuning is hard! Playing with RandomSearchCV and keras callbacks.

Day 035 & 036: Part 1: Learning how to build python packages (https://python-packaging.readthedocs.io/en/latest/minimal.html …). Part 2: Learned about data augmentation using ImageDataGenerator in keras (https://keras.io/preprocessing/image/ …). It expands my image training set by randomly modifying the images.

Day 037: Worked on the CIFAR-10 dataset. No matter what I did, I still couldn't get it to 90+%. Until I read this: https://machinelearningmastery.com/ensemble-methods-for-deep-learning-neural-networks/ … . I ensemble averaged 11 best models I trained and got 90%! I'm happy now. My notebook: https://colab.research.google.com/drive/10y49y0itU1rrDTtm6Avo-XascMrW-DXB

Day 038&039: Trying to enhance image resolution using Convnet. Having issue with model not learning the training set. From what I have read, dozens of things could have gone wrong. Kind of lost

Day 040-042: Got sick last couple of days. I finished working on using a neural network to enhance image resolution. Here are the results: (left: original, right: enhanced). Overall acceptable. Slight enhancement on the tiger and the zebra looks a bit cartoonish.

Day 043&044: Exploring a related topic--image inpainting. Trying to regenerate deleted parts of an image. Realized my model architecture didn't work. Started exploring variational autoencoder (https://blog.keras.io/building-autoencoders-in-keras.html …) Generative neural networks are fun.

Day 045-047: Spent two days getting the neural network architecture set up. Spent today trying to train the model to reconstruct missing part of an image. Not even overfitting. Something is wrong with the model.
