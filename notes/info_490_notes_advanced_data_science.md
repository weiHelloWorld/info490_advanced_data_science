INFO 490: Advanced Data Science
===========

## Week 1: Introduction to Machine Learning

### Lesson 1: Intro to Machine Learning

- Iris Data and classification problem
    - two convenience plotting functions
        - confusion matrix (e.g. heatmap of 2d histogram using prediction and test data)
        - scatter plot (e.g. pair grid)
    - preprocessing of data
        - since sklearn does not support Pandas DataFrames, we need to extract data as numpy arrays
        - we need to use `cross_validation` module to create training and testing datasets
    - several learning algorithms
        - k-nearest neighbors
        - support vector machine
        - decision trees
        - random forest
            - an example of **emsemble learning**
            - combine decisions of a set of decision trees to make the final decision
    - cross-validation
        - repeatedly select **different training/validation datasets**, and get statistical measure of the performance of a specific algorithm
    - dimensionality reduction
        - PCA
    - clustering
        - k-means

- reading: basic concepts in machine learning
    - ref: http://machinelearningmastery.com/basic-concepts-in-machine-learning/
    - what is machine learning
        - traditional programming vs machine learning
            - ![](figures/f1.png)
        - key elements of machine learning
            - representation
            - evaluation
            - optimization
    - types of machine learning
        - supervised learning
        - unsupervised learning
        - semi-supervised learning
        - reinforcement learning
    - inductive learning
        - definition: given input samples {x} and output samples {f(x)}, the problem is to estimate the function f
        - it is the general theory behind supervised learning
    - some terminology
        - training sample, target function, hypothesis, concept, classifier, learner, hypothesis space, version space

- reading: a few useful things to know about machine learning
    - ref: http://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf
    
### Lesson 2: Introduction to Machine Learning Pre-processing

- preprocessing includes
    - get basic info of data: `describe()`
    - drop and replace missing data: `dropna()`, `fillna()`
    - converting data to type "category" if necessary (convenient to group similar data)
    - convert datetime format using `to_datetime()`

- reading: data, learning and modeling
    - ref: http://machinelearningmastery.com/data-learning-and-modeling/
    - terminology of data
        - instance, feature, data type, datasets, training dataset, testing dataset 
    - terminology of learning
        - induction, generalization, over/under-learning, online/offline learning, supervised/unsupervised learning
    - terminology of modeling
        - model selection, inductive bias, model variance, bias-variance tradeoff

- reading: The Quartz guide to bad data
    - ref: https://github.com/Quartz/bad-data-guide

- reading: Rescaling Data for Machine Learning in Python with Scikit-Learn
    - ref: http://machinelearningmastery.com/rescaling-data-for-machine-learning-in-python-with-scikit-learn/

### Lesson 3: Introduction to Regression

- polynomial fitting
    - `polyfit`
    - overfitting issue
    - pros and cons
        - pros
            - straightforward, easy to use
        - cons
            - using **fixed criterion** and less flexible

- linear regression (LR)
    - LR with `statsmodels`
        - `statsmodels` is a Python library that implements a number of statistical methodology, often in concert with Pandas.
        - basic idea
            - use [**formula language** (defined in `statsmodels.formula.api`)](https://patsy.readthedocs.org/en/latest/formulas.html) to construct model, and pass in training data
                - an example of formula language
                ```Python
                result_noi = smf.ols(formula='y ~ x - 1', data=df).fit()  # "- 1" means no intercept
                ```
            - use `summary()` to see the fitting information (e.g. R-squared, Method, Model, std error, etc)
    - LR with `sklearn`
        - `sklearn` includes lots of different estimators, e.g. Ridge Regression, Lasso, Elastic Net, Orthogonal Matching Pursuit, Bayesian regression, Stochastic Gradient Descent, Robust regression, etc
        - basic idea
            - use `sklearn.linear_model.LinearRegression` to fit data, and use `train_test_split` to do cross-validation
    - ref: http://nbviewer.jupyter.org/urls/s3.amazonaws.com/datarobotblog/notebooks/ordinary_least_squares_in_python.ipynb


## Week 2: General Linear Models

### Lesson 1: Introduction to Multiple Variable Linear Regression

- multiple variable linear regression
    - the difference from single-variable case is to include more variable in the formula language (all other analysis is similar), e.g. 
    ```Python
    result = smf.ols(formula='aDelay ~ dDelay + dTime + np.sqrt(Distance)', data=local).fit()
    ```
    - note 
        - we could use function of a variable to fit the model, such as `np.sqrt(Distance)` in this case
        - we could also handle categorical variables, using notation `C()`, e.g.
        ```Python
        est = smf.ols(formula='chd ~ C(famhist)', data=df).fit()
        ```
        - to study interaction, we can use operator `*` (interaction term with linear terms) or `:` (interaction term only), e.g.
        ```Python
        est = smf.ols(formula='mdvis ~ hlthp * logincome', data=df).fit()
        ```

- reading: assumptions of multiple linear regression
    - ref: http://www.statisticssolutions.com/assumptions-of-multiple-linear-regression/
    - check assumptions using graphs
        - basic idea: plot residues against fitted values, plot residuals
        - ref: section 8.3 of this: https://www.openintro.org/download.php?file=os2_08&referrer=/stat/textbook/textbook_os2_chapters.php

- reading: linear regression with Python
    - ref: http://connor-johnson.com/2014/02/18/linear-regression-with-python/
    - this material covers most information given by `result.summary()`

- reading: multiple regression in Python
    - ref: http://nbviewer.jupyter.org/urls/s3.amazonaws.com/datarobotblog/notebooks/multiple_regression_in_python.ipynb

### Lesson 2: Introduction to Regularization

- example: polynomial fitting
    - firstly we use `PolynomialFeatures()` to transform data x into polynomial terms (1, x, x^2, x^3, ...) and then pass these terms into `LinearRegression()` to do fitting.  These two steps (transformation and fitting) can be pipelined using `make_pipeline()` function, see more info here: http://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html
        - code example
        ```Python
        for idx, deg in enumerate(range(1, max_deg + 1)):
            est = make_pipeline(PolynomialFeatures(deg), LinearRegression())  
            # what this pipeline does is: transform the input (1D data) into corresponding polynomial terms (1, x, x^2, etc)
            # and then pass the terms into LinearRegression() to do fitting
            est.fit(x_train[:, np.newaxis], y_train)

            plt.plot(xf, est.predict(xf[:, np.newaxis]), c=cmp[deg - 1], label='{0} Order'.format(lbls[idx]))
        ```
    - by computing the MSE of the testing data, we can see the overfitting issue, and find the optimal fitting degree
    - regularization techniques
        - Ridge regression
            - basic idea: add L2 norm of regression coefficients as the penalty term
        - Lasso regularization
            - basic idea: add L1 norm of regression coefficients as the penalty term
            - due to the properties of L1 norm, many coefficients will become 0 at the optimal solution.  So Lasso typically gives a **sparse solution**
        - elastic net
            - basic idea: combine Ridge and Lasso
            - ref: https://en.wikipedia.org/wiki/Elastic_net_regularization#Specification

- reading: Understanding the Bias-Variance Tradeoff
    - ref: http://scott.fortmann-roe.com/docs/BiasVariance.html
    - definition
        - given a model, we get different functions for different training data.  
            - **bias**: the difference between the average prediction value (given by functions with different training data) and the expected value. 
            - **variance**: the difference among different predicted values (given by functions with different training data)
    - bias-variance tradeoff
        - ![](figures/f2.png)

### Lesson 3: Introduction to Logistic Regression

## Week 3: Introduction to Supervised Learning

### Lesson 1: k-Nearest Neighbors

- k-nearest neighbors
    - use `KNeighborsClassifier()`
        - by default, the distance function is [Minkowski distance](https://en.wikipedia.org/wiki/Minkowski_distance)
        - by default, each neighboring point has the same weight, you can modify that in `weights` parameter
        - ref: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    - model evaluation
        - use `seaborn.heatmap()` to visulize **confusion matrix**
        - use `sklearn.metrics.classification_report()` to generate a report about **precision, recall, and f1-score**
            - precision: true positives / (true positives + false positives)
            - recall: true positives / (true positives + false negatives)
            - ref
                - https://en.wikipedia.org/wiki/Precision_and_recall
                - https://en.wikipedia.org/wiki/F1_score

- reading: how to evaluate machine learning models: classification metrics
    - ref: http://blog.dato.com/how-to-evaluate-machine-learning-models-part-2a-classification-metrics
    - ways of measuring classification performance
        - accuracy: number of correct predictions / number of total data points
        - per-class accuracy: average of accuracies of all classes
        - confusion matrix
        - log-loss: deals with probability values, similar to information entropy
        - AUC (Area Under the Curve)
            - the curve here is [receiver operating characteristic curce](https://en.wikipedia.org/wiki/Receiver_operating_characteristic), which illustrates the performance of a binary classifier system as its **discrimination threshold** is varied, by plotting **true positive rate (TPR)** vs. **false positive rate (FPR)**
                - note
                    - it only works for **binary classifier** system, does not work for multiple classes, how do we use it for multiple classes? one way is to use `sklearn.preprocessing.label_binarize()` to binarize the classification labels, read this: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.binarize.html
                    - how do we determine whether an item belongs to a positive or a negative class? by comparing the score (calculated using input) with a threshold value.  If it is greater than threshold, then positive
                    - each data point on ROC curve corresponds to a threshold value
                    - as the threshold value increases, both TPR and FPR will increase

- reading: how to evaluate machine learning models: ranking and regression metrics
    - ref: http://blog.dato.com/how-to-evaluate-machine-learning-models-part-2b-ranking-and-regression-metrics
    - ranking metrics
        - precision-recall and f1-score
        - NDCG
            - main idea: assign different weights to different items (unulike precision-recall where we assign equal weights to all items)
    - regression metrics
        - RMSE
        - quantiles of errors
            - e.g. median or 90th percentile of absolute percent error
        - "almost correct" predictions
            - the percent of estimates that differ from the true value by no more than X%

### Lesson 2: Support Vector Machine

- introduction to SVM
    - three types of classifiers related to SVM
        - maximal margin classifier (MMC)
            - basic idea: use a hyperplane to segregate two classes, to maximize **margin**
            - constructing MMC
            ![](./figures/f3.png)
            - **support vectors** are those that affect the boundary lines, in this case, the support vectors are points located exactly on the edge of the margin
        - support vector classifier (SVC)
            - motivation: relax the requirement that a separating hyperplane will perfectly separate every training observation on the correct side of the line, using what is called a **soft margin** 
            - constructing SVC
            ![](./figures/f4.png)
        - support vector machine (SVM)
            - motivation: how to deal with non-linear decision boundaries?
                - one way is to construct additional non-linear features from existing features, as we did in linear regression, an example is give below
                    - the problem with this method is it can be quite **computationally expensive**
                    ![](./figures/f5.png)
                - another way is the **kernel trick**
                    - kernel trick is based on the fact that the SVC classifier **only depends on inner products** of observations, not observations themselves, the classifier could be written as $f(x)=\beta_0+\sum_{i=1}^n {\alpha_i \langle x,x_i \rangle}$
                    - we can see inner product as the **similarity function**, and we can replace this with another kernel function $K(x_i, x_k)$, one example is radial kernel, which decays exponentially with the distance, so it has **extremely localized behavior**, see [here](https://www.quantstart.com/articles/Support-Vector-Machines-A-Guide-for-Beginners)
                    - kernel trick is much more efficient than adding features approach
    - ref
        - https://www.quantstart.com/articles/Support-Vector-Machines-A-Guide-for-Beginners
        - Chapter 9 of Introduction to Statistical Learning
        - http://www.analyticsvidhya.com/blog/2015/10/understaing-support-vector-machine-example-code/

### Lesson 3: Naive Bayes

- reading: sklearn documentation of Naive Bayes algorithm
    - ref: http://scikit-learn.org/stable/modules/naive_bayes.html (a very good resource for introduction to NB algorithm)

- reading: 6 easy steps to learn Naive Bayes algorithm
    - ref: http://www.analyticsvidhya.com/blog/2015/09/naive-bayes-explained/
    - what is NB algorithm?
        - theory: $p(c|x) = p(x|c) p(c)/p(x)$
        - assumption: independence among predictors
        - it is particularly useful for **very large data sets**

## Week 4: Tree Algorithms and Ensemble Techniques

### Lesson 1: Decision Trees

- intro to decision trees
    - basic idea: construct a decision tree by **recursively splitting** a data set into new groupings based on a **statistical measure (e.g. histogram)** of the data along each different dimension 

- reading: A Visual Introduction to Machine Learning
    - ref: http://www.r2d3.us/visual-intro-to-machine-learning-part-1/
    - this is a very nice visualization to decision tree algorithm

### Lesson 2: Ensemble Techniques: Bagging

- random forest
    - basic idea: build a set of decision trees, and combine predictions from the different trees statistically to make a final prediction, it is an example of **bagging approach of ensemble learning**
        - **Bagging** means that when building each subsequent tree, we don’t look at the earlier trees, while in boosting we consider the earlier trees and strive to compensate for their weaknesses
        - ref: http://fastml.com/intro-to-random-forests/
    - steps
        - randomly divide data into several datasets, and train multiple decision trees
        - Each tree is grown to the largest extent possible and there is no pruning
        - Predict new data by aggregating the predictions of the ntree trees (**majority votes for classification, average for regression**)
    - tuning random forest
        - the most important parameter is the number of trees
            - usually, the more, the better, because the multitude of trees serves to **reduce variance**
            - ref: http://fastml.com/what-is-better-gradient-boosted-trees-or-random-forest/
    - ref: http://www.analyticsvidhya.com/blog/2015/09/random-forest-algorithm-multiple-challenges/

### Lesson 3: Ensemble Techniques: Boosting

- basic idea: build new better estimators based on previous estimators

## Week 5: Introduction to Unsupervised Learning

### Lesson 1: Introduction to Dimension Reduction

- PCA
    - basic idea
        - given zero-mean (by shifting the center of data) data set $X = [x_1, x_2, ..., x_n]$, where $x_i = \left[x_{1i}, ..., x_{mi}\right]^T$, a **orthogonal transformation** $W$, let $Y = WX$, $Y$ is the transformed data. find $W$ such that sum of squares of the 1st component of y's reaches maximum, and sum of squares of 2nd component reaches its maximum after eliminating 1st component, and ...
        - PCA does dimension reduction by ensuring that first few components **capture most of variance** of the data
            - metric: fraction of explained variance
    - PCA using `sklearn`
        - get information of PCA transformation 
            - `explained_variance_ratio_`
            - `components_`: transformation coefficients
        - reconstruction using first few componnets
        ```python
        pca = PCA(n_components, copy=True)
        tx = pca.fit_transform(x)
        hd.plot_numbers(pca.inverse_transform(tx)[0:10])
        ```
    - calculation of PCA
        - calculate covariance matrix, and find its eigenvalues and eigenvectors
        - diagonize the covariance matrix, sort all these components based on eigenvalues
        - pick first few components to form a transformation matrix and do transformation
    - note
        - PCA is often used together with other machine learning techniques, as a **preprocessing** method to compress data
        - scaling in one dimension of data may affect the PCA result, since it affects the covariance matrix
    - ref
        - http://sebastianraschka.com/Articles/2014_pca_step_by_step.html
        - http://www.lauradhamilton.com/introduction-to-principal-component-analysis-pca

### Lesson 2: Introduction to Clustering

- introduction to clustering
    - several clustering algorithms
        - Connectivity based clustering (hierarchical clustering)
        - Centroid-based clustering
        - Distribution-based clustering
        - Density-based clustering
    - ref
        - https://en.wikipedia.org/wiki/Cluster_analysis

- K-means
    - basic idea
        - step 1 (initialization): choose the number of classes to be k, pick k centroids (say, randomly)
        - step 2 (reassign points): assign every point to the cluster whose centroid is nearest to it
        - step 3 (update centroid): calculate center of all points in the cluster, as its new centroid, go to step 2, repeat until converge
    - properties
        - convergence
            - it is guaranteed to converge
        - uniqueness
            - the result may depend on the initial conditions, e.g. if we apply k-means with k = 4 to 3 clearly separate clusters, the result may be not unique
    - ref
        - http://www.naftaliharris.com/blog/visualizing-k-means-clustering/ (a very nice visualizaiton)

- DBSCAN
    - basic idea
        - we choose two parameters, a radius epsilon, and number of minPoints n
        - several concepts
            - core point
            - reachable (not a symmetric relation)
            - density-connected (a symmetric relation)
        - all mutually density-connected points form a cluster
        - points that are not density-connected to any other point are **outliers**
    - comparison with k-means
        - see [wiki](https://en.wikipedia.org/wiki/DBSCAN)
    - ref
        - https://en.wikipedia.org/wiki/DBSCAN
        - http://www.naftaliharris.com/blog/visualizing-dbscan-clustering/ (a nice visualization)

### Lesson 3: Density Estimation

- ref
    - https://en.wikipedia.org/wiki/Density_estimation
    - http://ned.ipac.caltech.edu/level5/March02/Silverman/Silver_contents.html
    - http://stanford.edu/~mwaskom/software/seaborn/tutorial/distributions.html
    - http://www.lancs.ac.uk/~struijke/density/index.html

## Week 6: Machine Learning Special Topics

### Lesson 1: Introduction to Recommender Systems

- basic idea
    - find similar users using some metrics, e.g. [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
    - do recommendations based on the similar users

- reading: Recommender systems, Part 1: Introduction to approaches and algorithms
    - ref: http://www.ibm.com/developerworks/library/os-recommender1/index.html
    - basic approaches
        - collaborative filtering
        - content-based filtering
        - hybrids
    - algorithms of recommender systems
    - challenges with recommender systems
        - non-typical behavior
        - scalability
        - privacy-protection considerations

### Lesson 2: Introduction to Anomaly Detection

- basic idea
    - visual analysis
    - statistical analysis
        - calculate mean/std, trimmed mean/std (which are less sensitive to outliers)
    - cluster analysis
        - we can use `DBSCAN` to find clusters, and identify noise points with `label = -1`
    - classification analysis

- ref
    - https://en.wikipedia.org/wiki/Anomaly_detection
    - http://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm

### Lesson 3: Introduction to Practical Concepts

- feature scaling

- feature selection
    - motivation
        - to reduce amount of data we need to process, one way is to apply dimension reduction techniques, e.g. PCA.  But when the raw data are too large (even for doing dimension reduction), it might be helpful to first select some of the features, before doing any processing.
    - methods
        - [recursive feature elimination](http://scikit-learn.org/stable/modules/feature_selection.html#recursive-feature-elimination)
            - basic idea
                - train the data with an external estimator
                - eliminate the feature with smallest weight (related to the estimator)
                - repeat until the desired number of features is reached
        - [random forest classifier](http://scikit-learn.org/stable/modules/feature_selection.html#tree-based-feature-selection)
            - basic idea
                - build RFC by randomly select features for each tree, and RFC will compute the overall importance of each feature, which is used for feature selection 
            - ref
                - http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#example-ensemble-plot-forest-importances-py
                - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    - ref
        - https://en.wikipedia.org/wiki/Feature_selection

- pipeline
    - package multiple machine learning techniques together, similar to Linux command line pipeline

- cross validation
    - several types
        - `KFold`, `StratifiedKFold`, `LeaveOneOut`, `ShuffleSplit`
    
- grid search
    - basic idea
        - use `GridSearchCV` object to perform a grid search to tune **hyperparameters** (e.g. learning rate for neural network, `eps` and `min_samples` for DBSCAN) for a machine learning model
        - ref
            - http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html

- validation/learning curves
    - basic idea
        - use `sklearn.learning_curve.validation_curve` to plot training score curve and validation score curve

## Week 7: Introduction to Text Analysis

### Lesson 1: Introduction to Text Analysis

- text analysis with `sklearn`
    - use `CountVectorizer` to get tokens (features), and construct [bag of words](https://en.wikipedia.org/wiki/Bag-of-words_model)
    - frequency analysis
    - [document-term matrix](https://en.wikipedia.org/wiki/Document-term_matrix)

- text analysis with `NLTK`

- reading: the definite guide to natural language processing
    - ref: https://blog.monkeylearn.com/the-definitive-guide-to-natural-language-processing/
    - applications of NLP
        - machine translation
        - automatic summarization
        - sentiment analysis
            - goal: identify subjective information (e.g. an opinion or an emotional state) in texts
            - a frequent kind: polarity detection
        - text classification
        - conversational agents
    - how does machine understand text?
        - understanding words
            - identify the nature of each word: **Part-of-Speech (PoS) tagging**
                - two commonly used approaches
                    - symbolic (based on rules)
                        - example: [Brill tagger](https://en.wikipedia.org/wiki/Brill_tagger)
                    - statistical (based on machine learning)
        - from words to structure
        - getting the meaning of words
            - [lexical semantics](https://en.wikipedia.org/wiki/Lexical_semantics)
            - [word sense disambiguation](https://en.wikipedia.org/wiki/Word-sense_disambiguation)
        - a word about pragmatics
            - understand sentences based on context
        - are all these analyses needed?
            - no, depends on the application
        - symbolic vs statistical
            - symbolic: easier to understand and control, but expensive
            - statistical: more efficient, but harder to interpret

### Lesson 2: Introduction to Text Classification

- text classification
    - basic idea
        - use Document-term matrix (i.e. the number of times each token appears) of the text and related tags to do training for a classifier (e.g. Naive Bayes classifier)
    - classification with stop words
        - stop words: common words with little useful information, e.g. a, the, in, of
        - the training result may be improved when removing stop words in `CountVectorizer`
    - TF-IFD
        - motivation: previously we simply use **the number of times a token occurs** in a document, this may overemphasize tokens that generally occur across many documents (e.g. names or general concepts), to improve that, we use **tf-idf** to calculate how important a word is to a document in a corpus
        - tf & idf
            - tf: term frequency, shows how frequently a term appears in a document
            - idf: inverse document frequency, if a term appears in many documents, we can assume that this term is not important
        - in sklearn, we use `TfidfVectorizer()` to generate tf-idf values for classification
        - ref
            - https://en.wikipedia.org/wiki/Tf%E2%80%93idf
            - http://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting
            - http://blog.christianperone.com/2011/09/machine-learning-text-feature-extraction-tf-idf-part-i/
    - text classification with other classifiers
        - SVM
        - SGD
        - logistic regression
    - sentiment analysis

- ref
    - https://en.wikipedia.org/wiki/Text_mining

### Lesson 3: Introduction to Text Mining

- text mining
    - n-grams
        - motivation: previous we are using "unigrams", and a sequence of more words may be more informative, this would improve classification power
        - ref: https://en.wikipedia.org/wiki/N-gram
    - stemming
        - ref: https://en.wikipedia.org/wiki/Stemming
    - lemmatisation
        - similar to stemming, but with knowledge of context
        - ref: https://en.wikipedia.org/wiki/Lemmatisation
    - dimension reduction and feature selection

## Week 8: Introduction to Social Media

### Lesson 1: Introduction to Social Media: Email

- email and text classification
    - structure of email
    - spam email classification, and blind test

- reading: Document Classification with scikit-learn
    - ref: http://zacstewart.com/2015/04/28/document-classification-with-scikit-learn.html
    - this is a good review of some concepts
        - feature extraction
        - pipelining
        - cross-validation
        - bags of words
        - tf-idf

### Lesson 2: Introduction to Social Media: Twitter

- mining twitter data and do classification

- ref
    - https://en.wikipedia.org/wiki/Twitter
    - http://online.wsj.com/public/resources/documents/TweetMetadata.pdf
    - http://tweepy.readthedocs.org/en/v3.5.0/
    - http://www.nltk.org/howto/twitter.html

### Lesson 3: Introduction to Social Media: Web

- mining web data
    - webscraping with `BeautifulSoup`
        - ref
            - http://www.gregreda.com/2013/03/03/web-scraping-101-with-python/
            - http://www.gregreda.com/2013/04/29/more-web-scraping-with-python/
    - working with Javascript data

## Week 9: Introduction to Natural Language Processing

### Lesson 1: Basic Concepts

- introduction to NLP
    - intro
        - previously we use simple tokenization (word counts, tf-idf, etc) to do text analysis, now we need to combine semantic information of the text to do natural language processing
    - tokenization by sentences/words/etc using `nltk`
        - `sent_tokenize()`, `word_tokenize()`, `WhitespaceTokenizer`, `WordPunctTokenizer`
    - collocations
        - use `nltk.collocations` to find collocations, and we can measure the importance of collocations using **pointwise mutual information** (which is likelihood of words occurring together)
        - we may also apply filters to collocations results
    - tagging
        - previously we use "bag-of-words" model, which lost information about the difference between two meanings of a specific word.  **tagging** is used to solve this issue by adding **information about context or the grammatical nature (noun or verb) of a word**
        - types of tagging
            - [part-of-speech tagging](https://en.wikipedia.org/wiki/Part-of-speech_tagging)
            - [named entity recognition](https://en.wikipedia.org/wiki/Named-entity_recognition)
        - linking tags
        - tagged text extraction

### Lesson 2: Topic Modeling

- introduction to topic modeling
    - basic idea
        - clustering based on content
    - clustering with non-negative matrix factorization (NMF)
        - algorithm: given a non-negative m-by-n matrix V, and value r (where $r << m, r << n$), find a m-by-r matrix W and r-by-n matrix H, to minimize $|V-WH|^2$
        - ref
            - https://en.wikipedia.org/wiki/Non-negative_matrix_factorization#Background (this is a nice example of explaining how it works for topic modeling)
    - latent Dirichlet allocation
    - topic modeling with Gensim

- ref
    - https://en.wikipedia.org/wiki/Topic_model
    - http://journalofdigitalhumanities.org/2-1/topic-modeling-a-basic-introduction-by-megan-r-brett/
    - http://journalofdigitalhumanities.org/2-1/topic-modeling-and-digital-humanities-by-david-m-blei/
    - http://blog.echen.me/2011/08/22/introduction-to-latent-dirichlet-allocation/

### Lesson 3: Semantic Analysis

## Week 10: Network Analysis

### Lesson 1: Graph Concepts

- ref
    - https://en.wikipedia.org/wiki/Graph_(abstract_data_type)
    - https://en.wikipedia.org/wiki/Adjacency_matrix
    - https://en.wikipedia.org/wiki/Incidence_matrix

### Lesson 2: Graph Analysis

- introduction to graph analysis
    - graph operations
    - graph generators
    - graph I/O
    - graph analysis

- graph traversal
    - ref: https://en.wikipedia.org/wiki/Graph_traversal

- graph as a data structure and its implementation in Python
    - ref: http://www.python-course.eu/graphs_python.php

### Lesson 3: Social Media

- ref
    - https://en.wikipedia.org/wiki/Social_network_analysis
    - http://www.orgnet.com/sna.html
    
## Week 11: Probabilistic Programming

### Lesson 1: Introduction to Bayesian Modeling

- Bayesian analysis using `pymc3` library
    - basic idea
        - given prior and data, we could use two approaches to do model fitting
            - maximum a posteriori (MAP) methods
                - find the optimal point (MAP point) in parameter space using optimization methods
            - sampling methods
                - basic idea: **sample the posterior** in parameter space using MCMC techniques
                    - note that we sample the parameters based on prior distribution, and find the probability that is posterior
                - trace analysis
                    - **trace** will be generated from a chain of sampled parameter values, and we could do some analysis such as:
                        - posterior sample distribution
                        - autocorrelation
                    - note that the traces may take a while to settle down to most likely values, this part is called **burn-in period**, we need to discard this part either manually or use `find_MAP()` to find a good point as the starting point for sampling
    - ref
        - http://pymc-devs.github.io/pymc3/getting_started/
        - http://blog.applied.ai/bayesian-inference-with-pymc3-part-1/
        - http://nbviewer.jupyter.org/github/markdregan/Bayesian-Modelling-in-Python/blob/master/Section%201.%20Estimating%20model%20parameters.ipynb
        - http://nbviewer.jupyter.org/github/markdregan/Bayesian-Modelling-in-Python/blob/master/Section%202.%20Model%20checking.ipynb

- Markov chain Monte Carlo methods
    - basic idea
        - construct a Markov chain such that when we iterate through this chain for many steps, we get a reasonable sample set of a specific probability distribution
    - some algorithms
        - Metropolis-Hastings algorithm
            - ref: https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm#Intuition
        - Gibbs sampling
            - ref: https://en.wikipedia.org/wiki/Gibbs_sampling
    - ref 
        - https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo
        - http://stats.stackexchange.com/questions/165/how-would-you-explain-markov-chain-monte-carlo-mcmc-to-a-layperson

### Lesson 2: Introduction to Hierarchical Modeling

- basic idea
    - given multiple sets of data (of the same type) which follow the same model (e.g. linear model) with different parameter values (e.g. different slopes and intercepts), we may use three ways to model them
        - completely pooled model
            - combine all datasets together to fit a set of glabal parameters for all datasets
            ![](./figures/f6.png)
        - unpooled/non-hierarchical model
            - treat them as completely independent datasets, and fit the model to get parameters for each dataset separately.
            ![](./figures/f7.png)
        - hierarchical/partially pooled/multilevel model
            - treat the parameters as data generated from a higher-level model, for example, we can assume all these slopes in the linear models are drawn from a normal distribution and then find the "hyperparameter" for the normal distribution
            - advantage
                - it takes the correlation of these models into account, and for some real problems this might be an important factor
            - shrinkage effect
                - the parameters of these model will be "pulled" towards the group mean, as a result of the common group distribution
            ![](./figures/f8.png)

- ref
    - http://pymc-devs.github.io/pymc3/GLM-hierarchical/

### Lesson 3: Introduction to General Linear Models

- TODO

## Week 12: Cloud Computing

### Lesson 1: Introduction to Hadoop

- ref
    - https://aws.amazon.com/what-is-cloud-computing/
    - https://aws.amazon.com/types-of-cloud-computing/
    - https://en.wikipedia.org/wiki/Apache_Hadoop

### Lesson 2: Introduction to MapReduce

- introduction
    - map/reduce phases
        - map phase
            - do computation on data in parallel, this could be fast when using HDFS, where data are widely distributed
            - identifies keys and associates with them a value
        - reduce phase
            - collect keys and aggregate the values

- ref
    - http://www.tutorialspoint.com/map_reduce/map_reduce_introduction.htm
    - http://www.tutorialspoint.com/map_reduce/map_reduce_algorithm.htm

### Lesson 3: Introduction to Pig

- introduction
    - Pig is a tool in the Hadoop ecosystem to simplify the creation of MapReduce programs.  The script is written in **Pig Latin**, a dataflow language.
        - Pig Latin vs traditional procedural and object-oriented programming languages
            - Pig Latin focuses on data flow rather than control flow, there are no `if` statements or `for` loops.
        - Pig Latin vs SQL
            - SQL is a query language, it allows users to form queries, but not how it is done.  In Pig Latin, users need to describe how it is done.
        - Pig Latin vs MapReduce
            - Pig Latin is higher-level, it is much easier to write and maintain with Pig Latin than MapReduce.
    - Pig example: word count
    ```Pig
    Lines = LOAD 'book.txt' AS (Line:chararray) ;
    Words = FOREACH Lines GENERATE FLATTEN (TOKENIZE (Line)) AS Word ;
    Groups = GROUP Words BY Word ;
    Counts = FOREACH Groups GENERATE group, COUNT (Words) ;
    Results = ORDER Counts BY $1 DESC ;
    Top_Results = LIMIT Results 10 ;
    STORE Results INTO 'top_words' ;
    DUMP Top_Results ;
    ```

- ref
    - http://chimera.labs.oreilly.com/books/1234000001811/ch01.html#use_cases
    - https://en.wikipedia.org/wiki/Pig_(programming_tool)

## Week 13: NoSQL DataStores

### Lesson 1: Introduction to Document-Oriented Databases

- introduction to MongoDB
    - MongoDB is **schema-less**, there are no tables or schemas.  but it also supports some of traditional database operations such as inserting data, querying data, updating data, and deleting data

- ref
    - http://www.w3resource.com/mongodb/nosql.php (a nice introduction to NoSQL, including motivation, history, pros/cons, etc, and also some intro to scalibility issues)
    - https://en.wikipedia.org/wiki/Document-oriented_database (intro to document-oriented databases, including comparison with traditional relational databases)
    - http://www.w3resource.com/mongodb/introduction-mongodb.php
    - http://api.mongodb.org/python/current/tutorial.html (using MongoDB with Python)

### Lesson 2: Introduction to Column-Oriented Databases

- ref
    - https://en.wikipedia.org/wiki/Column-oriented_DBMS (compare column-oriented with traditional row-oriended databases)
    - https://academy.datastax.com/resources/brief-introduction-apache-cassandra
    - https://datastax.github.io/python-driver/getting_started.html

### Lesson 3: Introduction to Graph Databases

- ref
    - https://en.wikipedia.org/wiki/Graph_database
    - http://www.tutorialspoint.com/neo4j/neo4j_overview.htm
    - https://www.safaribooksonline.com/blog/2013/07/23/using-neo4j-from-python/

