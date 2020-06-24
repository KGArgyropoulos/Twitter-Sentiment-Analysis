# Twitter-Sentiment-Analysis
This is a Python project on Sentiment Analysis. The data used in here come from the annual competition SemEval, specifically from the International Workshop on Semantic Evaluation 2017, which was about Tweets' sentiment analysis.

This repository contains three folders.
- lexica
    * This folder contains a variety of sentiment dictionaries. Each word takes a continuous value in the range [-1,1] which represents the "valence" of this word (means the intrinsic attractiveness/"good"-ness (positive valence) or averseness/"bad"-ness (negative valence) of an event, object, or situation).
- scr
    * This folder contains our python source file, as well as the junyper ipython notebook, where you can find a more organised code separation and the output of our project.
- twitter data (3 files)
    * This folder contains train2017.tsv, which is the data we use to train our models. There are 28061 tweets with indication possitive, negative or neutral.
    * It also contains test2017.tsv, which is the data we use to test our model and make predictions. There are 12284 tweets, indicated as unknown, as our model will decide whether they are possitive, negative or neutral.
    * Finally, it contains SemEval2017_task4_subtaskA_test_english_gold.txt, which contains the right labels for test2017.tsv. You should not use these labels to train your models. Use them just for verification that your models work fine.

# Implementation
After cleaning up our data, we analyze them and find the most often found words in the entire dataset, the most often words in the positive,the negative and the neutral tweets and we respesent them in 4 Word Clouds.

Then we proceed with the text vectorization of both test and train data using the techniques bag-of-words, tf-idf and word embeddings.\
After extracting the information, we save word embedding's features using .pkl files (comment/uncomment the python script and save/load features).

Afterwards, we calculate the mean value of each tweet, using a dictionary from the lexica folder, and add new features.

Then we proceed with classification using SVM and KNN classifiers, using F1 score measure to calculate our test's accuracy.\
There is an asterisk included at this point, as we faced a problem with the amount of data when running the project in Jupyter's Ipython Notepad. It works fine though, when testing it with Spyder or your terminal. Test it there and you'll get your results.