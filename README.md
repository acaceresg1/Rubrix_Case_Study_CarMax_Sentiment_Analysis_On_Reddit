# Rubrix_Case_Study_CarMax_Sentiment_Analysis_On_Reddit

## Code and Resources Used:
Products: 
- Rubrix (by Recognai)
- Streamlit
Packages: 
- Scrapping & Merging: json, praw, PushshiftAPI, pandas, datetime, calendar, requests, time, re, glob
- Tokenization: pandas, wordcloud, matplotlib
- Sentiment Analysis: pandas, spacy, rubrix, transformers, numpy, datasets, sci-kit learn
- Word Cloud: pandas, wordcloud, matplotlib, math, collections

## Executive Summary:

Final Group Project (Capstone Project) for the Business Analytics and Big Data Master at IE University.

Social media plays a vital role in the digital transformation of businesses as it gives companies a unique and personal perspective on how their customers interact with their brand and products.
As such, Natural Language Processing (NLP) has evolved from performing tasks related to speech-containing signals over radios to classifying sentiments over social media platforms such as Reddit, Twitter, and more.

This project took a closer look at the subreddit r/whatcarshouldIbuy to uncover consumer sentiments regarding particular brands,
specific car types and models, and other buying patterns and behaviors. The goal was to not only garner insights and recommendations for a car dealership, CarMax but also to test
the ease of use of Rubrix, a product by Recognai, which assists in the labeling of sentences with the end goal of having a more accurate sentiment, as a part of the NLP pipeline.
The steps are summarized below:

 - **Scraping:** Used PRAW and Pushshift API to extract Reddit posts and their comments
 - **Tokenization:** Used KeyBERT and Python’s tokenize model to identify keywords/tokens
 - **Sentiment Analysis:** Applied Spacy to split sentences from the scraped and tokenized data. Used Hugging Face Hub’s SST2 as a baseline model. Uploaded the data to Rubrix to relabel 1-2% of the results, added a new label for a total of 3 (POSITIVE, NEGATIVE, NEUTRAL), and applied the new data from Rubrix back to SST2 to fine-tune our model and get better sentiment results.
 - **Word Cloud:** Created color-coordinated word clouds to visualize results and combine tokenization and sentiment analysis.
 
Overall, the goal is to to address two clients, CarMax and Rubrix, and to provide insights and feedback for both. The technical approach will be building and fine-tuning a sentiment classifier and
tokenizer from scratch to classify our 1 million Reddit posts and 3 million + Reddit comments on the topics surrounding what cars users should buy.
 
 #### Introduction to Rubrix
 
Rubrix is a free and open-source tool for data-centric natural language processing, “which provides a production-ready framework for
building and improving datasets for natural language processing projects” (Rubrix Documentation, 2022). Given that typical natural language processing work requires
hand-labeling workflows that are both time-consuming and costly but still require keeping humans in the loop, Rubrix combines both hand-labeling with active learning, bulk-labeling,
zero-shot models, and weak supervision to optimize labeling and be more time-efficient.
Importantly, Rubrix aims to close the gap between handling data collection as one-off activity and machine learning projects that thrive with the number of iterations done. Hence, the tool
enables researchers and scientists doing natural language processing projects to iterate as much as needed with ease. Notably, Rubrix is compatible with major NLP libraries (Hugging
Face, Spacy, Stanford Stanza & Flair) (Rubrix Documentation, 2022).
 
The instalation of Rubrix is simple. In the following link everything is well explained:

https://rubrix.readthedocs.io/en/stable/getting_started/setup%26installation.html

## Data Set Creation: Scraping & Merging
The first step was to scrape the data from Reddit. To work around some limitations of PRAW (Python Reddit API Wrapper) were combined PRAW and Pushshift API getting the post IDs from Pushshift and then the actual post metadata from the PRAW API, using the scraped IDs.
This way was able to speed up the process. As the goal is to scrap the subreddit from 2015 to 2022, a vast amount of data, a RAM limitation would appear sooner or later. The solution was to pass timestamps to the API to scrape the
posts in smaller batches (by month) and save them in CSV files, merging them later for analysis.

Before saving the scraped data in CSV format, some data cleaning was applied:
  1. Text formatting for readability, e.g. removing new line (‘\n’ symbols)
  2. Cleaning dates and timestamps
  3. Adding 'https://www.reddit.com' to the scraped permalinks
  4. Selecting the relevant columns for our analysis: ['full_link', 'subreddit', 'post keywords', 'id', 'date', 'score', 'num_comments', 'author', 'title', 'selftext', 'top_comment', 'comment_score']

Once all of the functions were put together and ran by month since 2014, the total data collected were 1M Reddit posts and +3M Reddit comments. To find the comments for each post in our analysis, we could simply query the comments by their parentID (postID).

## Tokenization:

The tokenization part helps to find the most important terms within the scraped Reddit posts. 
The first attempt at keyword extraction was using the KeyBERT algorithm, a keyword extraction library that is used to get the most representative keywords from text documents. Unfortunately, it did not give enough information.

The final approach used Python’s tokenize module for the following:

  1. Filter out English stop words to remove unwanted words from our corpus as they do not provide any information to the model.
  2. Tokenize or split the Reddit posts into smaller lines and/or words. Tokenization helps to understand the context and develop a better model.
  3. Lemmatize to convert words to their base form so that they can be analyzed as a single item
![image](https://user-images.githubusercontent.com/115701510/196164245-59e377f4-bee6-4e77-a196-7f76421d2497.png) ![image](https://user-images.githubusercontent.com/115701510/196164720-a7dcc397-10fc-4f07-abdd-94975686d496.png)


After the tokenization, unnecessary stop words were dropped by creating a stoplist as  they are extremely common and uninformative with little discriminative value.
The final goal was to create WordCloud analysis, “a visualization technique where each word is picturized
with its importance and context”. Python’s word cloud library was used to generate the visualizations

## Sentiment Analysis:

The second undertaking was to fine-tune a sentiment classifier for the used-car topic, starting with no labeled data. The schema for the process was the following:

![image](https://user-images.githubusercontent.com/115701510/196166543-a29ad7fa-86fc-485e-b3f9-12cfb8802ba7.png)

To reach the goal of building and fine-tuning a sentiment classifier from scratch to classify 1M Reddit posts and +3M Reddit comments, were followed the subsequent steps primed by
Rubrix's documentation (2022):

  1. Scraped data using Reddit’s API.
  2. Read scraped data with the initial Jupyter Notebook.
  3. Applied Spacy to create a list with all the comments split by sentences. This step was essential since the sentiment classifier was designed to work by sentence instead of total comments.
  4. Tuned the most popular sentiment classifier on the Hugging Face Hub, which was fine-tuned on the SST2 sentiment dataset, the distillery-base-uncased-finetuned-sst-2-English.
  5. Labeled a training dataset from a shuffle sample of all the comments from the Subreddit "whatcarshouldIbuy" as a baseline to get initial pre-trained sentiment classifier predictions. Importantly, this step was crucial to establish the baseline that will be compared after using Rubrix.
  6. Fine-tuned the pre-trained classifier with the training dataset. For this step, a sample is uploaded into Rubrix, labeling a percentage of the total. Importantly, a NEUTRAL extra-label was created to represent NEUTRAL comments. Nonetheless, the original document only had binary predictors ("POSITIVE" and "NEGATIVE").
  7. Labeled more data by correcting the predictions of the fine-tuned model.
  8. Lastly, fine-tuned the pre-trained classifier with the extended training dataset.

Given the complexity of each separate step, these steps were aggregated into six critical parts described below:

### I. Step One: Running the pre-trained model over the dataset

For the first step, the pre-trained model was used to predict over our raw dataset. In order to do so, the Rubrix dataset was created first for labeling, then uploaded everything to Rubrix.

![image](https://user-images.githubusercontent.com/115701510/196168252-c19dc064-16d6-4865-bf59-cec454db599e.png)

### II.Step Two: Exploring and Labeling the data with a Pre-Trained Model

It is important to explore how the pre-trained model developed in the previous step performs with the current dataset. Notably, the pre-trained sentiment
classifier only had two labels: POSITIVE and NEGATIVE. However, after checking the comments, was found a need for adding a third label that could represent comments that were
neither POSITIVE nor NEGATIVE but NEUTRAL. Importantly, the Rubrix case study used to guide this Capstone Project did not have a NEUTRAL label.
To define the labels, it was necessary to establish rules and guidelines that clarified the reason for each classification.
Main set of rules is as follows:

1. A recommendation of a car brand or model is considered POSITIVE.
2. A selection of some cars on a list (are in the titles) are considered POSITIVE.
3. If the comment has something like: I love the car, but there is only one problem, it is POSITIVE.
4. If the comment has something like: I hate the car, but there is only one good thing: NEGATIVE.
5. If it has a balance of some negatives and some positives, it is NEUTRAL.
6. If it speaks about something around cars without influence, it is NEUTRAL.
7. Questions about reliability or models or whatever are NEUTRAL.
8. Questions about problems are NEGATIVES.
9. Questions about good features are POSITIVES

The rules established, helps to label the data. The Rubrix documentation provided two guidelines to optimize the results. First, to start
labeling examples sequentially without using the search feature not to skew the results. After understanding the data, the second was to use filters and search features to annotate examples
for specific labels.

Thereafter, the first 100.000 sentences of the sample list were uploaded to Rubrix, and 1000 records were validated and labeled by hand, representing 1% of the total
number of sentences. Importantly, the baseline classifier was tuned with those inputs.

### III. Step 3: Fine-Tunning the Pre-Trained Model

Firstly, to fine-tune the model, was used a Trainer API called Transformers from Hugging Face, a popular Python library providing pre-trained models.
The annotations validated were unloaded from Rubrix using the load method. Afterward, the training and testing datasets were created and prepared for the sentiment classifier, using the datasets library, splitting the data into the training and
evaluation datasets. To fine-tune the distilbert-base-uncased-finetuned-sst-2-English, the model was loaded into a python script.
Finally is created and saved a sentiment-classifier tuned with the first labeling with Rubrix

![image](https://user-images.githubusercontent.com/115701510/196177350-c247100e-8516-4d80-85d7-404a2ff1e2bd.png)

### IV. Step 4: Testing the Fine-Tuned Model

Once the sentiment-classifier is tuned, it is possible check using some senteces as examples their differences predicting the labels.

### V. Step 5: Running the Fine-Tuned Model over the dataset and Log predictions:

After testing the model, another dataset was created using the records that were not annotated in Step 1, and to do so, were downloaded from Rubrix only sentences with a Default status, meaning they were not labelled, to fetch the nonassigned
labels. This is comparable to the first step but now using the fine-tuned model. At the end of this step, this new dataset was uploaded back to Rubrix to label by hand another % of the sentences.

![image](https://user-images.githubusercontent.com/115701510/196178064-8b7771d6-1bdf-4fb4-abc9-9a2dfe4b51c4.png)

### VI. Step 6: Exploring and Labeling with the fine-tuned model.

This last step starts exploring how the fine-tuned model performed with our dataset in Rubrix. Then was labeled more than 300 annotated examples of sentences to further refine our
classifier. Moreso, given that the labels were "LABEL-0", "LABEL-1", and "LABEL-2", a mapping funcion was applied to change them back to "NEGATIVE," "NEUTRAL," and "POSITIVE." After
relabeling the names, the records were added to the previous training set and created a combined dataset ready for re-training. Upon training, we had the following metrics for our
sentiment classifier:

![image](https://user-images.githubusercontent.com/115701510/196179575-4954d043-1059-42ec-910b-596edf82019c.png)

Notably, the results presented are worse than expected compared to the original untuned classifier (distill-bert). Given that initially were only two labels, it was easier to get true
positives, resulting in a better F1 score. However, due to the new third label, it became easier for the model to misclassify a word or sentence, hence why the F1 score could have decreased.
Furthemore, BookCorpus, a dataset consisting of 11,038 unpublished books and English Wikipedia (excluding lists, tables, and headers) was used to train the untuned classifier.
Given that the model is not used to Reddit language and slang, it is coherent that the results present a lower F1 Score.

So after retraining the retuned model, was saved as final_car_sentiment_classifier_tuned_for_operation. Importantly, this model returned the
principal value of the sentence sentiment analysis. For example, each sentence gave back values such as 80% positive, 95% neutral, 60% negative, 
and so on, so summing the sentence's sentiment, was possible to calculate the sentiment of the entire comment. 
For example, if a comment has nine sentences, the sentiment result for each independent sentence could be NEGATIVE, POSITIVE, NEUTRAL,
NEUTRAL, NEUTRAL, POSITIVE, NEGATIVE, NEGATIVE, NEUTRAL. 
Then, counting positives as +1, negatives as -1, and neutrals as 0 the final result would have a total of -1.

As the last step, after getting the final scores of each comment, was linked the comments with its post title.
Why? Each post title has a lot of comments, so it is needed to sum the sentiment result of each of its comments to get the final overall score of the full post

## Word Cloud:

After developing both the Sentiment Analysis and tokenization, each token was linked to a sentiment, first to the comment and then to the token itself. Here, since the goal was
finding the sentiment of each word/token related to future queries, initially was developed a code that selected all tokens present in the same query, then added all of the positives to a list, and if they were negative to subtract from the list. 
Importantly, if a token is present in both lists, it would readjust its sentiment by summing or subtracting between them. For example,
if one queries "Honda," the code checks all the comments/tokens where Honda appears and first checks comments with total sentiment > 0. Suppose the related token Toyota within the
positive comments has a sentiment of 100 but also has a -10 within the comments with total sentiment < 0. Its final sentiment related to Honda had to be readjusted to 90. Hence, that is
what appears in the word cloud: a Toyota with a value of 90 in green.

![image](https://user-images.githubusercontent.com/115701510/196184165-f1e7961e-ebca-4111-b572-661073e32d1f.png)

## Streamlit:

Once everything is completed, using streamlit, the team was able to build a tool where each token could be check in a smooth way.
To run a streamlit script (as the attached .py) it is necessary to follow first the following tutorial

https://docs.streamlit.io/knowledge-base/using-streamlit/how-do-i-run-my-streamlit-script


## Analysis and Recommendations:

### CarMax

#### Analysis:

Were found four main insights in regards to the used-car industry:

**1. To each their own:** Each car brand and car type have a distinct perception among Reddit users.

**2. Location Matters:** Sentiment for particular cars differs by state and region.

**3. Monday Blues:** Salespeople are perceived negatively, with Monday being the worst day of the week to shop for cars.

**4. First Car:** When shopping for the first car with a parent or guardian, perceptions regarding which car to buy varies by parent.

#### Recommendations:


**1. CarMax should update its inventory to include more Hondas and Toyotas.**
These brands are the most popular among Reddit users, yet they represent less than half of the top-selling inventory for CarMax. Updating their inventory could yield more sales.

**2. CarMax should update its inventory by the geographic hub.** 
Each main CarMax based should review their inventory taking into account what users talk about in those areas. For example, inncluding more Hybrids in Southern California, focusing on trucks 
and SUVs in Texas, selling models with positive weather ratings, such as Honda SUVs, Subarus, and Toyota in Virginia, and providing an enhanced customer experience in Georgia given their
main competitor is currently slacking on the state.

**3. Retrain sales representatives and update digital platforms.** Users distrust used car salespeople. Hence, it is recommended that CarMax should retrain sales
representatives to be more knowledgeable and update their digital platforms to allow for more transactions to be made online.

### Rubrix:

#### Benefits:

Retraining: Given that the project had to retrain a hugging face model trained with academic
books, Rubrix's technological affordability was exceptional relabelling data to then retrain back on the SST2 model and garner better results with ease.

Improving the model accuracy: when used with the initial two standard labels of NEGATIVE AND POSITIVE, Rubrix improved the model accuracy by changing from an F1 of
0.75 to 0.81.

#### Challenging areas:

Coding Language: Given Rubrix has its own coding language, it was more often than not challenging to process without assistance.

No Collaboration: Given that Rubrix does not have the current capability of being used in collaboration with other users without a cloud, the burden of relabeling thousands of
words might fall on one individual.

**In conclusion, any company would benefit from using Rubrix in their Natural Language Processing pipelines, as it is user-friendly, practical, and allows for iterations and
retraining. As provided in the business case of CarMax, Rubrix did improve the model's accuracy, which fostered the insights presented in this business case. Moreso, if CarMax
wishes to re-run this model in a few months to identify new trends and buying patterns, there might also be plenty of potential to look into other aspects of the business.**
