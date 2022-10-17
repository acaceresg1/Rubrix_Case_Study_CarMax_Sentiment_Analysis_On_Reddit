import pandas as pd
from wordcloud import WordCloud, STOPWORDS, get_single_color_func
import math 
import collections
import matplotlib.pyplot as plt
import streamlit as st
import streamlit_wordcloud as wordcloud
import spacy
import matplotlib

from streamlit_autorefresh import st_autorefresh


@st.cache(allow_output_mutation=True)  # No need for TTL this time. It's static data :)
def get_data_by_state():
    df = pd.read_csv("posts_with_sentiment_score.csv")
    df['keywords'] = df['keywords'].apply(lambda x: list(eval(x)))
    return (df)

df = get_data_by_state()
df = df[0:5000].copy()



st.title("Try and test our reddit sentiment responder")

initial_word_selected = st.text_area("Please introduce the words you want to find", placeholder = "Honda Civic")
@st.cache(allow_output_mutation=True)
def words_selected(initial_word_selected):
	token_words_selected = []
	words_selected = initial_word_selected.replace(",","")
	words_selected = words_selected.lower()
	nlp = spacy.blank("en")
	doc = nlp(words_selected)
	token_words_selected = [token.text for token in doc]
	return(token_words_selected)

token_words_selected = words_selected(initial_word_selected)
st.write('You selected:', token_words_selected) 

@st.cache(allow_output_mutation=True)
def sentiment_per_keyword(dataframe):
    # lists to store the keywords in
    positive_keywords = []
    negative_keywords = []
    neutral_keywords = []

    # empty string to later store keywors for the wordcloud
    sentiment_adjusted_keywords = ' '
    negative_adjusted_keywords = ' '
    positive_adjusted_keywords = ' '

    # storing keywords 
    for words, sentiment_score in zip(dataframe.keywords, dataframe.sentiment_score):

        if sentiment_score == 0:
            neutral_keywords.extend(words)

        elif sentiment_score > 0:
            positive_keywords.extend(words * math.ceil(sentiment_score))

        elif sentiment_score < 0:
            negative_keywords.extend(words * abs(math.floor(sentiment_score)))
    
    
    dict_positive = collections.Counter(positive_keywords)
    dict_negative = collections.Counter(negative_keywords)

    positive = []
    negative = []
    
    ### for fred to find most positive and negative words
    # most_positive = []
    # most_negative = []
    ###
    
    all_words = set(positive_keywords + negative_keywords + neutral_keywords)
    
    for word in all_words:

        score = 0

        score -= dict_negative.get(word, 0)
        score += dict_positive.get(word, 0)

        if score == 0:
            sentiment_adjusted_keywords += " ".join([word])+" "

        elif score > 0:
            positive.append(word)
            
            ### for fred to find most positive and negative words
            # most_positive.extend(words * math.ceil(score))
            ###
            
            positive_adjusted_keywords += " ".join([word] * math.ceil(score))+" "
            sentiment_adjusted_keywords += " ".join([word] * math.ceil(score))+" "

        elif score < 0:
            negative.append(word)
            
            ### for fred to find most positive and negative words
            # most_negative.extend(words * abs(math.floor(score*-1)))
            ### 
            
            negative_adjusted_keywords += " ".join([word] * abs(math.floor(score*-1)))+" "
            sentiment_adjusted_keywords += " ".join([word] * abs(math.floor(score*-1)))+" "
    
    ### for fred to find most positive and negative words
    # most_positive 
    # most_negative 
    ###

    return negative_adjusted_keywords, positive_adjusted_keywords, sentiment_adjusted_keywords, positive, negative

@st.cache(allow_output_mutation=True)
def find_query_in_keywords(keywords, query):
    
    if all(word in keywords for word in query):
        
        # removing query from keywords -  not sure if this makes it any better
        #keywords = [item for item in keywords if item not in query]
        
        return keywords
    
    else:
        return float('nan')

query_df = df.copy()
query_df['keywords'] = query_df['keywords'].apply(lambda keywords: find_query_in_keywords(keywords, token_words_selected))
query_df = query_df[query_df['keywords'].notna()]
st.write(f"There are {query_df.shape[0]} results to your query")                                                   
st.write(query_df.head())

@st.cache(allow_output_mutation=True)
def wordcloud_negative(dataframe):    
    negative_adjusted_keywords, positive_adjusted_keywords, sentiment_adjusted_keywords, positive, negative = sentiment_per_keyword(dataframe)
    nlp = spacy.load('en_core_web_sm')   
    for sentiment_keywords in [negative_adjusted_keywords]:
        nlp.max_length = 1500000
        doc = nlp(sentiment_keywords)
        tokenizer = [token.text for token in doc]
        stopwords = [' \\n', '\\n', ' ','’', 'good', 'really', 'look', 'get', 'work', 'seem','like', 'car', 'guy', 'one','imo',
                'great', 'need', 'first', 'also', 'much', 'buy', 'drive', 'get', 'new', 'price', 'feature',
                'would', 'love', 'something', 'make', 'mile', 'long', 'year', 'want', 'market',
                'use', 'fun', 'take', 'cost', 'thank', 'deal', 'know', 'go', 'budget', 'well', 'live', 'model',
                'descent', 'advice', 'problem', 'break', 'time', 'fit', 'think',
                'preferably', 'reliable', 'maintenance', 'around', 'm', 's', 'transmission', 'driving', 'old',
                'right', 'feel', 'seat', '11', 'many', 'concern', 'come', 'keep', 'many', 'opinion', 'help', 'nothing',
                'cari', 'high', 'find', 'try', 'thing', 'across', 'accident', 'daily', 'replace', 'back', 'see', 'credit',
                'trade', 'possible', 'people', 'system', 'upgrade', 've', 'last', 'nice', 'trip', 'everything', 'example',
                 'etc', 'insight', 'even', 'decide', 'curious', 'ask', 'low', 'hard', 'seller', 'dealer', 'early', 'though',
                 'consider', 'due', 'job', 'private', 'life', 'review', 'issue', 'could', 'size', 'currently', 'toward',
                 'school', 'offer', 'read', 'two', 'month', 'show', 'regular', 'advance', 'si', 'fairly','reason',
                 'pretty', 'll', 'grand', 'run', 'con', 'set', 'area', 'fusion', 'especially', 'two', 'everyone',
                 'wife', 'narrow', 'likely', 'every', 'since', 'ga', 'give', 'wagon', 'leave', 'lot', 'prefer', 'experience',
                 'put', 'spend', 'dealership', 'somewhat', 'hp', 'ago', 'either', 'suggestion', 'okay', 'input', 'another', 'amount',
                 'anything', 'auto', 'reddit', 'fix', 'purchase', 'overall', 'edit', 'bad', 'towards',
                 'value', 'fine', 'major', 'welcome', 'yet', 'college', 'type', 'wonder', 'decent', 'option', 'hope',
                 'note', 'tear', 'half', 'less', 'say', 'lean', 'learn', 'city', 'eye', 'easy', 'sometime', 'line', 'follow',
                 'mostly', 'fan', 'plenty', 'payment', 'town', 'inside', 'tax', 'question', 'brand', 'dollar', 'probably', 
                 'view', 'let', 'basically', 'absolutely', 'home', 'cash', 'anyway', 'trouble', 'afford', 'ok', 'day',
                 'bring', 'avoid', 'weather', 'hour', 'recommendation', 'else', 'buyer', 'cause', 'occasional', 'limit',
                 'guess', 'cruise', 'parent', 'project', 'test', 'minor', 'thought', 'serie', 'student', 'plan', 'control',
                 'crazy', 'total', 'wheel', 'whatever', 'rather', 'series', 'stuff', 'someone', 'hopefully','true', 'fuel',
                 'may', 'negotiate', 'difference', 'current', 'se' 'end', 'almost', 'thinking', 'anyone', 'appreciate',
                 'title', 'tall', 'excellent', 'second', 'start', 'family', 'different', 'possibly', 'save',
                 'd', 'free', 'talk', 'extremely', 'mainly', 'awesome', 'choice', 'requirement', 'recommend', 'state',
                 'reasonable', 'graduate', 'far', 'expect', 'relatively', 'part', 'soon', 'door', 'next', 'worth', 'bit',
                 'end', 'road', 'maybe', 'interest', 'kind', 'must', 'hear',  'search', 'sure', 'driver', 'money',
                 'little', 'vehicle', 'enough', 'however', 'mind', 'still', 'actually', 'list', 'clearance', 'way', 'sit',
                 'enjoy', 'backup', 'couple', 'week', 'anybody', 'tomorrow', 'termn', 'pay', 'easily', 'choose', 'base',
                 'point', 'recently', 'buying', 'whether', 'sub', 'tech', 'colorado', 'friend', 'specifically', 'never',
                 'craigslist', 'intend', 'euro', 'past', 'website', 'idea', 'factor', 'priority', 'general', 'sell', 
                 'ideally','interested', 'ton', 'person', 'open', 'trim', 'already', 'miss', 'require', 'light', 'term', 'willing',
                 'post', 'economy', 'stay', 'passenger', 'defenitely', 'compare', 'suggest', 'maintain', 'front', 'turn', 'worry',
                 'ever', 'decision', 'least', 'handel', 'able', 'mean', 'ill', 'availiable', 'important', 'shop', 'often', 'similar',
                 'although', 'hate', 'quite', 'might'
                ]
    
    final_tokens = [x for x in tokenizer if x not in list(stopwords)]

    dicti = []
    for token in final_tokens:
        count_times = final_tokens.count(token)
        case = {'text': token, 'value': count_times}
        dicti.append(case)
        
    return (dicti)

@st.cache(allow_output_mutation=True)
def wordcloud_positive(dataframe):    
    negative_adjusted_keywords, positive_adjusted_keywords, sentiment_adjusted_keywords, positive, negative = sentiment_per_keyword(dataframe)

    nlp = spacy.load('en_core_web_sm')   

    for sentiment_keywords in [positive_adjusted_keywords]:
        nlp.max_length = 1500000
        doc = nlp(sentiment_keywords)
        tokenizer = [token.text for token in doc]
        stopwords = [' \\n', '\\n', ' ','’', 'good', 'really', 'look', 'get', 'work', 'seem','like', 'car', 'guy', 'one','imo',
                'great', 'need', 'first', 'also', 'much', 'buy', 'drive', 'get', 'new', 'price', 'feature',
                'would', 'love', 'something', 'make', 'mile', 'long', 'year', 'want', 'market',
                'use', 'fun', 'take', 'cost', 'thank', 'deal', 'know', 'go', 'budget', 'well', 'live', 'model',
                'descent', 'advice', 'problem', 'break', 'time', 'fit', 'think',
                'preferably', 'reliable', 'maintenance', 'around', 'm', 's', 'transmission', 'driving', 'old',
                'right', 'feel', 'seat', '11', 'many', 'concern', 'come', 'keep', 'many', 'opinion', 'help', 'nothing',
                'cari', 'high', 'find', 'try', 'thing', 'across', 'accident', 'daily', 'replace', 'back', 'see', 'credit',
                'trade', 'possible', 'people', 'system', 'upgrade', 've', 'last', 'nice', 'trip', 'everything', 'example',
                 'etc', 'insight', 'even', 'decide', 'curious', 'ask', 'low', 'hard', 'seller', 'dealer', 'early', 'though',
                 'consider', 'due', 'job', 'private', 'life', 'review', 'issue', 'could', 'size', 'currently', 'toward',
                 'school', 'offer', 'read', 'two', 'month', 'show', 'regular', 'advance', 'si', 'fairly','reason',
                 'pretty', 'll', 'grand', 'run', 'con', 'set', 'area', 'fusion', 'especially', 'two', 'everyone',
                 'wife', 'narrow', 'likely', 'every', 'since', 'ga', 'give', 'wagon', 'leave', 'lot', 'prefer', 'experience',
                 'put', 'spend', 'dealership', 'somewhat', 'hp', 'ago', 'either', 'suggestion', 'okay', 'input', 'another', 'amount',
                 'anything', 'auto', 'reddit', 'fix', 'purchase', 'overall', 'edit', 'bad', 'towards',
                 'value', 'fine', 'major', 'welcome', 'yet', 'college', 'type', 'wonder', 'decent', 'option', 'hope',
                 'note', 'tear', 'half', 'less', 'say', 'lean', 'learn', 'city', 'eye', 'easy', 'sometime', 'line', 'follow',
                 'mostly', 'fan', 'plenty', 'payment', 'town', 'inside', 'tax', 'question', 'brand', 'dollar', 'probably', 
                 'view', 'let', 'basically', 'absolutely', 'home', 'cash', 'anyway', 'trouble', 'afford', 'ok', 'day',
                 'bring', 'avoid', 'weather', 'hour', 'recommendation', 'else', 'buyer', 'cause', 'occasional', 'limit',
                 'guess', 'cruise', 'parent', 'project', 'test', 'minor', 'thought', 'serie', 'student', 'plan', 'control',
                 'crazy', 'total', 'wheel', 'whatever', 'rather', 'series', 'stuff', 'someone', 'hopefully','true', 'fuel',
                 'may', 'negotiate', 'difference', 'current', 'se' 'end', 'almost', 'thinking', 'anyone', 'appreciate',
                 'title', 'tall', 'excellent', 'second', 'start', 'family', 'different', 'possibly', 'save',
                 'd', 'free', 'talk', 'extremely', 'mainly', 'awesome', 'choice', 'requirement', 'recommend', 'state',
                 'reasonable', 'graduate', 'far', 'expect', 'relatively', 'part', 'soon', 'door', 'next', 'worth', 'bit',
                 'end', 'road', 'maybe', 'interest', 'kind', 'must', 'hear',  'search', 'sure', 'driver', 'money',
                 'little', 'vehicle', 'enough', 'however', 'mind', 'still', 'actually', 'list', 'clearance', 'way', 'sit',
                 'enjoy', 'backup', 'couple', 'week', 'anybody', 'tomorrow', 'termn', 'pay', 'easily', 'choose', 'base',
                 'point', 'recently', 'buying', 'whether', 'sub', 'tech', 'colorado', 'friend', 'specifically', 'never',
                 'craigslist', 'intend', 'euro', 'past', 'website', 'idea', 'factor', 'priority', 'general', 'sell', 
                 'ideally','interested', 'ton', 'person', 'open', 'trim', 'already', 'miss', 'require', 'light', 'term', 'willing',
                 'post', 'economy', 'stay', 'passenger', 'defenitely', 'compare', 'suggest', 'maintain', 'front', 'turn', 'worry',
                 'ever', 'decision', 'least', 'handel', 'able', 'mean', 'ill', 'availiable', 'important', 'shop', 'often', 'similar',
                 'although', 'hate', 'quite', 'might'
                ]
    
    final_tokens2 = [x for x in tokenizer if x not in list(stopwords)]

    dicti2 = []
    for token2 in final_tokens2:
        count_times = final_tokens2.count(token2)
        case2 = {'text': token2, 'value': count_times}
        dicti2.append(case2)
        
    return (dicti2)

@st.cache(allow_output_mutation=True)		
def unique(list1):
    # initialize a null list
    unique_list = []
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return(unique_list)

@st.cache(allow_output_mutation=True)
def myFunc(value):
    return value.get('value')

@st.cache(allow_output_mutation=True)
def rep_wordcloud_negative(unique_positive_list):
    import streamlit_wordcloud as wordcloud
    autumn = matplotlib.cm.get_cmap(name='autumn', lut=None)
    words = unique_positive_list

    return(object1)

@st.cache(allow_output_mutation=True)
def rep_wordcloud_positive(unique_positive_list2):
    import streamlit_wordcloud as wordcloud
    summer = matplotlib.cm.get_cmap(name='summer', lut=None)
    words2 = unique_positive_list2[0:100]
    object2 = wordcloud(words2, per_word_coloring=False, palette = summer)
    return(object2)

autumn = matplotlib.cm.get_cmap(name='autumn', lut=None)
summer = matplotlib.cm.get_cmap(name='summer', lut=None)


def dif_wordcloud(query_df):
	dicti2 = wordcloud_positive(query_df)
	dicti = wordcloud_negative(query_df)      
	unique_positive_list2 = unique(dicti2)
	unique_positive_list = unique(dicti)
	
	unique_positive_list.sort(key=myFunc, reverse=True)
	words=unique_positive_list[0:75]
	returnobj1 = wordcloud.visualize(words, per_word_coloring=False, palette = autumn)

	unique_positive_list2.sort(key=myFunc, reverse=True)
	words2 = unique_positive_list2[0:75]
	returnobj2 = wordcloud.visualize(words2, per_word_coloring=False, palette = summer)
	
	return(returnobj1, returnobj2)

if len(query_df != 0):
	dif_wordcloud(query_df)
	