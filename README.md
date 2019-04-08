# Detecting Emotions and Topic Modeling in Elections Related Suspended Twitter Accounts Allegedly Originated from Russia. 


|![twitter snippet](/images/twittersnippet.png)| ![twitter logo](/images/logo_twitter.png)|
|----------------------------------------------|------------------------------------------|




## Introduction and Project Goals
  Creating healthy public conversation through data analysis and thorough __text analysis__ of suspended Twitter accounts originated from Russia using **Twitter's Elections Integrity Dataset** that was published in January 2019. <br />
  
  In line with Twitters’ principles of transparency and to improve public understanding of alleged foreign influence campaigns, Twitter is making publicly available archives of Tweets and media that it believes resulted from potentially state-backed information operations on its service.<br/>
    **Ivan Corneillet**, instructor at Twitter University and former instructor here at Galvanize brought to us recently released data set (_January 2019_) of suspended accounts originated from Russia. It's been a topic of my interest already and when Ivan brought it in I knew that this is exactly what I want to be using for my Capstone Project. <br/>

## Problem: 

  **Engineering Public Opinion** is a significant concern to the public, officials and every one, really. It is on the forefront of any modern political and/or money divide. It is the problem of our time and Social networks have come under fire for their inability to prevent the manipulation of news and information by potentially malicious actors. <br/>
<p align="left"> 
<img src=/images/data.png width="75"/>
</p>

* [Twitter Election Integrity Data Set, Russia](https://about.twitter.com/en_us/values/elections-integrity.html#data) include all public, non deleted Tweets from accounts Twitter believes are connected to state-backed information operations. Tweets deleted by these users prior to their suspension (which are not included in this dataset) comprise less than 1% of their overall activity. <br/> 
* Dataset consisted of 416 accounts allegedly originated in Russia and their tweet content. <br/>

#### Data Set Features and Null Values (blue). ####
<p align="left"> 
<img src=/images/isnull.png width="500"/>
</p>

## Libraries and Top Tools Used:

`Pandas, numpy`  - Data Handling, Linear ALgebra <br/>
`Gensim, gensim corpora, gensim Coherence Model, Sklearn` - Topic Modeling <br/>
`Sklearn CountVectorizer` <br/>
`Matplotlib, Seaborn, pyLDAvis, WordCloud` - Visualizations <br/>
`NLTK, nltk.stem.wordnet.WordNetLemmatizer, nltk.stem.snowball.SnowballStemmer` - Text preprocessing using NLP <br/>
Python Regual expressions library, twitter text preprocessor <br/>
`Vader` - Twitter Sentiment <br/>
 
Target variable:  **Natural Language Components of tweets text**, it’s similarity and emotion weight of tweets. With this in mind I started with looking into the texts tweets. <br/>

**1. First Thing: Tweets or Retweets?** <br/>

**171,959.0** number of **tweets** 143,308.0 unique tweets , 28,651.0 repeated tweets <br/>
**59,3287.0** number of **retweets**, 524,243.0 unique retweets, 69,044.00 repeated retweets <br/>
The subject are the tweets.
For this project subject are the tweets only. (_I'm planning to continue my research on the Retweet subset as well_). <br/>
**2. Languages**.  English was selected as a language of this research given language proportion into consideration. Final English Only Subset of tweets had 333 accounts<br/>
Account_Language: <br/> 

	  162342 -- English <br/>
	  8125   -- French <br/>
	  1237   -- Russian <br/>
	  185    -- Indonesian <br/>
	  53     -- Turkish <br/>
	  17     -- Romanian <br/>
					
**ADD SEABORN COUNT PLOTS HERE

**3. User Reported Location** 
Tweets Volume per Location.
`loc_df.groupby('user_reported_location').count()['userid']` _pandas command to count users per location_
`clean_user_reportd_location(loc_df)` _cleaning and aggregating locations function in `src` folder_. 
I wrote a funciton to convert user reported locations to latitude-longitude of the city. Because I love geography I handcoded the latitudes and longitudes from the cities I needed. I used geopandas and matplotlib for the map plot. Below, The Volume of the tweets from each location. 

`import descartes` <br/>  
`import geopandas as gpd`  <br/>
`from shapely.geometry import Point, Polygon` <br/>

Clean Up Cities/Countries `loc_clean.py` in `src` 
<p align="center"> 
<img src=/plots/user_location.png width="1114"/>
</p>


## Text Pipeline and NLP
<p align="center"> 
<img src=/images/pipeline_flowchart.png width="500"/>
</p>

1.1.Words were lemmatized, stemmed. Punctuation removed. 
In linguistic morphology and information retrieval in **lemmatization** we are removing word endings to get to our target, the base or dictionary form of a word.  <br/>
Kittens - kitten, better - good, walking  - walk. <br/>

1.2. Stemming, the process of reducing inflected (or sometimes derived) words to their word stem, base or root form: <br/>
cats, catlike, and catty, cat ---> cat <br/>

`lemmer = WordNetLemmatizer()` <br/>

`stemmer = SnowballStemmer('english')` <br/>

1.3. Stop words. Noise. <br/>
Standard stop words library from nltk was used. <br/>
`stop_words = set(nltk.corpus.stopwords.words('english'))` <br/>

1.4 In addition the least meaningful words were arbitrary removed by the author using [**google trends**](https://trends.google.com/trends) and human comprehension. 
additional stopwords list (_developing_) in add_stop_w.py

1.5. Emojies, Urls, Hashtags and Mentions were out of scope of this research and removed from text using [Twitter text preprocessor](https://pypi.org/project/tweet-preprocessor/): </br>
`pip instal tweet-preprocessor`

## BAG OF WORDS
<p align="center"> 
<img src=/plots/bow_cloud.png width="888"/>
</p>

<p align="center"> 
<img src=/images/em_wheel.png width="888"/>
</p>


Most schools of thought can confirm: Emotion is often the driving force behind motivation, positive or negative as well as the ability of words represent emotional experience[1]. 


### Methodology. 

Basic Emotions: During the 1970s, psychologist Paul Eckman identified six basic emotions that he suggested were universally experienced in all human cultures. The emotions he identified were happiness, sadness, disgust, fear, surprise, and anger. 

**Emotions Detection** is an interesting blend of **Psychology** and **Technology**. As much as sentiment analyses is widely used nowadays, I wanted to have a slightly larger emotional palette rather than classic polarity analysis. 

**Tool Kit**
[Emotions Sensor Data Set available on Kaggle](https://www.kaggle.com/iwilldoit/emotions-sensor-data-set). Data set contains over 21000 unique English words classified statistically into of 7 basic emotions: 
### Disgust, Surprise, Neutral, Anger, Sad, Happy and Fear. 
By the authors of the dataset words have been manually and automatically labeled using _Andbrain_(published on Kaggle) engine from over 1.185.540 classified words, blogs, tweets and sentences. 

Using NLTK Vectorize tweets corpus over vocabulary (hyper-parameter of CountVectorizer)that is set of unique words with calculated emotions weight per word. 
`from sklearn.feature_extraction.text import CountVectorizer`
Comparing index and columns of two dataframse to make sure they are identical. But it is nice to check! 
Linear Algebra Magic:
<p align="left"> 
<img src=/images/linalg.png width="243">
</p>

## RESULTS

<p align="center"> 
<img src=/plots/timeseries.png width="1114">
</p>


## Results Discussion:
* Top interest. In tweets emotion detection time series observed **fear increased** in 2014 - 2016 right before the election. It is a very interesting observation which allows us to examine fear patterns available for research during that period of time to see if it’s noise or not and potentially opens up an important feature in decision making classifiers. 

* Quantitatively detecting emotions in social media messages can be instrumental in creating models that can classify emotional patterns of suspected accounts for controlling and managing healthy social conversation and possible ethical implications of trying to influence public opinion.  And therefore this research front should be further developed and improved upon. 


## Topic Modeling using LDA 
Methodology
[Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) is a generative statistical model of the joint probability distribution on _X × Y, P ( X , Y )_ that allows sets of observations to be explained by unobserved groups that explain why some parts of the data are similar and what are the similarities, what are the topics, the nodes of our text,  or  the abstract "topics" that occur in a collection of documents allowing us for discovery of _hidden semantic structures in a text body_ which is the goal of this project. <br/>

Particular words will be found in the document more or less frequently: "surf" and "ocean" will appear more often in documents about surfing, "smoked salmon" and "dill" will appear in documents about scandinavian recipes than others, and many common words that will appear equally in all topics. <br/>
A document typically concerns multiple topics in different proportions; thus, in a document that is 80% about surfing and 20% about scandinavian recipes, there would probably be about 8 times more words acount ocean and surf than about scandinavian recipes. The "topics" produced by topic modeling techniques are clusters of similar words. A topic model captures this intuition in a mathematical framework, which allows examining a set of documents and discovering, based on the statistics of the words in each, what the topics might be and what each document's balance of topics is. 

|![annot](/images/annot_lda.png)| ![lda](/images/lda_plate_notation.png)|
|----------------------------------------------|------------------------------------------|


**LDA is an unsupervised technique**, meaning that we don’t know prior to running the model **how many topics** exits in our corpus. **Coherence score** is a metric and main technique used to estimate the number of topics and to measure human understandability and interpretability.

## Model Evaluation. Coherence Score. 

A set of statements or facts is said to be coherent, if they support each other.
Interest into coherence measures has arisen in text mining, as unsupervised learning methods like topic models give no guarantees on the interpretability of their output. Topic models learn topics—typically represented as sets of important words—automatically from unlabeled documents in an unsupervised way. This is an attractive method to bring structure to otherwise unstructured text data. [needs reference]

## ADD math on coherence score. <br/> 
`from gensim import code models.coherencemodel `  Calculation and gensim Hypeparameter ‘U_mass’ <br/> 
 <br/>
 Math (link)
![Coherence](plots/coherence_scores.png)

## RESULTS
Topics Model Can be seen following this link: **html link to oprn pyLDAvis: LINK        
By Examining the clusters we can identify and suggest the latent groups discovered: We can see these main Clusters: 

1. Agression. Race. Terror. Disgust. <br/> 
2. Scandals. Politics. Ashley Madison. Topless. Donald. <br/> And other -- make a cloud
3. America Hate Shumer <br/>
The first 3 topics are of primery interest since they are talking about American politics which is interesting find and can be considered as a feature. 

4. Zimbabwe. <br/> WIll try to detect non english words. 
5. European tennis. Sports. <br/>
6. Sirya <br/>
7. Turkish, azeri. <br/>


## Future Steps: 

* Topic distribution across documents. Find the most representative documents for each topic 
* Network and Graph analysis can be applied to further determine connections between accounts and their social weights in the public conversation. 
* I wanted to investigate further where the tweets were originated from geographically: During next election create a classifier model based on this dataset to label based on previous trends and most interesting features developed based on LDA Model topics clusters.
* AI-powered keyword equipped text screener. 
* Creating a AI Tool for Emotion Analysis and Understanding.
 


## Conclusions

Inspired by the findings of this eda to continue searching for latent features available to be discovered using modern machine learning techniques and tools.
## Acknowledgements

## About me 

[**My LinkedIn**](https://www.linkedin.com/in/anastasia-simpson-61402629/) 
## References
1.  Gaulin, Steven J.C. and Donald H. McBurney. Evolutionary Psychology. Prentice Hall. 2003. ISBN 978-0-13-111529-3, Chapter 6, p 121-142.
2. 
