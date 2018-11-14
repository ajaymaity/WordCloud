#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Start with loading all necessary libraries
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# Load in the dataframe
df = pd.read_csv("data/winemag-data-130k-v2.csv", index_col=0)


# In[5]:


# Looking at first 5 rows of the dataset
df.head()


# In[7]:


df.shape


# In[8]:


print("There are {} observations and {} features in this dataset. \n".format(df.shape[0],df.shape[1]))

print("There are {} types of wine in this dataset such as {}... \n".format(len(df.variety.unique()),
                                                                           ", ".join(df.variety.unique()[0:5])))

print("There are {} countries producing wine in this dataset such as {}... \n".format(len(df.country.unique()),
                                                                                      ", ".join(df.country.unique()[0:5])))


# In[9]:


df[["country", "description","points"]].head()


# In[10]:


# Groupby by country
country = df.groupby("country")

# Summary statistic of all countries
country.describe().head()


# In[11]:


country.mean().sort_values(by="points",ascending=False).head()


# In[12]:


plt.figure(figsize=(15,10))
country.size().sort_values(ascending=False).plot.bar()
plt.xticks(rotation=50)
plt.xlabel("Country of Origin")
plt.ylabel("Number of Wines")
plt.show()


# In[13]:


plt.figure(figsize=(15,10))
country.max().sort_values(by="points",ascending=False)["points"].plot.bar()
plt.xticks(rotation=50)
plt.xlabel("Country of Origin")
plt.ylabel("Highest point of Wines")
plt.show()


# In[14]:


get_ipython().run_line_magic('pinfo', 'WordCloud')


# In[15]:


# Start with one review:
text = df.description[0]

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[16]:


# lower max_font_size, change the maximum number of word and lighten the background:
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[18]:


# Save the image in the img folder:
wordcloud.to_file("img/first_review.png")


# In[19]:


text = " ".join(review for review in df.description)
print ("There are {} words in the combination of all review.".format(len(text)))


# In[20]:


# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(["drink", "now", "wine", "flavor", "flavors"])

# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[23]:


wine_mask = np.array(Image.open("data/wine_mask_xgk1tq.png"))
wine_mask


# In[24]:


def transform_format(val):
    if val == 0:
        return 255
    else:
        return val


# In[25]:


# Transform your mask into a new one that will work with the function:
transformed_wine_mask = np.ndarray((wine_mask.shape[0],wine_mask.shape[1]), np.int32)

for i in range(len(wine_mask)):
    transformed_wine_mask[i] = list(map(transform_format, wine_mask[i]))


# In[26]:


# Check the expected result of your mask
transformed_wine_mask


# In[27]:


# Create a word cloud image
wc = WordCloud(background_color="white", max_words=1000, mask=transformed_wine_mask,
               stopwords=stopwords, contour_width=3, contour_color='firebrick')

# Generate a wordcloud
wc.generate(text)

# store to file
wc.to_file("img/wine.png")

# show
plt.figure(figsize=[20,10])
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

