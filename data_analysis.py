from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

def getStr(data):
    """DF TO COMBINED STRING"""
    s = ""
    for i in data.text:
        s = s + " " + i
    return s

# def cleanStr(string):
#     string = string.lower()
#     string = string.replace("https", "")
#     string = string.replace("http", "")
#     string = string.replace("<br>", "")
#     string = string.replace("<br />", "")
#     string = string.replace("</br>", "")
#     return string

def dataframe_to_wordcloud(name, data):
    s = getStr(data)
    return type(s)
    # s = cleanStr(s)
    # stopwords = set(STOPWORDS)
    # wordcloud = WordCloud(width=1000, height=1000, stopwords=stopwords, background_color="white").generate(s)
    # plt.figure( figsize=(20,20))
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis("off")
    # plt.savefig(f"{name}.png")

