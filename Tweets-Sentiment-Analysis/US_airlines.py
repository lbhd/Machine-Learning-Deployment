import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def main():

    st.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color: #1335cf;
           
        }
    </style>
    """, unsafe_allow_html=True)


    st.title("Sentiment Analysis on Tweets about US Airlines ✈️")
    st.sidebar.title("Sentiment Analysis on Tweets about US Airlines ✈️")
    st.sidebar.subheader("By [Andrew Liang](https://github.com/lbhd)")
    st.markdown("""---""")
    
    st.sidebar.markdown("[![ISSUE](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/lbhd/Machine-Learning-Deployment/tree/main/Tweets-Sentiment-Analysis)")
    st.subheader("A data-driven customer sentiment analysis web application on major U.S. airlines"
                 )
    st.sidebar.markdown("An exploratory sentiment analysis job based on tweets from U.S. airlines")

    @st.cache(persist=True)
    def load_data():
        url = 'https://raw.githubusercontent.com/lbhd/Machine-Learning-Deployment/main/Tweets-Sentiment-Analysis/Tweets.csv'
        data = pd.read_csv(url, index_col=0)
        data["tweet_created"] = pd.to_datetime(data["tweet_created"])
        return data

    data = load_data()


    st.sidebar.markdown("### Number of tweets by sentiment")
    #key : An optional string or integer to use as the unique key for the widget
    select = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='1')
    sentiment_count = data['airline_sentiment'].value_counts()
    sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, 'Tweets':sentiment_count.values})

    #x = data.sentiment.value_counts().sort_values()
    ### 1. profiling of tweets sentiments
    if not st.sidebar.checkbox("Hide", False):
        st.markdown("### Number of tweets by sentiment")
        if select == 'Bar plot':
            
            fig = px.bar(sentiment_count, x='Sentiment', y='Tweets', color='Tweets', height=500, text_auto='.2s')
            st.plotly_chart(fig)
        else:
            fig = go.Figure(data=[go.Pie(labels=sentiment_count.Sentiment, values=sentiment_count.Tweets, pull=[0.01, 0.01, 0.01])])
            fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=19,
                  marker=dict(line=dict(color='#000000', width=2)))

            st.plotly_chart(fig)
        
    ### 2. Tweet locations based on time of day
    st.sidebar.subheader("Tweet Locations Based on Time of Day")
    hour = st.sidebar.slider("Hour of Day", 0, 23)
    selected_data = data[data["tweet_created"].dt.hour == hour]
    if not st.sidebar.checkbox("Hide", False, key="2"):
        st.subheader("Distribution of Tweets by Time of Day")
        st.markdown(f"There were {len(selected_data)} tweets between {hour}:00 and {(hour + 1) % 24}:00")
        st.map(selected_data)

    ### 3. Breakdown airline tweets by sentiment
    st.sidebar.subheader("Tweets Sentiment by Airlines")
    choice = st.sidebar.multiselect("Select Airline(s)", tuple(pd.unique(data["airline"])),key=3)
    if not st.sidebar.checkbox("Hide", True, key=31):
        if len(choice) > 0:

            chosen_data = data[data["airline"].isin(choice)]

            fig = px.histogram(chosen_data, x="airline", y="airline_sentiment",
                                 histfunc="count", color="airline_sentiment",
                                 facet_col="airline_sentiment",facet_col_spacing=0.04,text_auto=True,width=800,height=500,title='Tweets Sentiment Distribution by Airlines')


        
            st.plotly_chart(fig)


    ### 4. Reasons for Negative Tweets by Airlines
    st.sidebar.subheader("Reasons for Negativity Across Airlines")
    df_neg = data[data.airline_sentiment == 'negative']
    negs = st.sidebar.multiselect("Select Airline(s)", tuple(pd.unique(df_neg["airline"])),key=4)
    if not st.sidebar.checkbox("Hide", True,key=41):
        if len(negs) > 0:

            chosen_data = df_neg[df_neg["airline"].isin(negs)]

            #fig = px.histogram(chosen_data2.dropna(), x="airline", y="negativereason",
                             #    histfunc="count", color="negativereason",
                           #      facet_col="negativereason",facet_col_spacing=0.04,text_auto=True,width=800,height=500)

            fig = px.histogram(chosen_data, x='airline', color="negativereason", barmode='group',facet_col_spacing=0.04,text_auto=True,width=900,height=650,title='Reasons for Negativity Across Airlines')
        
            st.plotly_chart(fig)


    import re
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    from nltk.tokenize import word_tokenize
    nltk.download('punkt')
    def clean_text(d):
        pattern = r'[^a-zA-Z\s]'
        text = re.sub(pattern, '', d)
        return text

    names = ['delta', 'deltaair', 'united', 'unitedair', 'southwest', 'southwestair', 'usairways',
            'virginamerica', 'american', 'americanair', 'jetblue', 'jetblues', 'usairway',
            'flight', 'airline', 'airlines']
    def clean_stopword(d):
        stop_words = stopwords.words('english')
        for name in names:
            stop_words.append(name)
        return " ".join([w.lower() for w in d.split() if w.lower() not in stop_words and len(w) > 1])

    def tokenize(d):
        return word_tokenize(d)
    

  
    # st.sidebar.header("Word Cloud")
    # word_sentiment = st.sidebar.radio('Select sentiment type for word cloud', ('positive', 'neutral', 'negative'))
    # st.set_option('deprecation.showPyplotGlobalUse', False)
    # if not st.sidebar.checkbox("Hide", False, key=6):
    #     st.subheader('Word cloud for %s sentiment' % (word_sentiment))
    #     cleaned_data = data[data['airline_sentiment']==word_sentiment].text.apply(clean_text).apply(clean_stopword).apply(tokenize)
        
    #     cleaned_data = [" ".join(cleaned_data.values[i]) for i in range(len(cleaned_data))]
    #     cleaned_data = [" ".join(cleaned_data)][0]


       
    #     wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=650).generate(cleaned_data)
    #     plt.imshow(wordcloud)
    #     plt.xticks([])
    #     plt.yticks([])
    #     st.pyplot()


    word_sentiment = st.sidebar.radio('Select sentiment type for word cloud', ('positive', 'neutral', 'negative'))
    #st.set_option('deprecation.showPyplotGlobalUse', False)
    if not st.sidebar.checkbox("Hide", True, key=6):
        st.subheader('Word cloud for %s sentiment' % (word_sentiment))
        df = data[data['airline_sentiment']==word_sentiment]
        words = ' '.join(df['text'])
        processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
        wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=640).generate(processed_words)
        plt.imshow(wordcloud)
        plt.xticks([])
        plt.yticks([])
        st.pyplot()
            
 
    




if __name__ == "__main__":
    main()
