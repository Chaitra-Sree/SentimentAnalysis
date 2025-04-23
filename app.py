import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

tab1, tab2 = st.tabs(["Sentiment Analysis On Social Media Reviews on the brand, The Ordinary.", "Model Comparison"])

with tab1:
    st.markdown("<h1 style='text-align: center; color: #B9D9EB; font-family: monospace;'>Sentiment Analysis Dashboard</h1>", unsafe_allow_html=True)
    df = pd.read_excel("/content/drive/MyDrive/Colab Notebooks/vader_custom_scored_reviews.xlsx")

    df['vader_label'] = df['vader_label'].str.lower()

    sentiment_counts = df['vader_label'].value_counts()
    labels = sentiment_counts.index.tolist()

    fig = px.pie(
        names=labels,
        values=sentiment_counts.values,
        color=labels,
        color_discrete_map={
            'positive': '#ffabab',
            'neutral': '#0068c9',
            'negative': '#83c8ff'
        },
        title="Distribution of Sentiments in Reviews"
    )

    fig.update_traces(textinfo='percent+label')
    fig.update_layout(title_font_size=20)

    st.plotly_chart(fig)
    st.markdown("---")
    st.markdown("### Conclusion")
    st.write("""
    The sentiment analysis reveals a predominantly positive perception of the brand, with 73.5% of customer reviews expressing positive sentiment.
    While negative reviews account for 14.6% and neutral reviews 12.0%, the overwhelming positive feedback indicates **strong overall customer satisfaction.**

    Addressing the concerns raised in the negative reviews remains important for continuous improvement.
    """)
    st.markdown("---")




    # Platform vs sentiment
    platform_sentiment = df.groupby(['platform', 'vader_label']).size().unstack().fillna(0)
    st.write("### Sentiment Distribution by Platform")
    st.bar_chart(platform_sentiment)

    st.markdown("### Conclusion")
    st.write("""
    Across Facebook, Instagram, and Twitter, the sentiment towards the skincare brand shows a consistent pattern with the highest volume of positive sentiment,
    followed by neutral, and then negative sentiment having the lowest volume.
    Facebook and Twitter show a similar overall volume of mentions, while Instagram has a notably lower volume across all sentiment categories.
    """)
    st.markdown("---")




    # Time vs sentiment
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])

    df['month'] = df['timestamp'].dt.to_period('M').dt.to_timestamp()

    monthly_sentiment = df.groupby(['month', 'vader_label']).size().unstack().fillna(0)

    st.write("### Sentiment Over Time")
    st.line_chart(monthly_sentiment)
    st.markdown("### Conclusion")
    st.write("""
    The sentiment towards the skincare brand from **2018 to 2024** has been predominantly **positive**, with frequent noticeable peaks.
    **Negative sentiment** remains consistently low, showing only occasional, mild spikes.
    Meanwhile, **neutral sentiment** fluctuates modestly over time.
    Overall, this indicates a generally **favorable public perception** of the brand, with periods of increased engagement likely linked to campaigns, events, or product launches.
    """)
    st.markdown("---")





    # Discovery source vs sentiment
    discovery_sentiment = df.groupby(['discovery_source', 'vader_label']).size().unstack().fillna(0)
    discovery_sentiment = discovery_sentiment.loc[discovery_sentiment.sum(axis=1).sort_values(ascending=False).index]

    st.markdown("### Sentiment by Discovery Source")
    st.bar_chart(discovery_sentiment)

    st.markdown("### Conclusion")
    st.write("""
    Among all discovery sources, **TikTok**, **Search Engines**, and **Influencers** brought in the highest number of total reviews,
    with **TikTok** and **Influencers** showing a strong skew toward **positive sentiment**.

    Interestingly, sources like **Friends** and **In-store experiences** also led to a large number of reviews, but the sentiment distribution there was more mixed.

    This suggests that **social media platforms and influencer marketing** not only drive high engagement but also tend to result in
    **more favorable customer perceptions**.
    """)
    st.markdown("---")




    # Review Length by Sentiment
    st.markdown("### Review Length by Sentiment")
    df['review_length'] = df['processed_text'].apply(lambda x: len(str(x).split()))

    fig1 = px.box(df,
                  x='vader_label',
                  y='review_length',
                  color='vader_label',
                  color_discrete_map={
                      "Positive": "#8dd3c7",
                      "Neutral": "#ffffb3",
                      "Negative": "#fb8072"
                  },
                  title="How Review Length Varies with Sentiment")

    fig1.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title_font_size=18
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("### Conclusion")
    st.write("""
    Positive reviews tend to be longer and receive more likes, suggesting that detailed, enthusiastic feedback resonates well with others.
    Negative reviews, while shorter on average, also garner attention, possibly due to their emotional impact.
    Neutral reviews are typically brief and receive the least engagement.
    """)
    st.markdown("---")




    # Likes by Sentiment
    st.markdown("### Likes by Sentiment")

    fig2 = px.box(df,
                  x='vader_label',
                  y='likes',
                  color='vader_label',
                  color_discrete_map={
                      "Positive": "#8dd3c7",
                      "Neutral": "#ffffb3",
                      "Negative": "#fb8072"
                  },
                  title="How Likes Vary with Sentiment")

    fig2.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title_font_size=18
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("### Conclusion")
    st.write("""
    Content with strong sentiment (positive or negative) is generally more engaging and tends to receive more likes than neutral content.
    This reinforces the idea that emotional tone plays a significant role in social media engagement.
    """)
    st.markdown("---")



    st.markdown("### Most Liked Reviews by Sentiment")

    df = df.dropna(subset=['review_text', 'vader_label', 'likes'])
    df['likes'] = pd.to_numeric(df['likes'], errors='coerce')
    df = df.dropna(subset=['likes'])
    df['vader_label'] = df['vader_label'].str.capitalize()

    for sentiment in ['Positive', 'Neutral', 'Negative']:
        st.write(f"#### {sentiment} Reviews")
        top_reviews = df[df['vader_label'] == sentiment].sort_values(by='likes', ascending=False).head(3)
        st.dataframe(top_reviews[['review_text', 'likes', 'platform']])


    st.markdown("### Conclusion")
    st.write("""The most liked reviews across all sentiments are detailed and emotionally expressive.
    Positive reviews highlight product effectiveness, while top negative reviews are often tied to skin reactions.
      This suggests users resonate most with personal experiences that feel relatable or extreme.""")
    st.markdown("---")




    st.markdown("### Average Sentiment Score by Discovery Source")
    source_sentiment = df.groupby('discovery_source')['vader_polarity'].mean().sort_values(ascending=False).reset_index()

    fig = px.bar(
        source_sentiment,
        x='discovery_source',
        y='vader_polarity',
        title="Average Sentiment Score by Discovery Source",
        labels={'vader_polarity': 'Avg Sentiment Score', 'discovery_source': 'Discovery Source'},
        color='vader_polarity',
        color_continuous_scale='RdYlGn'
    )

    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig)
    st.markdown("### Conclusion")
    st.write("""Marketers and brands might want to focus more on Social Media, Reddit, and Search Engines to improve perception and engagement.
    Beauty Blogs and Ads may need reassessment or different strategies to improve their public perception.
    """)




with tab2:
    st.markdown("### Model Accuracy Comparison")
    st.write("This section compares the performance of different sentiment classification models:")

    # Create a small DataFrame of your accuracy scores
    accuracy_df = pd.DataFrame({
        "Model": ["VADER", "Naive Bayes", "SVM"],
        "Accuracy": [71.17, 60.87, 43.48]
    })

    # Build the Plotly bar chart
    fig_acc = px.bar(
        accuracy_df,
        x="Model",
        y="Accuracy",
        text="Accuracy",
        range_y=[0, 100],
        color="Model",
        color_discrete_map={"VADER": "#636EFA", "Naive Bayes": "#EF553B", "SVM": "#00CC96"}
    )

    fig_acc.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig_acc.update_layout(
        yaxis_title="Accuracy (%)",
        xaxis_title="Model",
        title="Model Accuracy Comparison",
        showlegend=False,
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )

    st.plotly_chart(fig_acc, use_container_width=True)

    st.markdown("### Detailed Metrics")

    vader_metrics = """
| Class     | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Negative  | 0.735     | 0.735  | 0.735    | 34      |
| Neutral   | 0.826     | 0.514  | 0.633    | 37      |
| Positive  | 0.648     | 0.875  | 0.745    | 40      |
| **Avg (W)**| 0.734    | 0.712  | 0.705    | 111     |
    """

    nb_metrics = """
| Class     | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Negative  | 0.60      | 0.43   | 0.50     | 7       |
| Neutral   | 0.80      | 0.50   | 0.62     | 8       |
| Positive  | 0.54      | 0.88   | 0.67     | 8       |
| **Avg (W)**| 0.65     | 0.61   | 0.60     | 23      |
    """

    svm_metrics = """
| Class     | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Negative  | 0.38      | 0.43   | 0.40     | 7       |
| Neutral   | 0.43      | 0.38   | 0.40     | 8       |
| Positive  | 0.50      | 0.50   | 0.50     | 8       |
| **Avg (W)**| 0.44     | 0.43   | 0.43     | 23      |
    """

    st.markdown("#### VADER")
    st.markdown(vader_metrics)

    st.markdown("#### Naive Bayes")
    st.markdown(nb_metrics)

    st.markdown("#### SVM")
    st.markdown(svm_metrics)

    st.markdown("---")
    st.markdown("### Summary")
    st.write("""
VADER performed best overall, especially with Positive and Negative reviews, thanks to its sentiment-focused lexicon.
Naive Bayes had good recall on Positive but struggled with Negative.
SVM underperformed on this dataset, likely due to the size of the dataset.
    """)

