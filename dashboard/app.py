import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import sqlite3
import json
import os
import traceback

# --- Config ---
st.set_page_config(
    page_title="SentimentPulse",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Force Dark Theme via CSS (Streamlit natively supports dark mode, 
# but this ensures colors look good in our custom dark layout)
st.markdown("""
<style>
    /* Dark theme specific adjustments */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .css-1d391kg, .css-1lcbmhc {
        background-color: #1E2127;
    }
    /* Hide footer */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        data_path = os.path.join("outputs", "cleaned_tweets.csv")
        df = pd.read_csv(data_path)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_resource
def load_db_connection():
    db_path = os.path.join("outputs", "sentiment_pulse.db")
    if os.path.exists(db_path):
        return sqlite3.connect(db_path)
    return None

def main():
    try:
        st.sidebar.title("📈 SentimentPulse")
        st.sidebar.markdown("Real-Time Social Media Sentiment Analysis & Trend Intelligence Platform")
        
        pages = [
            "Sentiment Overview", 
            "Model Comparison", 
            "Trend Intelligence", 
            "RL Agent Alerts", 
            "SQL Insights"
        ]
        
        selection = st.sidebar.radio("Navigation", pages)
        
        st.sidebar.markdown("---")
        
        if selection == "Sentiment Overview":
            page_sentiment_overview()
        elif selection == "Model Comparison":
            page_model_comparison()
        elif selection == "Trend Intelligence":
            page_trend_intelligence()
        elif selection == "RL Agent Alerts":
            page_rl_agent_alerts()
        elif selection == "SQL Insights":
            page_sql_insights()

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.code(traceback.format_exc())

# --- Page 1: Sentiment Overview ---
def page_sentiment_overview():
    st.title("Sentiment Overview")
    df = load_data()
    
    if df.empty:
        st.info("Please run this command in terminal: python sql/setup_db.py")
        return
        
    total_tweets = len(df)
    counts = df['sentiment'].value_counts()
    pos_pct = (counts.get('positive', 0) / total_tweets) * 100
    neg_pct = (counts.get('negative', 0) / total_tweets) * 100
    neu_pct = (counts.get('neutral', 0) / total_tweets) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tweets Analyzed", f"{total_tweets:,}")
    col2.metric("Positive %", f"{pos_pct:.1f}%")
    col3.metric("Negative %", f"{neg_pct:.1f}%")
    col4.metric("Neutral %", f"{neu_pct:.1f}%")
    
    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Sentiment Distribution")
        fig_pie = px.pie(
            names=counts.index, 
            values=counts.values,
            color=counts.index,
            color_discrete_map={'positive':'#00CC96', 'negative':'#EF553B', 'neutral':'#636EFA'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_chart2:
        st.subheader("Top 10 Most Active Users")
        user_counts = df['user'].value_counts().head(10).reset_index()
        user_counts.columns = ['user', 'count']
        fig_bar = px.bar(user_counts, x='user', y='count', color='count', color_continuous_scale='Blues')
        st.plotly_chart(fig_bar, use_container_width=True)
        
    st.subheader("Sentiment Trend Over Time")
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True, errors='coerce')
    df = df.dropna(subset=['date'])
    df['date'] = df['date'].dt.date
    daily_sent = df.groupby(['date', 'sentiment']).size().reset_index(name='count')
    daily_sent = daily_sent.sort_values('date')
    fig_line = px.line(
        daily_sent, 
        x='date', 
        y='count', 
        color='sentiment',
        title='Sentiment Trend Over Time',
        labels={'date': 'Date', 'count': 'Number of Tweets'},
        color_discrete_map={'positive':'#00CC96', 'negative':'#EF553B', 'neutral':'#636EFA'}
    )
    st.plotly_chart(fig_line, use_container_width=True)

# --- Page 2: Model Comparison ---
def page_model_comparison():
    st.title("Model Comparison (VADER vs TF-IDF vs TextBlob)")
    
    metrics_path = os.path.join("outputs", "model_metrics.json")
    results_path = os.path.join("outputs", "sentiment_results.csv")
    
    if not os.path.exists(metrics_path) or not os.path.exists(results_path):
        st.info("Please run this command in terminal: python models/sentiment_analysis.py")
        return
        
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    metrics_df = pd.DataFrame(metrics).T.reset_index(drop=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Accuracy Table")
        st.dataframe(metrics_df[['model', 'accuracy', 'precision', 'recall', 'f1']].style.format({'accuracy': '{:.4f}', 'precision': '{:.4f}', 'recall': '{:.4f}', 'f1': '{:.4f}'}))
        
    with col2:
        st.subheader("F1 Score Comparison")
        fig = px.bar(metrics_df, x='model', y='f1', color='model', color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)
        
    st.subheader("Best Model Recommendation")
    best_model = metrics_df.loc[metrics_df['f1'].idxmax()]['model']
    st.info(f"🏆 Based on F1 Score, the recommended model is **{best_model}**.")
    
    st.subheader("Sample Predictions")
    results_df = pd.read_csv(results_path)
    st.dataframe(results_df.head(20))

# --- Page 3: Trend Intelligence ---
def page_trend_intelligence():
    st.title("Trend Intelligence")
    
    topics_path = os.path.join("outputs", "topics.json")
    keywords_path = os.path.join("outputs", "trending_keywords.csv")
    
    if not os.path.exists(keywords_path):
        st.info("Please run this command in terminal: python models/trend_detection.py")
        return
        
    keywords_df = pd.read_csv(keywords_path)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Trending Keywords WordCloud")
        overall = keywords_df[keywords_df['sentiment_type'] == 'overall']
        word_freq = dict(zip(overall['keyword'], overall['frequency']))
        if word_freq:
            wc = WordCloud(width=800, height=400, background_color='black').generate_from_frequencies(word_freq)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            fig.patch.set_facecolor('#0E1117')
            st.pyplot(fig)
            
    with col2:
        st.subheader("Top 10 Trending Keywords")
        top10 = overall.head(10)
        fig_bar = px.bar(top10, x='keyword', y='frequency', color='frequency', color_continuous_scale='Purples')
        st.plotly_chart(fig_bar, use_container_width=True)
        
    st.subheader("Positive vs Negative Keywords")
    c1, c2 = st.columns(2)
    pos_kw = keywords_df[keywords_df['sentiment_type'] == 'positive'].head(10)
    neg_kw = keywords_df[keywords_df['sentiment_type'] == 'negative'].head(10)
    with c1:
        fig_pos = px.bar(pos_kw, x='keyword', y='frequency', title="Positive Keywords", color_discrete_sequence=['#00CC96'])
        st.plotly_chart(fig_pos, use_container_width=True)
    with c2:
        fig_neg = px.bar(neg_kw, x='keyword', y='frequency', title="Negative Keywords", color_discrete_sequence=['#EF553B'])
        st.plotly_chart(fig_neg, use_container_width=True)
        
    if os.path.exists(topics_path):
        st.subheader("Topic Modeling Results (LDA)")
        with open(topics_path, "r") as f:
            topics = json.load(f)
        
        cols = st.columns(5)
        for i, (topic, words) in enumerate(topics.items()):
            col = cols[i % 5]
            with col:
                st.markdown(f"**{topic}**")
                for w in words:
                    st.write(f"- {w}")

# --- Page 4: RL Agent Alerts ---
def page_rl_agent_alerts():
    st.title("Reinforcement Learning Agent Alerts")
    
    rl_path = os.path.join("outputs", "rl_results.csv")
    if not os.path.exists(rl_path):
        st.info("Please run this command in terminal: python models/rl_agent.py")
        return
        
    rl_df = pd.read_csv(rl_path)
    
    # Assuming one episode's worth of data for visualization (e.g. episode 0)
    ep_df = rl_df[rl_df['episode'] == rl_df['episode'].iloc[0]].copy()
    
    avg_reward = rl_df.groupby('episode')['reward'].sum().mean()
    alert_acc = rl_df['correct'].mean() * 100
    total_alerts = len(rl_df[rl_df['action_taken'].isin([1, 2])])
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Average Episode Reward", f"{avg_reward:.2f}")
    c2.metric("Alert Accuracy", f"{alert_acc:.1f}%")
    c3.metric("Total Alerts Issued", f"{total_alerts}")
    
    st.subheader("Sentiment Score & Agent Actions")
    
    # Create chart
    fig = go.Figure()
    
    # Add line
    fig.add_trace(go.Scatter(
        x=ep_df['date'] if 'date' in ep_df.columns and not ep_df['date'].isnull().all() else ep_df.index,
        y=ep_df['sentiment_score'],
        mode='lines',
        name='Sentiment Score'
    ))
    
    # Add markers for actions
    pos_alerts = ep_df[ep_df['action_taken'] == 1]
    neg_alerts = ep_df[ep_df['action_taken'] == 2]
    no_alerts = ep_df[ep_df['action_taken'] == 0]
    
    x_col = 'date' if 'date' in ep_df.columns and not ep_df['date'].isnull().all() else ep_df.index
    
    fig.add_trace(go.Scatter(
        x=pos_alerts[x_col], y=pos_alerts['sentiment_score'],
        mode='markers', marker=dict(color='green', size=12, symbol='triangle-up'),
        name='Positive Alert'
    ))
    fig.add_trace(go.Scatter(
        x=neg_alerts[x_col], y=neg_alerts['sentiment_score'],
        mode='markers', marker=dict(color='red', size=12, symbol='triangle-down'),
        name='Negative Alert'
    ))
    fig.add_trace(go.Scatter(
        x=no_alerts[x_col], y=no_alerts['sentiment_score'],
        mode='markers', marker=dict(color='grey', size=8),
        name='No Alert'
    ))
    
    fig.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Recent Alerts Table")
    alerts_only = ep_df[ep_df['action_taken'] != 0].copy()
    alerts_only['action_str'] = alerts_only['action_taken'].map({1: 'Positive', 2: 'Negative'})
    st.dataframe(alerts_only[['date', 'sentiment_score', 'action_str', 'reward', 'correct']])

# --- Page 5: SQL Insights ---
def page_sql_insights():
    st.title("SQL Database Insights")
    
    conn = load_db_connection()
    if not conn:
        st.info("Please run this command in terminal: python sql/setup_db.py")
        return
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Tweet Count by Hour")
        try:
            q_hour = "SELECT substr(date, 12, 2) as hour, COUNT(*) as count FROM tweets GROUP BY hour ORDER BY hour"
            df_hour = pd.read_sql(q_hour, conn)
            # Create a heatmap representation
            fig_hm = px.density_heatmap(df_hour, x="hour", y="count", color_continuous_scale="Viridis")
            st.plotly_chart(fig_hm, use_container_width=True)
        except Exception as e:
            st.error("Could not load hour data.")
            
    with col2:
        st.subheader("Sentiment Distribution By Top Users")
        try:
            # Get top users first
            top_users = pd.read_sql("SELECT user FROM tweets GROUP BY user ORDER BY COUNT(*) DESC LIMIT 5", conn)['user'].tolist()
            if top_users:
                users_str = "','".join(top_users)
                q_dist = f"SELECT user, sentiment, COUNT(*) as count FROM tweets WHERE user IN ('{users_str}') GROUP BY user, sentiment"
                df_dist = pd.read_sql(q_dist, conn)
                fig_bar = px.bar(df_dist, x='user', y='count', color='sentiment', barmode='group')
                st.plotly_chart(fig_bar, use_container_width=True)
        except Exception as e:
            st.error("Could not load user data.")
            
    st.subheader("Most Common Words Sample")
    try:
        q_words = "SELECT cleaned_text FROM tweets WHERE cleaned_text IS NOT NULL LIMIT 5000"
        df_words = pd.read_sql(q_words, conn)
        all_words = ' '.join(df_words['cleaned_text'].astype(str)).split()
        from collections import Counter
        common = Counter(all_words).most_common(20)
        df_common = pd.DataFrame(common, columns=['Word', 'Count'])
        st.dataframe(df_common.head(10))
    except Exception as e:
        st.error("Could not load word data.")

    st.subheader("Database Statistics")
    try:
        total_rows = pd.read_sql("SELECT COUNT(*) FROM tweets", conn).iloc[0,0]
        db_size_mb = os.path.getsize(os.path.join("outputs", "sentiment_pulse.db")) / (1024*1024)
        st.info(f"**Total Records**: {total_rows:,}")
        st.info(f"**Database Size**: {db_size_mb:.2f} MB")
    except Exception as e:
        pass

if __name__ == "__main__":
    main()
