
import re

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from wordcloud import WordCloud

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="YouTube Comment Sentiment Analysis",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}
div[data-testid="stMetric"] {
    background: #171b24;
    border: 1px solid #303746;
    padding: 16px;
    border-radius: 12px;
}
.success-banner {
    background: #123b2b;
    color: #8ff0b5;
    padding: 14px 18px;
    border-radius: 10px;
    margin: 10px 0 22px;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 YouTube Comment Sentiment Analysis")
st.caption(
    "Analyze public YouTube comments using a trained "
    "TF-IDF + Logistic Regression model."
)

API_BASE = "https://www.googleapis.com/youtube/v3"
MODEL_PATH = "sentiment_pipeline.pkl"


# --------------------------------------------------
# 2. LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# 3. EXTRACT YOUTUBE VIDEO ID
# --------------------------------------------------

def extract_video_id(url):
    url = (url or "").strip()

    patterns = [
        r"(?:youtube\.com/watch\?.*?\bv=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/(?:shorts|live|embed)/)([A-Za-z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url

    return None


# --------------------------------------------------
# 4. YOUTUBE API REQUEST
# --------------------------------------------------

def api_get(endpoint, params):
    response = requests.get(
        f"{API_BASE}/{endpoint}",
        params=params,
        timeout=30,
    )

    try:
        payload = response.json()
    except ValueError:
        payload = {}

    if response.status_code != 200:
        error = payload.get("error", {}).get(
            "message", response.text
        )
        raise RuntimeError(
            f"YouTube API error ({response.status_code}): {error}"
        )

    return payload


# --------------------------------------------------
# 5. FETCH VIDEO DETAILS
# --------------------------------------------------

def fetch_video_details(video_id, api_key):
    data = api_get(
        "videos",
        {
            "part": "snippet,statistics",
            "id": video_id,
            "key": api_key,
        },
    )

    items = data.get("items", [])

    if not items:
        raise RuntimeError(
            "Video not found or unavailable."
        )

    item = items[0]
    snippet = item.get("snippet", {})
    stats = item.get("statistics", {})

    return {
        "title": snippet.get("title", "Unknown"),
        "channel": snippet.get("channelTitle", "Unknown"),
        "published_at": snippet.get("publishedAt"),
        "views": int(stats.get("viewCount", 0)),
        "likes": int(stats.get("likeCount", 0)),
        "comment_count": int(stats.get("commentCount", 0)),
    }


# --------------------------------------------------
# 6. FETCH PUBLIC COMMENTS WITH PAGINATION
# --------------------------------------------------

def fetch_comments(video_id, api_key, max_comments):
    rows = []
    page_token = None

    while len(rows) < max_comments:

        page_size = min(
            100, max_comments - len(rows)
        )

        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": page_size,
            "textFormat": "plainText",
            "order": "time",
            "key": api_key,
        }

        if page_token:
            params["pageToken"] = page_token

        data = api_get("commentThreads", params)

        for item in data.get("items", []):
            comment = (
                item.get("snippet", {})
                .get("topLevelComment", {})
                .get("snippet", {})
            )

            rows.append({
                "CommentText": comment.get(
                    "textDisplay", ""
                ),
                "Author": comment.get(
                    "authorDisplayName", "Unknown"
                ),
                "PublishedAt": comment.get(
                    "publishedAt"
                ),
                "Likes": int(
                    comment.get("likeCount", 0)
                ),
            })

        page_token = data.get("nextPageToken")

        if not page_token or not data.get("items"):
            break

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df[
            df["CommentText"].astype(str).str.strip().ne("")
        ].copy()

        df["PublishedAt"] = pd.to_datetime(
            df["PublishedAt"],
            errors="coerce",
            utc=True,
        )

    return df


# --------------------------------------------------
# 7. PREDICT SENTIMENT
# --------------------------------------------------

def normalize_sentiment(value):
    value = str(value).strip().lower()

    labels = {
        "positive": "Positive",
        "negative": "Negative",
        "neutral": "Neutral",
    }

    return labels.get(value, value.title())


def predict_sentiments(model, df):
    result = df.copy()

    texts = result["CommentText"].astype(str)

    result["Sentiment"] = [
        normalize_sentiment(label)
        for label in model.predict(texts)
    ]

    # Confidence is available for models supporting predict_proba.
    result["Confidence"] = float("nan")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(texts)
        model_classes = [
            normalize_sentiment(label)
            for label in model.classes_
        ]

        result["Confidence"] = [
            float(
                probabilities[i][
                    model_classes.index(label)
                ]
            )
            for i, label in enumerate(result["Sentiment"])
        ]

    return result


# --------------------------------------------------
# 8. INPUT FORM
# --------------------------------------------------

with st.form("analyze_form"):

    video_url = st.text_input(
        "🎬 Enter YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=XXXXXXXXXXX",
    )

    max_comments = st.select_slider(
        "Maximum comments to analyze",
        options=[100, 200, 500, 1000],
        value=500,
    )

    submitted = st.form_submit_button(
        "Analyze Video",
        use_container_width=True,
    )


# --------------------------------------------------
# 9. ANALYZE VIDEO
# --------------------------------------------------

if submitted:

    video_id = extract_video_id(video_url)

    if not video_id:
        st.error("Please enter a valid YouTube video URL.")
        st.stop()

    try:
        api_key = st.secrets["YOUTUBE_API_KEY"]
    except Exception:
        st.error(
            "API key missing. Add YOUTUBE_API_KEY "
            "to .streamlit/secrets.toml."
        )
        st.stop()

    try:
        model = load_model()
    except Exception as exc:
        st.error(
            f"Could not load {MODEL_PATH}. "
            f"Keep it beside app.py. Details: {exc}"
        )
        st.stop()

    with st.spinner(
        "Fetching comments and predicting sentiment..."
    ):
        try:
            video = fetch_video_details(
                video_id, api_key
            )

            comments_df = fetch_comments(
                video_id,
                api_key,
                int(max_comments),
            )

        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()

        except requests.RequestException as exc:
            st.error(f"Network error: {exc}")
            st.stop()

        if comments_df.empty:
            st.warning(
                "No public comments found for this video."
            )
            st.stop()

        analyzed = predict_sentiments(
            model, comments_df
        )

    st.session_state["analysis_result"] = {
        "video": video,
        "comments": analyzed,
    }


# --------------------------------------------------
# 10. DISPLAY DASHBOARD
# --------------------------------------------------

if "analysis_result" in st.session_state:

    result = st.session_state["analysis_result"]

    video = result["video"]
    df = result["comments"].copy()

    st.markdown(
        '<div class="success-banner">'
        '✅ Analysis completed successfully.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Video information
    st.subheader("🎥 Video Details")

    st.markdown(f"**Title:** {video['title']}")
    st.markdown(f"**Channel:** {video['channel']}")

    if video.get("published_at"):
        st.caption(
            f"Published: {video['published_at']}"
        )

    c1, c2, c3 = st.columns(3)

    c1.metric("👀 Views", f"{video['views']:,}")
    c2.metric("👍 Likes", f"{video['likes']:,}")
    c3.metric(
        "💬 Total Video Comments",
        f"{video['comment_count']:,}",
    )

    # Sentiment metrics
    st.subheader("📊 Sentiment Analysis")

    counts = df["Sentiment"].value_counts()

    positive = int(counts.get("Positive", 0))
    negative = int(counts.get("Negative", 0))
    neutral = int(counts.get("Neutral", 0))
    total = len(df)

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Total Analyzed", f"{total:,}")
    m2.metric(
        "😊 Positive",
        f"{positive:,}",
        f"{positive / total:.1%}",
    )
    m3.metric(
        "😠 Negative",
        f"{negative:,}",
        f"{negative / total:.1%}",
    )
    m4.metric(
        "😐 Neutral",
        f"{neutral:,}",
        f"{neutral / total:.1%}",
    )

    # Pie chart and bar chart
    left, right = st.columns(2)

    with left:
        st.markdown("#### Sentiment Distribution")

        pie_data = pd.Series({
            "Positive": positive,
            "Negative": negative,
            "Neutral": neutral,
        })

        pie_data = pie_data[pie_data > 0]

        fig, ax = plt.subplots(figsize=(7, 5))

        if not pie_data.empty:
            ax.pie(
                pie_data.values,
                labels=pie_data.index,
                autopct="%1.1f%%",
                startangle=90,
                explode=[0.04] * len(pie_data),
            )

            ax.axis("equal")

            st.pyplot(
                fig,
                use_container_width=True,
            )
        else:
            st.info("No sentiment predictions to chart.")

        plt.close(fig)

    with right:
        st.markdown("#### Sentiment Counts")

        bar_data = pd.DataFrame({
            "Sentiment": [
                "Positive", "Neutral", "Negative"
            ],
            "Comments": [
                positive, neutral, negative
            ],
        }).set_index("Sentiment")

        st.bar_chart(bar_data)

    # Sentiment over time
    st.subheader("📌 Sentiment Over Time")

    timeline = df.dropna(
        subset=["PublishedAt"]
    ).copy()

    if not timeline.empty:

        timeline["Date"] = (
            timeline["PublishedAt"].dt.floor("D")
        )

        timeline["Score"] = timeline[
            "Sentiment"
        ].map({
            "Positive": 1,
            "Neutral": 0,
            "Negative": -1,
        })

        trend = (
            timeline.groupby("Date")["Score"]
            .mean()
            .sort_index()
        )

        if len(trend) > 1:
            st.line_chart(trend)
            st.caption(
                "Daily sentiment score: Positive = +1, "
                "Neutral = 0, Negative = −1."
            )
        else:
            st.info(
                "Not enough distinct comment dates "
                "to show a time trend."
            )
    else:
        st.info("Comment dates are unavailable.")

    # Word cloud
    st.subheader("☁️ Word Cloud")

    cloud_text = " ".join(
        df["CommentText"].dropna().astype(str)
    )

    try:
        if cloud_text.strip():
            wordcloud = WordCloud(
                width=1200,
                height=550,
                background_color="white",
                collocations=False,
                max_words=150,
            ).generate(cloud_text)

            fig, ax = plt.subplots(figsize=(14, 6))
            ax.imshow(
                wordcloud,
                interpolation="bilinear",
            )
            ax.axis("off")

            st.pyplot(
                fig,
                use_container_width=True,
            )

            plt.close(fig)
        else:
            st.info("No text available for the word cloud.")

    except ValueError:
        st.info(
            "There aren't enough usable words "
            "to generate a word cloud."
        )

    # Top comments
    st.subheader("💬 Comment Highlights")

    positive_df = df[
        df["Sentiment"] == "Positive"
    ].copy()

    negative_df = df[
        df["Sentiment"] == "Negative"
    ].copy()

    if df["Confidence"].notna().any():
        positive_df = positive_df.sort_values(
            "Confidence", ascending=False
        )
        negative_df = negative_df.sort_values(
            "Confidence", ascending=False
        )

    with st.expander("🌟 Top 5 Positive Comments"):

        if positive_df.empty:
            st.info("No positive comments predicted.")
        else:
            for i, row in enumerate(
                positive_df.head(5).itertuples(),
                start=1,
            ):
                st.markdown(
                    f"**{i}.** {row.CommentText}"
                )
                st.caption(f"👍 {row.Likes} likes")
                st.divider()

    with st.expander("🚨 Top 5 Negative Comments"):

        if negative_df.empty:
            st.info("No negative comments predicted.")
        else:
            for i, row in enumerate(
                negative_df.head(5).itertuples(),
                start=1,
            ):
                st.markdown(
                    f"**{i}.** {row.CommentText}"
                )
                st.caption(f"👍 {row.Likes} likes")
                st.divider()

    # Complete comment table
    st.subheader("📄 Analyzed Comments")

    display_df = df[
        [
            "CommentText",
            "Sentiment",
            "PublishedAt",
            "Likes",
        ]
    ].copy()

    display_df["PublishedAt"] = (
        display_df["PublishedAt"].astype(str)
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    # Download results
    st.download_button(
        "⬇️ Download Analyzed Comments (CSV)",
        data=df.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="youtube_sentiment_analysis.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        "Sentiment labels are model predictions. "
        "Spam detection is not implemented."
    )

else:
    st.info(
        "Paste a YouTube video URL above and click "
        "**Analyze Video** to get started."
    )
