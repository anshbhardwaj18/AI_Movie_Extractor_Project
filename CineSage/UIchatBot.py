import streamlit as st
from streamlit_lottie import st_lottie
import requests
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Movie Extractor",
    page_icon="🎬",
    layout="wide"
)

load_dotenv()

model = ChatMistralAI(
    model="mistral-small-2506"
)

# ---------------- CSS ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        -45deg,
        #0f172a,
        #1e3a8a,
        #4c1d95,
        #0f172a
    );
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
}

@keyframes gradient {
    0% {background-position:0% 50%;}
    50% {background-position:100% 50%;}
    100% {background-position:0% 50%;}
}

.main-card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(30px);
    border-radius: 30px;
    padding: 40px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 8px 40px rgba(0,0,0,0.4);
    animation: slideUp 1s ease;
}

@keyframes slideUp {
    from {
        opacity:0;
        transform:translateY(50px);
    }
    to {
        opacity:1;
        transform:translateY(0px);
    }
}

.title {
    text-align:center;
    font-size:60px;
    font-weight:900;
    color:white;
    animation: glow 2s infinite alternate;
}

@keyframes glow {
    from {
        text-shadow:0 0 20px #3b82f6;
    }
    to {
        text-shadow:
            0 0 20px #8b5cf6,
            0 0 40px #8b5cf6;
    }
}

.subtitle {
    text-align:center;
    color:#d1d5db;
    font-size:20px;
    margin-bottom:30px;
}

# .stTextArea textarea {
#     border-radius:20px !important;
#     background:rgba(255,255,255,0.08) !important;
#     color:white !important;
#     border:1px solid rgba(255,255,255,0.2) !important;
# }
.stTextArea label {
    font-size: 24px !important;
    font-weight: 700 !important;
    color: white !important;
}
        
.stButton>button {
    width:100%;
    height:65px;
    border-radius:20px;
    background:linear-gradient(
        90deg,
        #8b5cf6,
        #3b82f6
    );
    color:white;
    font-size:22px;
    font-weight:700;
    border:none;
    transition:all 0.4s ease;
}

.stButton>button:hover {
    transform:translateY(-5px) scale(1.03);
    box-shadow:
        0 0 30px #8b5cf6,
        0 0 60px #3b82f6;
}

.result-card {
    background:rgba(255,255,255,0.1);
    border-radius:25px;
    padding:30px;
    color:white;
    animation:fadeIn 1s ease;
    margin-top:30px;
}

@keyframes fadeIn {
    from {
        opacity:0;
        transform:translateY(30px);
    }
    to {
        opacity:1;
        transform:translateY(0px);
    }
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOTTIE ----------------
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

movie_animation = load_lottie(
    "https://assets9.lottiefiles.com/packages/lf20_j1adxtyb.json"
)

st_lottie(
    movie_animation,
    height=250,
    key="movie"
)

# ---------------- TITLE ----------------
st.markdown(
    """
    <div class='title'>🎬 Movie Information Extractor</div>
    <div class='subtitle'>
        Extract useful information from movie descriptions using AI
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- PROMPT ----------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an expert movie information extractor Assistant.

        Analyze the given movie paragraph and extract:

        - Movie Name
        - IMDB Rating
        - Genre
        - Director
        - Cast
        - Main Characters
        - Plot Summary
        - Themes
        - Keywords
        - Notable Features
        - Quick Summary (2-3 lines)

        If information is unavailable, mention 'Not Available'.
        """
    ),
    (
        "human",
        """
        Extract information from this paragraph:

        {paragraph}
        """
    )
])

# ---------------- INPUT CARD ----------------
st.markdown("<div class='main-card'>", unsafe_allow_html=True)

paragraph = st.text_area(
    "Movie Description",
    height=250,
    placeholder="Paste your movie description here..."
)

extract = st.button("🚀 Extract Information")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- RESULT ----------------
if extract:
    if paragraph.strip() == "":
        st.warning("Please enter a movie description.")
    else:
        with st.spinner("Analyzing movie..."):
            final_prompt = prompt.invoke(
                {"paragraph": paragraph}
            )

            response = model.invoke(final_prompt)

        st.markdown(
            f"""
            <div class='result-card'>
            {response.content.replace(chr(10), '<br>')}
            </div>
            """,
            unsafe_allow_html=True
        )