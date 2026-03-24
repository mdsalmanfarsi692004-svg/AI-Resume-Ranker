import streamlit as st
import PyPDF2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- PAGE CONFIGURATION (Must be the first Streamlit command) ---
st.set_page_config(page_title="AI Resume Ranker Pro", page_icon="🎯", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS FOR TITLE, ALIGNMENT & HIDING DEFAULT ELEMENTS ---
st.markdown("""
<style>
    /* Hide Streamlit top margin and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 2rem;}
    
    /* Glowing Title with properly colored emojis */
    .main-title {
        text-align: center; font-size: 3.5rem; font-weight: 900; margin-bottom: 0px;
    }
    .gradient-text {
        font-family: 'Inter', sans-serif;
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent;
    }
    .sub-title { text-align: center; font-size: 1.2rem; color: #a0aec0; margin-top: -10px; margin-bottom: 30px; }
    
    /* Custom Styling for the Primary Button */
    button[kind="primary"] {
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        border: none;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.5);
    }
    button[kind="primary"]:hover {
        transform: translateY(-2px);
    }

    /* Force Sidebar Image to Center */
    [data-testid="stSidebar"] img {
        border-radius: 10px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- NLP LOGIC FUNCTIONS ---
@st.cache_data
def clean_text(text):
    text = re.sub(r'\W+', ' ', text)  
    text = re.sub(r'\d+', ' ', text)  
    return text.lower().strip()

def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        return "".join(page.extract_text() or "" for page in pdf_reader.pages)
    except Exception: return ""

def get_matched_keywords(jd_text, resume_text):
    jd_words = set(re.findall(r'\b[a-z]{3,}\b', jd_text)) - ENGLISH_STOP_WORDS
    res_words = set(re.findall(r'\b[a-z]{3,}\b', resume_text))
    matched = list(jd_words.intersection(res_words))
    return ", ".join(matched[:15]) if matched else "No major keywords found"

# --- SIDEBAR (Fully Centered & Optimized) ---
with st.sidebar:
    st.image("https://tse2.mm.bing.net/th/id/OIP.6YUE2M3eFDTDTy_BRHyHSwHaHa?pid=ImgDet&w=184&h=184&c=7&dpr=1.3&o=7&rm=3", use_container_width=True)
    
    st.markdown("""
        <h2 style='text-align: center; margin-top: 0px;'>⚙️ ATS Controls</h2>
        <div style='text-align: center; background-color: rgba(59, 130, 246, 0.1); padding: 12px; border-radius: 8px; color: #60A5FA; font-size: 14px; margin-bottom: 20px;'>
            Upload candidate resumes and provide a Job Description. The AI will rank them instantly.
        </div>
        <hr style='border: 1px solid rgba(255,255,255,0.1);'>
        <h3 style='text-align: center; margin-bottom: 15px;'>💡 How it works:</h3>
        
        <div style='text-align: center; font-size: 14px; line-height: 2; color: #E2E8F0; white-space: nowrap;'>
            1. Extracts text from PDFs.<br>
            2. Cleans & Tokenizes data.<br>
            3. Uses <b>TF-IDF Vectorization</b>.<br>
            4. Calculates <b>Cosine Similarity</b>.
        </div>
        <hr style='border: 1px solid rgba(255,255,255,0.1); margin-top: 20px;'>
        <p style='text-align: center; font-size: 12px; color: #94A3B8;'>Powered by Scikit-Learn & Python</p>
    """, unsafe_allow_html=True)

# --- HERO SECTION ---
st.markdown('<div class="main-title">🎯 <span class="gradient-text">AI Resume Ranker Pro</span> 🎯</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Advanced ATS Engine with TF-IDF, Keyword Extraction & Analytics</div>', unsafe_allow_html=True)

# --- INPUT SECTION ---
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 📝 **Step 1: Job Description**")
    job_description = st.text_area("JD Input", height=200, placeholder="Paste the Job Description here (Skills, Requirements, Roles)...", label_visibility="collapsed")

with col2:
    st.markdown("### 📂 **Step 2: Candidate Resumes**")
    uploaded_files = st.file_uploader("Upload PDFs", type=['pdf'], accept_multiple_files=True, label_visibility="collapsed")

st.write("") # Spacer

# --- NATIVE CENTERED BUTTON ---
btn_col1, btn_col2, btn_col3 = st.columns([1, 1.5, 1])

with btn_col2:
    start_analysis = st.button("⚡ Analyze & Rank Candidates", type="primary", use_container_width=True)

st.divider()

# --- RANKING ENGINE & DASHBOARD ---
if start_analysis:
    if not job_description.strip() or not uploaded_files:
        st.error("⚠️ Please provide BOTH a Job Description and at least one Resume.")
    else:
        with st.spinner("🧠 AI is analyzing skills, vectors, and scoring resumes..."):
            
            cleaned_jd = clean_text(job_description)
            
            resume_data = []
            for file in uploaded_files:
                raw_text = extract_text_from_pdf(file)
                cleaned_text = clean_text(raw_text)
                matched_kws = get_matched_keywords(cleaned_jd, cleaned_text)
                resume_data.append({
                    "Candidate Name": file.name.replace('.pdf', ''), 
                    "Cleaned_Text": cleaned_text,
                    "Matched Keywords": matched_kws
                })
            
            all_texts = [cleaned_jd] + [data["Cleaned_Text"] for data in resume_data]
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
            for i, data in enumerate(resume_data):
                data["Match Score (%)"] = round(similarity_scores[i] * 100, 2)
            
            df_results = pd.DataFrame(resume_data)
            df_results = df_results.sort_values(by="Match Score (%)", ascending=False).reset_index(drop=True)
            
            top_candidate = df_results.iloc[0]['Candidate Name']
            top_score = df_results.iloc[0]['Match Score (%)']
            avg_score = round(df_results['Match Score (%)'].mean(), 2)
            total_candidates = len(df_results)

            # BULLETPROOF HTML CONCATENATION WITH NEW COLORED BORDERS
            html_dashboard = (
                f"<div style='background: linear-gradient(90deg, #064e3b, #047857); color: white; text-align: center; padding: 15px; border-radius: 10px; font-weight: bold; font-size: 18px; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(4, 120, 87, 0.4); border: 1px solid #059669;'>"
                f"✅ Analysis Complete! Here is your detailed AI HR Report."
                f"</div>"
                f"<div style='display: flex; justify-content: center; gap: 20px; margin-bottom: 35px;'>"
                
                f"<div style='background: linear-gradient(145deg, #1e293b, #0f172a); border: 1px solid #334155; border-top: 4px solid #38bdf8; border-radius: 15px; padding: 25px 20px; width: 30%; text-align: center; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.5);'>"
                f"<div style='color: #94a3b8; font-size: 14px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;'>👥 Total Evaluated</div>"
                f"<div style='color: #f8fafc; font-size: 42px; font-weight: 900; margin-bottom: 5px;'>{total_candidates}</div>"
                f"<div style='color: #38bdf8; font-size: 14px; font-weight: 500;'>Candidates Processed</div>"
                f"</div>"

                f"<div style='background: linear-gradient(145deg, #1e293b, #0f172a); border: 1px solid #334155; border-top: 4px solid #10b981; border-radius: 15px; padding: 25px 20px; width: 30%; text-align: center; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.5);'>"
                f"<div style='color: #94a3b8; font-size: 14px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;'>🏆 Top Match Score</div>"
                f"<div style='color: #10b981; font-size: 42px; font-weight: 900; margin-bottom: 5px;'>{top_score}%</div>"
                f"<div style='color: #34d399; font-size: 14px; font-weight: 600;'>Best: {top_candidate}</div>"
                f"</div>"

                f"<div style='background: linear-gradient(145deg, #1e293b, #0f172a); border: 1px solid #334155; border-top: 4px solid #a78bfa; border-radius: 15px; padding: 25px 20px; width: 30%; text-align: center; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.5);'>"
                f"<div style='color: #94a3b8; font-size: 14px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;'>📊 Average Score</div>"
                f"<div style='color: #f8fafc; font-size: 42px; font-weight: 900; margin-bottom: 5px;'>{avg_score}%</div>"
                f"<div style='color: #a78bfa; font-size: 14px; font-weight: 500;'>Overall Pool Quality</div>"
                f"</div>"
                
                f"</div>"
            )
            st.markdown(html_dashboard, unsafe_allow_html=True)

            # Chart & Table Row
            c1, c2 = st.columns([1.5, 2])
            
            with c1:
                st.markdown("### 📈 **Comparison Chart**")
                chart_data = df_results.set_index("Candidate Name")[["Match Score (%)"]]
                st.bar_chart(chart_data, color="#00C9FF")
            
            with c2:
                st.markdown("### 🏆 **Leaderboard**")
                st.dataframe(
                    df_results[["Candidate Name", "Match Score (%)"]],
                    column_config={
                        "Match Score (%)": st.column_config.ProgressColumn("Match Score (%)", format="%f%%", min_value=0, max_value=100)
                    },
                    hide_index=True, use_container_width=True
                )
            
            st.divider()
            
            # In-Depth Profile Expanders
            st.markdown("### 🕵️‍♂️ **In-Depth Candidate Analysis**")
            for index, row in df_results.iterrows():
                with st.expander(f"📌 {row['Candidate Name']} - Score: {row['Match Score (%)']}%"):
                    st.write(f"**🔥 Top Matched Skills/Keywords:**")
                    st.info(row['Matched Keywords'])
            
            # Download Button
            st.divider()
            csv_export = df_results[["Candidate Name", "Match Score (%)", "Matched Keywords"]].to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download Full HR Report (CSV)", data=csv_export, file_name='advanced_ats_report.csv', mime='text/csv')