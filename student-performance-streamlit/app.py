import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Page Configuration & Custom Styling
st.set_page_config(page_title="Student Performance Pro", layout="wide")

# Inject Custom CSS for modern fonts and card styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    div[data-testid="stMetric"] {
        background-color: #1f2937;
        border: 1px solid #374151;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.25);
    }

    div[data-testid="stMetric"] label {
        color: #e5e7eb !important;
    }

    div[data-testid="stMetric"] div {
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎓 Student Performance Analytics")
st.markdown("---")

# 2. Data Upload & Processing
uploaded_file = st.file_uploader("📤 Upload Dataset (student-por.csv)", type=["csv"])

required_columns = ["sex", "internet", "studytime", "G3", "G2", "absences", "Walc"]

if uploaded_file is not None:
    try:
        # Auto-detect CSV separator, so both semicolon (;) and comma (,) files work.
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=None, engine="python")

        # Clean column names in case the CSV has extra spaces.
        df.columns = df.columns.str.strip()

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            st.error(
                "The uploaded file is missing these required columns: "
                + ", ".join(missing_cols)
            )
            st.stop()

    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
else:
    st.info("💡 Please upload the student-por.csv file to begin the analysis.")
    st.stop()

# 3. Professional Sidebar Filters
with st.sidebar:
    st.header("⚙️ Filter Settings")

    with st.expander("👤 Demographics", expanded=True):
        gender = st.multiselect(
            "Select Gender",
            options=sorted(df["sex"].dropna().unique()),
            default=sorted(df["sex"].dropna().unique())
        )
        internet = st.radio(
            "Internet Access",
            options=sorted(df["internet"].dropna().unique()),
            index=0,
            horizontal=True
        )

    with st.expander("📚 Study Habits", expanded=True):
        study_options = sorted(df["studytime"].dropna().unique())
        study_range = st.select_slider(
            "Study Time Level (1: Low -> 4: High)",
            options=study_options,
            value=(min(study_options), max(study_options))
        )

# Filtering the dataset
filtered_df = df[
    (df["sex"].isin(gender)) &
    (df["internet"] == internet) &
    (df["studytime"].between(study_range[0], study_range[1]))
].copy()

if filtered_df.empty:
    st.warning("No data matches the selected filters. Please adjust the sidebar filters.")
    st.stop()

# 4. KPI Metrics
total_students = filtered_df.shape[0]
avg_g3 = round(filtered_df["G3"].mean(), 1)
avg_study = round(filtered_df["studytime"].mean(), 1)
avg_absences = round(filtered_df["absences"].mean(), 1)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Students", total_students)
m2.metric("Avg Final Grade (G3)", f"{avg_g3} / 20")
m3.metric("Avg Study Level", avg_study)
m4.metric("Avg Absences", avg_absences)

st.markdown("<br>", unsafe_allow_html=True)

# 5. Data Visualization
chart_colors = px.colors.qualitative.Safe

c1, c2 = st.columns(2)

with c1:
    st.subheader("📊 Distribution of Final Grades")
    fig1 = px.histogram(
        filtered_df,
        x="G3",
        nbins=15,
        color_discrete_sequence=["#4F46E5"],
        marginal="box"
    )
    fig1.update_layout(template="plotly_white", bargap=0.1)
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    st.subheader("⏳ Study Time vs. Final Grade")
    fig2 = px.box(
        filtered_df,
        x="studytime",
        y="G3",
        color="studytime",
        color_discrete_sequence=chart_colors
    )
    fig2.update_layout(template="plotly_white", showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

c3, c4 = st.columns([6, 4])

with c3:
    st.subheader("📈 G2 vs. G3 Correlation")
    try:
        fig3 = px.scatter(
            filtered_df,
            x="G2",
            y="G3",
            color="sex",
            size="absences",
            hover_data=["absences", "studytime"],
            color_discrete_sequence=["#F43F5E", "#10B981"],
            trendline="ols"
        )
    except Exception:
        # Fallback if statsmodels is not installed.
        fig3 = px.scatter(
            filtered_df,
            x="G2",
            y="G3",
            color="sex",
            size="absences",
            hover_data=["absences", "studytime"],
            color_discrete_sequence=["#F43F5E", "#10B981"]
        )
        st.info("Install statsmodels to show the OLS trendline: pip install statsmodels")

    fig3.update_layout(template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    st.subheader("🍷 Weekend Alcohol Consumption Impact")
    walc_avg = filtered_df.groupby("Walc", as_index=False)["G3"].mean()
    fig4 = px.line(
        walc_avg,
        x="Walc",
        y="G3",
        markers=True,
        color_discrete_sequence=["#8B5CF6"]
    )
    fig4.update_layout(template="plotly_white", yaxis_range=[0, 20])
    st.plotly_chart(fig4, use_container_width=True)

# Optional parental education chart if the dataset has Medu and Fedu
if {"Medu", "Fedu"}.issubset(filtered_df.columns):
    st.markdown("---")
    st.subheader("👪 Parental Education and Final Grade")
    filtered_df["Parent_Edu"] = filtered_df["Medu"] + filtered_df["Fedu"]

    fig5 = px.box(
        filtered_df,
        x="Parent_Edu",
        y="G3",
        color="Parent_Edu",
        points="all",
        title="Final Grade Distribution by Combined Parental Education"
    )
    fig5.update_layout(template="plotly_white", showlegend=False, yaxis_range=[0, 22])
    st.plotly_chart(fig5, use_container_width=True)

# 6. Data Table Preview
with st.expander("📋 View Detailed Dataset"):
    st.dataframe(filtered_df, use_container_width=True)
    st.markdown("---")
st.header("🔍 Summary of Key Findings")

st.markdown("""
1. **Previous performance matters:** Students with higher G2 scores tend to achieve higher final grades, making G2 a strong predictor of G3.

2. **Study time has a positive but limited effect:** Students who study more generally perform better, but the differences are not extremely large across all groups.

3. **Absences may affect academic outcomes:** Students with more absences may face lower academic performance, although this relationship should be examined together with other variables.

4. **Weekend alcohol consumption shows a negative pattern:** Higher weekend alcohol consumption is associated with lower average final grades.

5. **Digital access may support learning:** Internet access can be considered an important learning resource, especially for students who need online materials and academic support.
""")
st.markdown("""
This dashboard analyzes student academic performance using demographic, behavioral, and study-related variables. 
The goal is to explore which factors are associated with final grades and provide interactive visual insights for educational data analysis.
""")
st.info("Note: These insights show associations in the dataset and should not be interpreted as direct causal effects.")