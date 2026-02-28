import streamlit as st

st.set_page_config(page_title="About", layout="wide")

st.title("About")
st.write(
    "This application detects abnormal electricity consumption behavior and generates "
    "inspection-prioritization reports using time-series feature engineering and anomaly detection."
)

st.subheader("FAQ")

with st.expander("Why use anomaly detection instead of a theft classifier?"):
    st.write(
        "Because confirmed theft labels are often unavailable. Anomaly detection helps prioritize "
        "suspicious behavior without requiring labeled fraud cases."
    )

with st.expander("Does this system confirm electricity theft?"):
    st.write(
        "No. It produces inspection-priority signals. Confirmation requires operational validation "
        "and field verification."
    )

with st.expander("What does the risk score (0–100) mean?"):
    st.write(
        "It is a normalized inspection priority indicator derived from anomaly severity and frequency. "
        "Higher scores indicate higher priority for inspection."
    )

with st.expander("Why include weather features?"):
    st.write(
        "Weather affects legitimate demand. Adding temperature/precipitation context helps reduce "
        "false positives caused by normal environmental variation."
    )

with st.expander("What is the global threshold?"):
    st.write(
        "A single cutoff applied across all meters to flag the most anomalous days overall, allowing "
        "realistic variation in anomaly frequency across customers."
    )