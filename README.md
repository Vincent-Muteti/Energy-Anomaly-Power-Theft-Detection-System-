# Energy Anomaly & Automated Power Theft Detection System  
### A Machine Learning–Driven Time-Series Intelligence Framework for Smart Grid Inspection Prioritization

---

## Overview

Electricity utilities face significant revenue losses due to non-technical losses arising from electricity theft, meter tampering, irregular consumption behavior, and undetected abnormal demand shifts.

While smart meters generate large volumes of time-series data, many utilities lack structured machine learning systems capable of converting raw consumption signals into actionable inspection alerts.

This project develops an end-to-end machine learning and time-series anomaly detection framework designed to:

- Detect abnormal electricity usage patterns  
- Prioritize high-risk meters for inspection  
- Generate structured human-readable alerts  
- Export investigation-ready inspection reports  

The system transitions from raw smart meter time-series data to an operational inspection prioritization engine.

---

## Project Objectives

This project seeks to answer the following core questions:

1. How can smart meter time-series data be transformed into structured behavioral features capable of distinguishing normal electricity usage from suspicious consumption patterns?

2. Can machine learning techniques—specifically Isolation Forest and Random Forest—effectively detect and validate electricity theft–like behavior under both unlabeled and benchmark conditions?

3. How can anomaly scores be aggregated, normalized, and translated into interpretable risk indicators that support reliable inspection prioritization?

4. Is it possible to design a scalable, deployable anomaly detection system that converts raw consumption data into automated, investigation-ready alerts for utility operations?

---

## Project Architecture

The framework is structured in two major phases.

---

## Phase A — Research & Model Calibration

Benchmark datasets were used to validate modeling strategies under both labeled and unlabeled conditions.

### Techniques Employed

- Isolation Forest (Unsupervised anomaly detection)
- Random Forest Classifier (Supervised theft classification)
- Feature importance analysis
- Precision@K inspection simulation
- Per-entity modeling for distribution-aware detection

### Key Research Outcomes

- Random Forest ROC-AUC ≈ 0.945  
- Theft detection F1-score ≈ 0.91  
- Precision@1% inspection simulation ≈ 56%  
- Significant performance improvement using per-entity modeling  

This phase ensured that the anomaly modeling strategy was statistically validated before deployment.

---

## Phase B — Production Deployment Simulation

The calibrated anomaly detection methodology was applied to a structured multi-household electricity dataset integrated with contextual weather variables.

### Data Layers Integrated

**1. Electricity Consumption Layer**
- Daily aggregated smart meter statistics  
- Structured behavioral indicators  

**2. Weather Context Layer**
- Temperature (tmax, tmin)  
- Precipitation  
- Wind speed  

**3. Machine Learning Risk Layer**
- Per-meter Isolation Forest modeling  
- Rolling 30-day behavioral baselines  
- Residual and z-score deviation features  
- Global anomaly score thresholding  

---

## Machine Learning & Time-Series Methodology

### Time-Series Feature Engineering

- 30-day rolling consumption baselines  
- Rolling standard deviations  
- Residual deviations from historical behavior  
- Standardized anomaly indicators (z-scores)  
- Volatility metrics  

### Unsupervised Anomaly Detection

- Per-meter Isolation Forest training  
- Global 2% anomaly score threshold  
- Natural variation across meters  
- Persistent anomaly streak detection  

### Supervised Benchmark Validation

- Random Forest theft classification  
- ROC-AUC evaluation  
- Feature importance extraction  

### Risk Scoring & Prioritization

- Total anomalous days per meter  
- Longest anomaly streak  
- Worst anomaly score  
- Normalized risk score (0–100)  
- Risk categorization: Low / Medium / High  
- Ranked inspection prioritization list  
- Automated alert generation  
- Structured CSV export for operational workflows  

---

## Final System Outputs

The final system produces:

- Meter-level anomaly detection  
- Normalized risk scoring (0–100 scale)  
- Structured inspection prioritization  
- Persistent anomaly analysis  
- Automated human-readable alert messages  
- Export-ready inspection report suitable for utility workflows  

The system functions as an inspection prioritization engine rather than a static fraud classifier.

---

## Business Problem Addressed

Electricity utilities face persistent non-technical losses due to:

- Electricity theft  
- Meter tampering  
- Irregular consumption behavior  
- Inefficient manual inspection processes  

Traditional fraud detection approaches are:

- Reactive  
- Rule-based  
- Labor-intensive  
- Operationally costly  

Although smart meters generate large volumes of time-series data, most utilities lack structured machine learning systems capable of:

- Distinguishing legitimate variability from suspicious behavior  
- Adjusting for contextual influences such as weather  
- Prioritizing high-risk customers  
- Automatically generating inspection-ready alerts  

This project addresses the central question:

How can utilities leverage machine learning and time-series analysis to proactively detect abnormal electricity behavior and automatically generate structured inspection prioritization reports?

The framework shifts utilities from reactive inspection models to proactive, data-driven anomaly intelligence.

---

## Key Insights

- Per-entity modeling significantly improves anomaly detection performance.  
- Global anomaly thresholding produces realistic inspection variability.  
- Contextual weather integration reduces false positive risk.  
- Supervised classification validates consumption feature separability.  
- Structured risk scoring bridges analytics and operational decision-making.  

---

## Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-Learn  
- Matplotlib  
- Isolation Forest  
- Random Forest  
- Time-Series Feature Engineering  

---

## Dataset Disclaimer

The deployment dataset contains simulated meter identifiers and represents a structured smart meter environment for modeling demonstration purposes.

The architecture, anomaly detection methodology, and risk scoring framework are transferable to real-world utility systems subject to validation using official operational data.

---

## Conclusion

This project demonstrates how machine learning and time-series analysis can power a scalable electricity anomaly detection framework capable of:

- Detecting abnormal consumption behavior  
- Prioritizing high-risk meters  
- Generating automated inspection alerts  
- Exporting structured investigation reports  

By integrating unsupervised anomaly detection, supervised benchmarking, contextual adjustment, rolling time-series modeling, and automated reporting, the system provides a robust and deployable inspection prioritization framework suitable for smart grid environments.

The framework illustrates how utilities can transition from reactive inspection practices toward proactive, machine learning–driven anomaly intelligence supported by structured alert generation.

## Links
- Website : https://energy-anomaly-power-theft-detection-system-usho3gs893bcxfpikw.streamlit.app/ 
- Slides : https://www.canva.com/design/DAHCnDdv5YI/EY3460EEZG5d6N39tioGnQ/edit?ui=eyJBIjp7fX0