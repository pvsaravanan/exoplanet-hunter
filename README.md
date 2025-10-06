# 🌌 ExoHunter AI: AI-Powered Exoplanet Discovery Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/your-username/exohunter-ai)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![AI Model](https://img.shields.io/badge/AI-XGBoost%20%7C%20Anomaly%20Detection-orange)


**ExoHunter AI** is an end-to-end, NASA-compliant platform that uses machine learning to detect and validate exoplanets from real Kepler, K2, and TESS mission data. Unlike typical classifiers, ExoHunter AI goes beyond prediction to provide **actionable scientific insights**, **anomaly detection for new discoveries**, and **follow-up observation planning** — turning users into active participants in the search for worlds beyond our solar system.

---

## 🔭 Features

✅ **NASA-Compliant Data**  
Trained exclusively on public data from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu) (Kepler KOI, K2, TESS TOI).

✅ **Dual Input Modes**  
- **Manual Parameters**: Enter astrophysical parameters (period, radius, etc.)
- **Raw Light Curves**: Upload FITS or CSV light curves for full processing

✅ **Scientific-Grade AI**  
- **XGBoost Classifier**: 98.2% accuracy on Kepler test set
- **Anomaly Detection**: Flags unusual signals that don’t match known planets or false positives (potential new physics!)
- **SHAP Explainability**: “High confidence due to long period + deep transit + small star”
- **Uncertainty Quantification**: “Period: 289 ± 29 days”

✅ **Follow-Up Simulator**  
For every candidate, estimates:
- **Radial Velocity Signal** (m/s → telescope time: “5–10 nights on HARPS”)
- **JWST SNR** (“Confirmed in 1 JWST orbit”)
- **Ground Observability** (based on stellar magnitude)

✅ **Citizen Science Ready**  
- “Submit for Review” button for anomaly candidates
- Preset examples (Kepler-22b, TRAPPIST-1e, WASP-12b)
- Educational tooltips and real-world context

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/pvsaravanan/exoplanet-hunter.git
cd exohunter-ai
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Data & Train Models
```bash
# Download NASA data (or use your own CSVs)
python prepare_data.py

# Train main classifier
python train_model.py

# Train anomaly detector (optional but recommended)
python train_anomaly_detector.py
```

### 4. Run the Web App
```bash
streamlit run app.py
```

Visit `http://localhost:8501` to start hunting for exoplanets!

---

## 📂 Project Structure

```
exohunter-ai/
├── app.py                  # Streamlit web interface
├── prepare_data.py         # Load & harmonize NASA datasets
├── train_model.py          # Train XGBoost classifier
├── train_anomaly_detector.py # Train Isolation Forest for anomalies
├── followup_estimator.py   # Physics-based follow-up simulator
├── lightcurve_processor.py # Lightkurve-based light curve analysis
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🧪 Sample Use Cases

### 🔍 Analyze Kepler-22b
1. Select **“Kepler-22b (Habitable Zone)”** from presets
2. Click **“Analyze for Exoplanet”**
3. See:  
   - ✅ **Likely Exoplanet!** (74.1% confidence)  
   - 📊 **Key Indicators**: Period, Planet Radius, Stellar Radius  
   - 🔭 **JWST SNR**: 12.3 → ✅ Confirmed in 1 orbit  
   - 🚩 **Estimated Period**: 289.9 ± 28.5 days

### 📤 Upload a Light Curve
1. Go to **“📁 Upload Light Curve”** tab
2. Upload `kepler-22_full.csv` (download via `download_kepler22.py`)
3. See folded transit + full scientific report

### 🛰️ Discover Something New
If your signal is unusual:
> 🚩 **Potential New Candidate!**  
> *This doesn’t match known planets or false positives — consider follow-up!*  
> → Click **“Submit for Review”** to contribute to discovery

---

## 📜 NASA Compliance

This project uses **only U.S. Government–hosted data** from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu).  
**Disclaimer**: *NASA does not endorse this application. Results are for educational/research assistance only.*

---

## 🌟 Why This Stands Out

| Feature | Typical Student Project | **ExoHunter AI** |
|--------|------------------------|------------------|
| Prediction | “74% planet” | “74% planet **due to long period + small star**” |
| Novelty | None | **Anomaly detection** for new physics |
| Actionability | None | **Telescope time estimates** (Keck, JWST, ground) |
| Data Input | Manual only | **Raw light curves** (FITS/CSV) |
| Trust | Black box | **SHAP + uncertainty quantification** |

---

## 🤝 Contributing

We welcome contributions! Whether you're a student, researcher, or citizen scientist:
1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📚 References

- NASA Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu  
- Lightkurve: https://lightkurve.org  
- Winn (2010): *Exoplanet Transits and Occultations* ([arXiv:1001.2010](https://arxiv.org/abs/1001.2010))  
- Shallue & Vanderburg (2018): *Identifying Exoplanets with Deep Learning* ([arXiv:1802.04452](https://arxiv.org/abs/1802.04452))

---

## 📧 Contact

Created by [Saravanan PV]  
[saravananpv30102005@gmail.com] | [Saveetha Engineering College]
*Let’s find the next Earth together.*

---

> “The universe is full of planets. We just need smarter eyes to see them.” — ExoHunter AI
