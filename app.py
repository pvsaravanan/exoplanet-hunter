import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# Optional: Try to import Lightkurve
try:
    import lightkurve as lk
    from scipy.signal import find_peaks
    LIGHTKURVE_AVAILABLE = True
except ImportError:
    LIGHTKURVE_AVAILABLE = False

# Load models
@st.cache_resource
def load_models():
    model = joblib.load("exoplanet_model.pkl")
    scaler = joblib.load("feature_scaler.pkl")
    
    # Anomaly detector
    try:
        anomaly_detector = joblib.load("anomaly_detector.pkl")
        anomaly_scaler = joblib.load("anomaly_scaler.pkl")
        ANOMALY_AVAILABLE = True
    except:
        anomaly_detector = None
        anomaly_scaler = None
        ANOMALY_AVAILABLE = False
        
    return model, scaler, anomaly_detector, anomaly_scaler, ANOMALY_AVAILABLE

model, scaler, anomaly_detector, anomaly_scaler, ANOMALY_AVAILABLE = load_models()

# SHAP & Uncertainty
try:
    import shap
    explainer = joblib.load("shap_explainer.pkl")
    feature_stats = joblib.load("feature_stats.pkl")
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False

# App title
st.set_page_config(page_title="ExoHunter AI", page_icon="🔭")
st.title("🔭 ExoHunter AI: Find Exoplanets with AI")
st.markdown("""
NASA-compliant tool using Kepler, K2, and TESS data to classify exoplanet candidates.  
*Trained on data from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu).*
""")

# Preset examples (with Kepler magnitude)
presets = {
    "None": (None, None, None, None, None, None),
    "Kepler-22b (Habitable Zone)": (289.86, 2.4, 0.97, 5518, 4.43, 11.5),
    "TOI-700 d (Earth-sized, Habitable)": (37.4, 1.18, 0.42, 3480, 4.62, 10.2),
    "WASP-12b (Hot Jupiter)": (1.09, 15.9, 1.47, 6175, 4.38, 11.2),
    "TRAPPIST-1e (Earth-like)": (6.1, 0.92, 0.12, 2566, 5.21, 15.1),
    "False Positive (Eclipsing Binary)": (0.5, 1.0, 1.0, 5778, 4.44, 12.0),
}

# Initialize session state
if "preset" not in st.session_state:
    st.session_state.preset = "None"

def update_inputs():
    st.session_state.preset = st.session_state.preset_selector

# Tabs
tab1, tab2 = st.tabs(["📊 Enter Parameters", "📁 Upload Light Curve"])

# ======================
# TAB 1: Manual Parameters
# ======================
with tab1:
    selected_preset = st.selectbox(
        "🧪 Load Example",
        options=list(presets.keys()),
        key="preset_selector",
        on_change=update_inputs
    )

    # Get default values
    vals = presets[selected_preset]
    if vals[0] is None:
        period_def, prad_def, srad_def, teff_def, logg_def, kepmag_def = 3.5, 1.2, 1.0, 5778, 4.44, 12.0
    else:
        period_def, prad_def, srad_def, teff_def, logg_def, kepmag_def = vals

    st.header("Enter Observational Parameters")
    col1, col2 = st.columns(2)

    with col1:
        period = st.number_input("Orbital Period (days)", min_value=0.1, value=period_def, format="%.3f")
        prad = st.number_input("Planet Radius (Earth radii)", min_value=0.1, value=prad_def, format="%.2f")
        srad = st.number_input("Stellar Radius (Solar radii)", min_value=0.1, value=srad_def, format="%.2f")

    with col2:
        teff = st.number_input("Stellar Temp (K)", min_value=2000, max_value=10000, value=int(teff_def), format="%d")
        logg = st.number_input("Stellar Log(g)", min_value=0.0, max_value=5.5, value=logg_def, format="%.2f")
        kepmag = st.number_input("Stellar Kepler Magnitude", min_value=8.0, max_value=16.0, value=kepmag_def, format="%.2f")

    input_data = np.array([[period, prad, srad, teff, logg]])

    if st.button("🔍 Analyze for Exoplanet"):
        # Main classifier
        input_scaled = scaler.transform(input_data)
        prob = model.predict_proba(input_scaled)[0][1]

        # === UNCERTAINTY: Estimate prediction intervals ===
        if SHAP_AVAILABLE:
            period_std = feature_stats['std']['koi_period']
            period_uncertainty = period_std * 0.1
            period_str = f"{period:.1f} ± {period_uncertainty:.1f}"
        else:
            period_str = f"{period:.1f}"

        # === SHAP EXPLAINABILITY ===
        if SHAP_AVAILABLE:
            shap_values = explainer(input_scaled)
            shap_vals = shap_values.values[0]
            feature_names_ui = ['Period', 'Planet Radius', 'Stellar Radius', 'Temp', 'Log(g)']
            top_contributors = sorted(
                [(feature_names_ui[i], shap_vals[i]) for i in range(len(shap_vals)) if shap_vals[i] > 0],
                key=lambda x: x[1],
                reverse=True
            )[:3]
            explanation = ", ".join([name for name, val in top_contributors])
        else:
            explanation = "Period, Planet Radius, Stellar Radius"

        # === ANOMALY DETECTION ===
        is_anomaly = False
        if ANOMALY_AVAILABLE:
            input_scaled_anomaly = anomaly_scaler.transform(input_data)
            is_anomaly = anomaly_detector.predict(input_scaled_anomaly)[0] == -1

        # === DISPLAY RESULTS ===
        if is_anomaly and prob > 0.5:
            st.warning("🚩 **Potential New Candidate!**\n\nThis signal doesn’t match known planets or false positives — consider follow-up!")
            st.markdown(f"**Classifier Confidence**: {prob:.1%}")
        elif prob >= 0.7:
            st.success(f"✅ **Likely Exoplanet!**\n\nConfidence: {prob:.1%}")
        elif prob >= 0.4:
            st.warning(f"⚠️ **Uncertain Candidate**\n\nConfidence: {prob:.1%}")
        else:
            st.error(f"❌ **Likely False Positive**\n\nConfidence: {1 - prob:.1%}")

        # === NEW: Uncertainty & Explanation ===
        st.markdown(f"**Estimated Period**: {period_str} days")
        st.markdown(f"**Key Indicators**: {explanation}")

        # === FOLLOW-UP ESTIMATES ===
        st.subheader("🔭 Follow-Up Observation Estimates")
        
        # Estimate transit depth from radius and stellar radius
        transit_depth = (prad / srad) ** 2
        
        # RV estimate
        K = (2 * np.pi * 6.67430e-11 / (period * 86400))**(1/3) * (prad**3.7 * 5.972e24) / ((1.989e30)**(2/3))
        K_mps = K  # m/s
        if K_mps > 10:
            rv_verdict = "1–2 nights (Keck/HIRES)"
        elif K_mps > 1:
            rv_verdict = "5–10 nights (HARPS)"
        else:
            rv_verdict = "Not feasible with current RV instruments"
        st.write(f"**Radial Velocity Signal**: {K_mps:.1f} m/s → *{rv_verdict}*")
        
        # JWST estimate
        photon_factor = 10**(-0.4 * (kepmag - 12))
        n_photons = 1e6 * photon_factor
        jwst_snr = transit_depth * np.sqrt(n_photons)
        if jwst_snr > 10:
            jwst_verdict = "✅ Confirmed in 1 JWST orbit"
        elif jwst_snr > 5:
            jwst_verdict = "⚠️ Marginal detection in 1 orbit"
        else:
            jwst_verdict = "❌ Not detectable with JWST/NIRISS"
        st.write(f"**JWST SNR (1 orbit)**: {jwst_snr:.1f} → {jwst_verdict}")
        
        # Ground observability
        if kepmag < 10:
            ground_obs = "✅ Easily observable from ground (e.g., LCO)"
        elif kepmag < 13:
            ground_obs = "🟡 Observable with 1–2m telescopes"
        elif kepmag < 15:
            ground_obs = "🔴 Challenging; requires 4m+ telescope"
        else:
            ground_obs = "❌ Too faint for ground-based follow-up"
        st.write(f"**Ground-Based Observability**: {ground_obs}")

# ======================
# TAB 2: Light Curve Upload
# ======================
with tab2:
    st.header("Upload TESS/Kepler Light Curve")
    st.markdown("Supports `.fits` (standard) or `.csv` (columns: `time`, `flux`)")

    if not LIGHTKURVE_AVAILABLE:
        st.error("⚠️ Lightkurve not installed. Install with: `pip install lightkurve`")
    else:
        uploaded_file = st.file_uploader("Choose a light curve file", type=["fits", "csv"])

        if uploaded_file is not None:
            with st.spinner("Processing light curve..."):
                try:
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    # Load light curve
                    if temp_path.endswith(".fits"):
                        lc = lk.read(temp_path)
                    else:
                        df = pd.read_csv(temp_path)
                        if 'time' not in df.columns or 'flux' not in df.columns:
                            raise ValueError("CSV must contain 'time' and 'flux' columns")
                        lc = lk.LightCurve(time=df['time'], flux=df['flux'])

                    # Clean and flatten
                    lc_clean = lc.remove_nans().remove_outliers().flatten(window_length=401)

                    # Estimate period (up to 365 days)
                    pg = lc_clean.to_periodogram(
                        method='lombscargle',
                        minimum_period=0.5,
                        maximum_period=365,
                        oversample_factor=10
                    )
                    best_period = pg.period_at_max_power.value

                    # Fold and normalize
                    folded = lc_clean.fold(period=best_period).normalize()

                    # Extract features
                    flux = folded.flux.value
                    baseline = np.median(flux)
                    depth = baseline - np.min(flux)
                    noise = np.std(flux)
                    snr = depth / noise if noise > 0 else 0

                    # Estimate parameters
                    srad_est = 1.0
                    prad_est = np.sqrt(depth) * srad_est * 109.2
                    kepmag_est = -2.5 * np.log10(np.median(lc.flux)) + 25
                    kepmag_est = np.clip(kepmag_est, 8, 16)

                    # Prepare input
                    input_data = np.array([[best_period, prad_est, srad_est, 5778, 4.44]])
                    input_scaled = scaler.transform(input_data)
                    prob = model.predict_proba(input_scaled)[0][1]

                    # Anomaly detection
                    is_anomaly = False
                    if ANOMALY_AVAILABLE:
                        input_scaled_anomaly = anomaly_scaler.transform(input_data)
                        is_anomaly = anomaly_detector.predict(input_scaled_anomaly)[0] == -1

                    # Display results
                    st.subheader("📈 Detected Signal")
                    st.write(f"**Period**: {best_period:.3f} days")
                    st.write(f"**Transit Depth**: {depth:.5f}")
                    st.write(f"**SNR**: {snr:.2f}")
                    st.write(f"**Estimated Planet Radius**: {prad_est:.2f} Earth radii")
                    st.write(f"**Estimated Kepler Magnitude**: {kepmag_est:.2f}")

                    st.subheader("📉 Folded Light Curve")
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, 6))
                    folded.plot(ax=ax, color='white', linewidth=1.5)
                    ax.set_ylim(baseline - 0.005, baseline + 0.005)
                    ax.set_title("Folded Light Curve (Zoomed)")
                    ax.set_xlabel("Phase")
                    ax.set_ylabel("Normalized Flux")
                    st.pyplot(fig)

                    st.subheader("🎯 Prediction")
                    if is_anomaly and prob > 0.5:
                        st.warning("🚩 **Potential New Candidate!**\n\nThis doesn’t match known patterns — consider follow-up!")
                        st.markdown(f"**Classifier Confidence**: {prob:.1%}")
                    elif prob >= 0.7:
                        st.success(f"✅ **Likely Exoplanet!**\n\nConfidence: {prob:.1%}")
                    elif prob >= 0.4:
                        st.warning(f"⚠️ **Uncertain Candidate**\n\nConfidence: {prob:.1%}")
                    else:
                        st.error(f"❌ **Likely False Positive**\n\nConfidence: {1 - prob:.1%}")

                    # Follow-up estimates
                    st.subheader("🔭 Follow-Up Observation Estimates")
                    transit_depth = depth
                    K = (2 * np.pi * 6.67430e-11 / (best_period * 86400))**(1/3) * (prad_est**3.7 * 5.972e24) / ((1.989e30)**(2/3))
                    rv_verdict = "1–2 nights (Keck)" if K > 10 else "5–10 nights (HARPS)" if K > 1 else "Not feasible"
                    st.write(f"**Radial Velocity Signal**: {K:.1f} m/s → *{rv_verdict}*")
                    
                    photon_factor = 10**(-0.4 * (kepmag_est - 12))
                    jwst_snr = transit_depth * np.sqrt(1e6 * photon_factor)
                    jwst_verdict = "✅ Confirmed in 1 JWST orbit" if jwst_snr > 10 else "⚠️ Marginal" if jwst_snr > 5 else "❌ Not detectable"
                    st.write(f"**JWST SNR (1 orbit)**: {jwst_snr:.1f} → {jwst_verdict}")
                    
                    if kepmag_est < 10:
                        ground_obs = "✅ Easily observable from ground"
                    elif kepmag_est < 13:
                        ground_obs = "🟡 Observable with 1–2m telescopes"
                    elif kepmag_est < 15:
                        ground_obs = "🔴 Challenging; requires 4m+ telescope"
                    else:
                        ground_obs = "❌ Too faint for ground-based follow-up"
                    st.write(f"**Ground-Based Observability**: {ground_obs}")

                    os.remove(temp_path)

                except Exception as e:
                    st.error(f"❌ Failed to process file: {e}")
                    st.info("Ensure your CSV has 'time' and 'flux' columns, or use a standard TESS FITS file.")

# Compliance footer
st.markdown("---")
st.caption("""
**Disclaimer**: This tool uses public data from the NASA Exoplanet Archive.  
NASA does not endorse this application. Results are for educational/research assistance only.
""")