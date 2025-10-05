import download_lightcurve as lk
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings("ignore")

def process_tess_lightcurve(file_path):
    """
    Load and process a TESS light curve file (FITS or CSV).
    Returns: dict with extracted features + normalized flux array
    """
    try:
        # Load light curve
        if file_path.endswith('.fits'):
            lc = lk.read(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            # Assume columns: time, flux, flux_err
            lc = lk.LightCurve(time=df['time'], flux=df['flux'])
        else:
            raise ValueError("Unsupported file format. Use .fits or .csv")
        
        # Detrend using median filter (simple but effective)
        lc_clean = lc.flatten(window_length=501)
        
        # Find period candidates (using Lomb-Scargle)
        periodogram = lc_clean.to_periodogram(method='lombscargle', minimum_period=0.5, maximum_period=50)
        best_period = periodogram.period_at_max_power.value
        
        # Phase fold on best period
        folded = lc_clean.fold(period=best_period)
        
        # Extract features
        features = {
            'period': best_period,
            'duration': estimate_transit_duration(folded),
            'depth': estimate_transit_depth(folded),
            'snr': estimate_snr(folded),
            'num_transits': len(find_peaks(-folded.flux)[0]),  # number of dips
            'mean_flux': np.mean(lc_clean.flux),
            'std_flux': np.std(lc_clean.flux),
            'skew_flux': np.skew(lc_clean.flux),
            'kurtosis_flux': np.kurtosis(lc_clean.flux)
        }
        
        # Return features + folded flux for CNN (if needed)
        return {
            'features': features,
            'folded_flux': folded.flux.value[:201],  # Fixed length for CNN
            'period': best_period,
            'plot_data': folded
        }
        
    except Exception as e:
        raise RuntimeError(f"Error processing light curve: {e}")

def estimate_transit_duration(folded_lc):
    """Simple estimation: width of deepest dip"""
    flux = folded_lc.flux.value
    time = folded_lc.time.value
    
    # Find deepest dip
    min_idx = np.argmin(flux)
    half_width = 10  # pixels around min
    window = flux[max(0, min_idx-half_width):min(len(flux), min_idx+half_width)]
    
    # Estimate duration as FWHM-like
    if len(window) < 5:
        return 0.1  # fallback
    
    # Simple: time between 25% and 75% of depth
    depth = np.min(window)
    baseline = np.median(flux)
    threshold = baseline - 0.5 * (baseline - depth)
    
    above_threshold = flux > threshold
    if np.any(above_threshold):
        left = np.where(above_threshold[:min_idx])[0]
        right = np.where(above_threshold[min_idx:])[0] + min_idx
        if len(left) > 0 and len(right) > 0:
            return time[right[0]] - time[left[-1]]
    return 0.1

def estimate_transit_depth(folded_lc):
    """Depth = max drop from baseline"""
    flux = folded_lc.flux.value
    baseline = np.median(flux)
    depth = baseline - np.min(flux)
    return depth

def estimate_snr(folded_lc):
    """Signal-to-noise ratio of transit"""
    flux = folded_lc.flux.value
    baseline = np.median(flux)
    depth = baseline - np.min(flux)
    noise = np.std(flux)
    return depth / noise if noise > 0 else 0.0