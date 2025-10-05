# lightcurve_processor.py
import lightkurve as lk
import numpy as np
from scipy.signal import find_peaks

def process_tess_lightcurve(file_path):
    """
    Load and process a TESS/Kepler light curve file (FITS or CSV).
    Returns: dict with extracted features + plot data
    """
    try:
        # Load light curve
        if file_path.endswith('.fits'):
            lc = lk.read(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            if 'time' not in df.columns or 'flux' not in df.columns:
                raise ValueError("CSV must contain 'time' and 'flux' columns")
            lc = lk.LightCurve(time=df['time'], flux=df['flux'])
        else:
            raise ValueError("Unsupported file format. Use .fits or .csv")

        # Clean and flatten
        lc_clean = lc.remove_nans().flatten(window_length=401)

        # Estimate period via Lomb-Scargle
        pg = lc_clean.to_periodogram(method='lombscargle', minimum_period=0.5, maximum_period=30)
        best_period = pg.period_at_max_power.value

        # Fold
        folded = lc_clean.fold(period=best_period)

        # Extract features
        flux = folded.flux.value
        baseline = np.median(flux)
        depth = baseline - np.min(flux)
        noise = np.std(flux)
        snr = depth / noise if noise > 0 else 0

        # Estimate duration
        threshold = baseline - 0.5 * depth
        in_transit = flux < threshold
        if np.any(in_transit):
            transit_indices = np.where(in_transit)[0]
            duration = folded.time.value[transit_indices[-1]] - folded.time.value[transit_indices[0]]
        else:
            duration = 0.1

        # Return results
        return {
            'features': {
                'period': best_period,
                'duration': duration,
                'depth': depth,
                'snr': snr,
                'mean_flux': np.mean(lc_clean.flux),
                'std_flux': np.std(lc_clean.flux)
            },
            'plot_data': folded,
            'period': best_period
        }

    except Exception as e:
        raise RuntimeError(f"Error processing light curve: {e}")