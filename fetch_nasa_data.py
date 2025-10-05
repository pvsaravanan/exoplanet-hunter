import pandas as pd
import numpy as np
import requests
from io import StringIO

# Base TAP URL
TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

def query_nasa_table(table_name):
    """
    Query NASA Exoplanet Archive TAP service and return DataFrame.
    """
    query = f"select * from {table_name}"
    params = {
        'query': query,
        'format': 'csv'
    }
    print(f"Fetching {table_name} data from NASA...")
    response = requests.get(TAP_URL, params=params, timeout=60)
    response.raise_for_status()  # Raises an HTTPError if status != 200
    return pd.read_csv(StringIO(response.text), low_memory=False)

def fetch_and_clean_koi():
    df = query_nasa_table('koi')
    valid_labels = ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']
    df = df[df['koi_disposition'].isin(valid_labels)].copy()
    df['label'] = df['koi_disposition'].map({
        'CONFIRMED': 1,
        'CANDIDATE': 1,
        'FALSE POSITIVE': 0
    })
    features = [
        'koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 
        'koi_srad', 'koi_teq', 'koi_insol', 'koi_model_snr',
        'koi_steff', 'koi_slogg'
    ]
    df = df[features + ['label']].dropna()
    df['mission'] = 'Kepler'
    return df

def fetch_and_clean_toi():
    df = query_nasa_table('toi')
    
    # Find disposition column
    disp_col = None
    for col in df.columns:
        if 'disp' in col.lower() and 'tfop' in col.lower():
            disp_col = col
            break
    if disp_col is None:
        # Fallback: use any column with 'disp'
        for col in df.columns:
            if 'disp' in col.lower():
                disp_col = col
                break
    if disp_col is None:
        raise KeyError("Disposition column not found in TOI data")
    
    label_map = {'PC': 1, 'CP': 1, 'KP': 1, 'FP': 0}
    df['label'] = df[disp_col].map(label_map)
    df = df.dropna(subset=['label']).copy()
    
    # Standardize column names to KOI style
    col_map = {
        'pl_orbper': 'koi_period',
        'pl_trandur': 'koi_duration',
        'pl_trandep': 'koi_depth',
        'pl_rade': 'koi_prad',
        'st_rad': 'koi_srad',
        'st_teff': 'koi_steff',
        'st_logg': 'koi_slogg'
    }
    df = df.rename(columns=col_map)
    
    features = ['koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 
                'koi_srad', 'koi_steff', 'koi_slogg']
    df = df[features + ['label']].dropna()
    df['mission'] = 'TESS'
    return df

def fetch_and_clean_k2():
    df = query_nasa_table('k2')
    
    disp_col = None
    for col in df.columns:
        if 'archive_disp' in col.lower():
            disp_col = col
            break
    if disp_col is None:
        for col in df.columns:
            if 'disp' in col.lower():
                disp_col = col
                break
    if disp_col is None:
        raise KeyError("Disposition column not found in K2 data")
    
    df['label'] = df[disp_col].map({
        'CONFIRMED': 1,
        'CANDIDATE': 1,
        'FALSE POSITIVE': 0
    })
    df = df.dropna(subset=['label']).copy()
    
    features = ['koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 
                'koi_srad', 'koi_steff', 'koi_slogg']
    df = df[features + ['label']].dropna()
    df['mission'] = 'K2'
    return df

if __name__ == "__main__":
    try:
        koi = fetch_and_clean_koi()
        print(f"Kepler: {len(koi)} rows")
        
        toi = fetch_and_clean_toi()
        print(f"TESS: {len(toi)} rows")
        
        k2 = fetch_and_clean_k2()
        print(f"K2: {len(k2)} rows")
        
        full_data = pd.concat([koi, k2, toi], ignore_index=True)
        full_data.to_csv("nasa_exoplanet_data.csv", index=False)
        print(f"\n✅ Total dataset saved: {len(full_data)} rows")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Tip: If you're behind a firewall or having network issues,")
        print("   try downloading the CSVs manually from:")
        print("   - KOI: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=koi")
        print("   - TOI: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=toi")
        print("   Then click 'Download' → 'CSV'")