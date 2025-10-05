import pandas as pd

def load_and_clean_koi(file_path='koi.csv'):
    print("Loading Kepler KOI data...")
    df = pd.read_csv(file_path, low_memory=False, on_bad_lines='skip')
    df = df[df['koi_disposition'].isin(['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'])].copy()
    df['label'] = df['koi_disposition'].map({'CONFIRMED': 1, 'CANDIDATE': 1, 'FALSE POSITIVE': 0})
    
    features = ['koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_srad', 'koi_steff', 'koi_slogg']
    df = df[features + ['label']].dropna()
    df['mission'] = 'Kepler'
    print(f"  → {len(df)} valid Kepler rows")
    return df

def load_and_clean_toi(file_path='toi.csv'):
    print("Loading TESS TOI data...")
    df = pd.read_csv(file_path, low_memory=False, on_bad_lines='skip')
    
    # Map disposition
    df['label'] = df['tfopwg_disp'].map({'PC': 1, 'CP': 1, 'KP': 1, 'FP': 0})
    df = df.dropna(subset=['label']).copy()
    
    # Rename TESS columns to match Kepler style
    col_map = {
        'pl_orbper': 'koi_period',
        'pl_trandurh': 'koi_duration',   # ← TESS uses 'pl_trandurh' (hours)
        'pl_trandep': 'koi_depth',
        'pl_rade': 'koi_prad',
        'st_rad': 'koi_srad',
        'st_teff': 'koi_steff',
        'st_logg': 'koi_slogg'
    }
    df = df.rename(columns=col_map)
    
    features = ['koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_srad', 'koi_steff', 'koi_slogg']
    df = df[features + ['label']].dropna()
    df['mission'] = 'TESS'
    print(f"  → {len(df)} valid TESS rows")
    return df

def load_and_clean_k2(file_path='k2.csv'):
    print("Loading K2 data...")
    df = pd.read_csv(file_path, low_memory=False, on_bad_lines='skip')
    
    # Map disposition
    df['label'] = df['disposition'].map({'CONFIRMED': 1, 'CANDIDATE': 1, 'FALSE POSITIVE': 0})
    df = df.dropna(subset=['label']).copy()
    
    # Rename K2 columns
    col_map = {
        'pl_orbper': 'koi_period',
        # K2 does NOT have transit duration → skip it
        'pl_rade': 'koi_prad',
        'st_rad': 'koi_srad',
        'st_teff': 'koi_steff',
        'st_logg': 'koi_slogg'
    }
    df = df.rename(columns=col_map)
    
    # Only use features that exist (no duration)
    features = ['koi_period', 'koi_prad', 'koi_srad', 'koi_steff', 'koi_slogg']
    df = df[features + ['label']].dropna()
    df['mission'] = 'K2'
    print(f"  → {len(df)} valid K2 rows")
    return df

if __name__ == "__main__":
    try:
        koi = load_and_clean_koi('koi.csv')
        toi = load_and_clean_toi('toi.csv')
        k2 = load_and_clean_k2('k2.csv')
        
        # Align all DataFrames to the smallest common feature set
        # (Kepler has the most features; K2 has fewer)
        common_features = ['koi_period', 'koi_prad', 'koi_srad', 'koi_steff', 'koi_slogg']
        
        # Add missing columns as NaN for K2 (so concat works)
        for df in [koi, toi, k2]:
            for col in common_features:
                if col not in df.columns:
                    df[col] = pd.NA
        
        # Also ensure Kepler & TESS have only common features + label + mission
        koi = koi[common_features + ['label', 'mission']]
        toi = toi[common_features + ['label', 'mission']]
        k2 = k2[common_features + ['label', 'mission']]
        
        full_data = pd.concat([koi, toi, k2], ignore_index=True)
        full_data = full_data.dropna()  # Final dropna after alignment
        
        print(f"\n✅ Combined dataset: {len(full_data)} rows")
        print(f"   - Planets (1): {sum(full_data['label'] == 1)}")
        print(f"   - False Positives (0): {sum(full_data['label'] == 0)}")
        print(f"   - Features used: {list(common_features)}")
        
        full_data.to_csv("nasa_exoplanet_data_clean.csv", index=False)
        print("\n💾 Saved to 'nasa_exoplanet_data_clean.csv'")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()