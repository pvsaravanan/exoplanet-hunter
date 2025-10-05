# # download_lightcurve.py
# import lightkurve as lk

# search_result = lk.search_lightcurve("Kepler-22", mission="Kepler", exptime=1800)
# lc = search_result[0].download()

# # Add overwrite=True to replace existing files
# lc.to_fits("kepler-22.fits", overwrite=True)
# lc.to_csv("kepler-22.csv", overwrite=True)

# print(f"✅ Saved light curve from Quarter {lc.quarter}")

# download_kepler22_full.py
import lightkurve as lk

print("🔍 Searching for Kepler-22 data...")
search_result = lk.search_lightcurve("Kepler-22", mission="Kepler", exptime=1800)

if len(search_result) == 0:
    raise ValueError("No Kepler-22 data found. Try 'Kepler-22' or 'KIC 10593626'")

print(f"✅ Found {len(search_result)} light curves. Downloading and stitching...")
lc_collection = search_result.download_all()
lc_stitched = lc_collection.stitch()

# Save as CSV
lc_stitched.to_csv("kepler-22_full.csv", overwrite=True)

print(f"🎉 Saved kepler-22_full.csv with {len(lc_stitched)} data points!")
print(f"   Time range: {lc_stitched.time[0]:.1f} to {lc_stitched.time[-1]:.1f} days")
print(f"   Expected period: ~289.9 days (Kepler-22b)")