import numpy as np

def estimate_rv_semi_amplitude(period_days, planet_radius_earth, stellar_mass=1.0, stellar_radius=1.0):
    """
    Estimate RV semi-amplitude K (m/s) using Winn (2010) Eq. 15.
    
    Parameters:
    - period_days: orbital period (days)
    - planet_radius_earth: planet radius (Earth radii)
    - stellar_mass: stellar mass (solar masses)
    - stellar_radius: stellar radius (solar radii)
    
    Returns:
    - K: RV semi-amplitude in m/s
    - telescope_nights: estimated nights needed on Keck/HARPS
    """
    # Convert to SI
    P = period_days * 86400  # seconds
    R_p = planet_radius_earth * 6.371e6  # meters
    R_s = stellar_radius * 6.96e8  # meters
    M_s = stellar_mass * 1.989e30  # kg
    
    # Estimate planet mass via mass-radius relation (Chen & Rogers 2016)
    if planet_radius_earth < 1.5:
        # Rocky: M ~ R^3.7
        M_p = (planet_radius_earth ** 3.7) * 5.972e24  # kg
    elif planet_radius_earth < 4.0:
        # Gassy: M ~ R^2.06
        M_p = (planet_radius_earth ** 2.06) * 5.972e24
    else:
        # Gas giant: M ~ R^0.0 (Jupiter-like)
        M_p = 317.8 * 5.972e24  # Jupiter mass
    
    # RV semi-amplitude (Winn 2010, Eq. 15)
    G = 6.67430e-11
    K = (2 * np.pi * G / P)**(1/3) * (M_p * np.sin(np.pi/2)) / (M_s**(2/3)) / np.sqrt(1 - 0**2)
    K_mps = K  # m/s
    
    # Estimate telescope time (empirical)
    if K_mps > 10:
        nights = "1–2 nights (Keck/HIRES)"
    elif K_mps > 1:
        nights = "5–10 nights (HARPS)"
    else:
        nights = "Not feasible with current RV instruments"
    
    return K_mps, nights

def estimate_jwst_snr(transit_depth, stellar_kepmag, n_orbits=1):
    """
    Estimate JWST/NIRISS SNR for transit spectroscopy.
    Approximate: SNR ~ (transit depth) * sqrt(N_photons)
    N_photons ~ 10^6 * 10^(-0.4 * (kepmag - 12)) per orbit
    """
    # Photon count scaling (very approximate)
    photon_factor = 10**(-0.4 * (stellar_kepmag - 12))
    n_photons = 1e6 * photon_factor * n_orbits
    snr = transit_depth * np.sqrt(n_photons)
    
    if snr > 10:
        verdict = f"✅ Confirmed in {n_orbits} JWST orbit(s)"
    elif snr > 5:
        verdict = f"⚠️ Marginal detection in {n_orbits} orbit(s)"
    else:
        verdict = "❌ Not detectable with JWST/NIRISS"
    
    return snr, verdict

def estimate_ground_observability(stellar_kepmag):
    """
    Simple ground-based observability based on magnitude.
    """
    if stellar_kepmag < 10:
        return "✅ Easily observable from ground (e.g., LCO, Las Cumbres)"
    elif stellar_kepmag < 13:
        return "🟡 Observable with 1–2m telescopes"
    elif stellar_kepmag < 15:
        return "🔴 Challenging; requires 4m+ telescope"
    else:
        return "❌ Too faint for ground-based follow-up"