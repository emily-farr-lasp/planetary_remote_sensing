import numpy as np
from scipy.integrate import trapz

def integrate_opacity_limb(opacity_profile: np.ndarray,
                         altitude_grid: np.ndarray,
                         z1: float,
                         z2: float,
                         impact_parameter: float,
                         planet_radius: float) -> float:
    """Calculate integrated opacity along a limb path.
    
    This function implements numerical integration of opacity along a curved path
    through a spherically symmetric atmosphere. It uses the path geometry to properly
    account for the varying path length through each atmospheric layer.
    
    Args:
        opacity_profile: Vertical profile of opacity per unit length (1/m)
        altitude_grid: Altitudes corresponding to opacity profile (m)
        z1: Start altitude of path (m)
        z2: End altitude of path (m)
        impact_parameter: Closest approach distance from planet center (m)
        planet_radius: Planet radius (m)
    
    Returns:
        Integrated optical depth along limb path
    """
    # Create integration grid between start and end points
    n_points = 100  # Number of integration points
    z_int = np.linspace(z1, z2, n_points)
    
    # Calculate path length segments
    dz = np.diff(z_int)
    z_mid = (z_int[1:] + z_int[:-1]) / 2
    
    # Calculate path lengths through each segment using geometry
    # r² = (R + z)² = b² + x² where b is impact parameter
    x = np.sqrt((planet_radius + z_mid)**2 - impact_parameter**2)
    ds = dz * np.sqrt(1 + (x / (planet_radius + z_mid))**2)
    
    # Interpolate opacity profile to integration points
    opacity_interp = np.interp(z_mid, altitude_grid, opacity_profile)
    
    # Integrate opacity * path length
    tau = np.sum(opacity_interp * ds)
    
    return tau

def surf_transmission(opacity_profile: np.ndarray,
                     angle: float,
                     spacecraft_alt: float,
                     planet_radius: float,
                     altitude_grid: np.ndarray,
                     direction: str = 'from') -> float:
    """Calculate transmission along path to/from surface.
    
    This function handles both upward and downward paths between the surface
    and spacecraft, accounting for atmospheric curvature effects.
    
    Args:
        opacity_profile: Vertical profile of opacity per unit length (1/m)
        angle: Surface incidence/emission angle (radians)
        spacecraft_alt: Altitude of spacecraft (m)
        planet_radius: Planet radius (m)
        altitude_grid: Altitudes corresponding to opacity profile (m)
        direction: Either 'to' (downward) or 'from' (upward) surface
    
    Returns:
        Total transmission along the path
    """
    # Calculate impact parameter from surface angle
    if direction == 'from':
        # For upward paths, starting at surface
        impact_parameter = planet_radius * np.sin(angle)
        z1 = 0  # Start at surface
        z2 = spacecraft_alt
    else:
        # For downward paths, starting at spacecraft
        impact_parameter = (planet_radius + spacecraft_alt) * np.sin(np.pi - angle)
        z1 = spacecraft_alt
        z2 = 0
    
    # Calculate integrated opacity
    tau = integrate_opacity_limb(opacity_profile, altitude_grid, z1, z2,
                               impact_parameter, planet_radius)
    
    # Convert to transmission
    transmission = np.exp(-tau)
    
    return transmission

def limb_transmission(opacity_profile: np.ndarray,
                     tangent_alt: float,
                     spacecraft_alt: float,
                     planet_radius: float,
                     altitude_grid: np.ndarray) -> float:
    """Calculate transmission for limb path through curved atmosphere.
    
    Implements limb path transmission calculation for atmospheric sounding,
    where the line of sight passes through the atmosphere without hitting
    the surface.
    
    Args:
        opacity_profile: Vertical profile of opacity per unit length (1/m)
        tangent_alt: Tangent altitude of line of sight (m)
        spacecraft_alt: Altitude of spacecraft (m)
        planet_radius: Planet radius (m)
        altitude_grid: Altitudes corresponding to opacity profile (m)
    
    Returns:
        Total transmission along the limb path
    """
    # Impact parameter is distance of closest approach to planet center
    impact_parameter = planet_radius + tangent_alt
    
    # Calculate integrated opacity along the limb path
    tau = integrate_opacity_limb(opacity_profile, altitude_grid, tangent_alt,
                               spacecraft_alt, impact_parameter, planet_radius)
    
    # Double the opacity since path goes through atmosphere twice
    tau = 2.0 * tau
    
    # Convert to transmission
    transmission = np.exp(-tau)
    
    return transmission

def cloud_visible_brightness(surface_albedo: float,
                           solar_flux: float,
                           solar_zenith: float,
                           emission_angle: float,
                           azimuth_angle: float,
                           optical_depth: float,
                           single_scatter_albedo: float,
                           asymmetry_parameter: float) -> float:
    """Calculate cloud brightness at visible wavelengths using single-scattering.
    
    Implements the single-scattering approximation for cloud brightness including
    both direct scattered sunlight and surface-reflected components.
    
    Args:
        surface_albedo: Surface Lambert albedo (0-1)
        solar_flux: Incident solar flux at top of atmosphere (W/m2)
        solar_zenith: Solar zenith angle (radians)
        emission_angle: Viewing angle from nadir (radians)
        azimuth_angle: Relative azimuth between sun and viewing direction (radians)
        optical_depth: Cloud optical depth
        single_scatter_albedo: Single scattering albedo (0-1)
        asymmetry_parameter: Asymmetry parameter g (-1 to 1)
    
    Returns:
        Cloud brightness in W/m2/sr
    """
    # Calculate cosines of angles
    mu0 = np.cos(solar_zenith)
    mu = np.cos(emission_angle)
    
    # Calculate scattering angle using spherical trigonometry
    cos_theta = -mu * mu0 + np.sqrt(1 - mu**2) * np.sqrt(1 - mu0**2) * np.cos(azimuth_angle)
    
    # Calculate Henyey-Greenstein phase function
    g = asymmetry_parameter
    P = (1 - g**2) / (1 + g**2 - 2*g*cos_theta)**(3/2)
    
    # Calculate modified optical depth for combined path
    tau_star = optical_depth/mu0 + optical_depth/mu
    
    # Direct scattered component
    I_direct = (solar_flux * mu0 * single_scatter_albedo * P / (4*np.pi)) * (1 - np.exp(-tau_star))
    
    # Surface reflected component
    I_surface = (surface_albedo * solar_flux * mu0 / np.pi) * np.exp(-tau_star)
    
    # Total brightness
    I_total = I_direct + I_surface
    
    return I_total
