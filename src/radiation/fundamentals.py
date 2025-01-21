import numpy as np

def planck_function(wavelength, temperature):
   
    # Constants
    h = 6.62607015e-34  # Planck constant (J·s)
    c = 2.99792458e8    # Speed of light (m/s)
    k = 1.380649e-23    # Boltzmann constant (J/K)

    # Convert wavelength to meters if necessary (e.g., from micrometers)
    wavelength_m = wavelength

    # Planck's law formula
    exponent = (h * c) / (wavelength_m * k * temperature)
    spectral_radiance = (2 * h * c**2) / (wavelength_m**5 * (np.exp(exponent) - 1))

    # Convert from W.m^-2.sr^-1.m^-1 to W.m^-2.sr^-1.um^-1
    spectral_radiance_um = spectral_radiance * 1e-6

    return spectral_radiance_um

def planck_inv(wavelength, spectral_radiance):
   
    # Constants
    h = 6.62607015e-34  # Planck constant (J·s)
    c = 2.99792458e8    # Speed of light (m/s)
    k = 1.380649e-23    # Boltzmann constant (J/K)

    # Convert spectral radiance from W.m^-2.sr^-1.um^-1 to W.m^-2.sr^-1.m^-1
    spectral_radiance_m = spectral_radiance * 1e6

    # Solve for temperature using Planck's law
    term1 = (2 * h * c**2) / (wavelength**5 * spectral_radiance_m)
    exponent = np.log(term1 + 1)
    temperature = (h * c) / (wavelength * k * exponent)

    return temperature

def brightness_temperature(radiance, band_center, band_width):
    """
    Calculate brightness temperature for a given rectangular bandpass.

    Implements numerical integration over a rectangular bandpass defined
    by its center wavelength and width. The bandpass is assumed to have
    unity transmission within its bounds and zero outside.

    Args:
        radiance (float): Observed radiance in W.m^-2.sr^-1.um^-1.
        band_center (float): Center wavelength of bandpass in meters.
        band_width (float): Width of bandpass in meters.

    Returns:
        float: Brightness temperature in Kelvin.

    Raises:
        ValueError: If band_width <= 0 or band_center <= band_width / 2.
    """
    if band_width <= 0 or band_center <= band_width / 2:
        raise ValueError("Invalid bandpass: band_width must be > 0 and band_center must be > band_width / 2.")

    # Define the wavelength range for the bandpass
    lower_bound = band_center - band_width / 2
    upper_bound = band_center + band_width / 2

    # Calculate the average temperature within the bandpass
    def integrand(wavelength):
        return planck_inv(wavelength, radiance)

    wavelengths = np.linspace(lower_bound, upper_bound, 1000)
    temperatures = integrand(wavelengths)

    brightness_temp = np.mean(temperatures)

    return brightness_temp

def radiance(temperature, band_center, band_width):
    """
    Calculate band-integrated radiance for a given temperature and rectangular bandpass.

    Integrates Planck function over a rectangular bandpass defined
    by its center wavelength and width. The bandpass is assumed to
    have unity transmission within its bounds and zero outside.

    Args:
        temperature (float): Temperature in Kelvin.
        band_center (float): Center wavelength of bandpass in meters.
        band_width (float): Width of bandpass in meters.

    Returns:
        float: Band-integrated radiance in W.m^-2.sr^-1.

    Raises:
        ValueError: If temperature <= 0, band_width <= 0, or band_center <= band_width / 2.
    """
    if temperature <= 0 or band_width <= 0 or band_center <= band_width / 2:
        raise ValueError("Invalid input: Check temperature, band_width, and band_center values.")

    # Define the wavelength range for the bandpass
    lower_bound = band_center - band_width / 2
    upper_bound = band_center + band_width / 2

    # Numerical integration of Planck's function over the bandpass
    wavelengths = np.linspace(lower_bound, upper_bound, 1000)
    radiances = planck_function(wavelengths, temperature)

    band_integrated_radiance = np.trapz(radiances, wavelengths)

    return band_integrated_radiance

def calculate_NEDT(temperature, NER, band_center, band_width):
    """
    Calculate the noise-equivalent differential temperature (NEDT).

    Uses numerical derivative of band-integrated radiance with respect
    to temperature to determine the temperature uncertainty corresponding
    to the NER.

    Args:
        temperature (float): Scene temperature in Kelvin.
        NER (float): Noise-equivalent radiance in W.m^-2.sr^-1.
        band_center (float): Center wavelength of bandpass in meters.
        band_width (float): Width of bandpass in meters.

    Returns:
        float: NEDT in Kelvin.

    Raises:
        ValueError: If temperature <= 0, NER <= 0, band_width <= 0, or
        band_center <= band_width / 2.
    """
    if temperature <= 0 or NER <= 0 or band_width <= 0 or band_center <= band_width / 2:
        raise ValueError("Invalid input: Check temperature, NER, band_width, and band_center values.")

    # Compute band-integrated radiance at the given temperature
    radiance_center = radiance(temperature, band_center, band_width)

    # Small perturbation to compute numerical derivative
    delta_temp = 1e-3  # Small temperature change in Kelvin
    radiance_perturbed = radiance(temperature + delta_temp, band_center, band_width)

    # Numerical derivative of radiance with respect to temperature
    dradiance_dtemp = (radiance_perturbed - radiance_center) / delta_temp

    # Calculate NEDT
    NEDT = NER / dradiance_dtemp

    return NEDT

# Example usage
if __name__ == "__main__":
    wavelength = 500e-9  # 500 nm in meters
    temperature = 5778  # Approximate temperature of the Sun in Kelvin
    radiance_value = planck_function(wavelength, temperature)
    print(f"Spectral radiance: {radiance_value} W·m^-2·sr^-1·um^-1")

    calculated_temp = planck_inv(wavelength, radiance_value)
    print(f"Calculated temperature: {calculated_temp} K")

    band_center = 500e-9  # 500 nm in meters
    band_width = 50e-9   # 50 nm in meters
    brightness_temp = brightness_temperature(radiance_value, band_center, band_width)
    print(f"Brightness temperature: {brightness_temp} K")

    band_radiance = radiance(temperature, band_center, band_width)
    print(f"Band-integrated radiance: {band_radiance} W·m^-2·sr^-1")

    NER = 1e-6  # Example noise-equivalent radiance in W·m^-2·sr^-1
    nedt = calculate_NEDT(temperature, NER, band_center, band_width)
    print(f"NEDT: {nedt} K")
