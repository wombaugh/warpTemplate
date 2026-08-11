#!/usr/bin/env python3
"""Calculate Type Ia SN magnitude difference between two phases using sncosmo."""

import sncosmo


def mag_diff_between_phases(
    source_name: str = "salt2",
    band: str = "sdss::r",
    z: float = 0.0,
    phase1: float = 0.0,
    phase2: float = 20.0,
    x1: float = 0.0,
    c: float = 0.0,
    hostebv: float = 0.0,
    hostr_v: float = 3.1,
) -> float:
    """
    Calculate magnitude difference with host and MW dust extinction.

    Parameters
    ----------
    hostebv, hostr_v : float
        Host galaxy E(B-V) and R_V (CCM89 law)
    """
    # Initialize source
    source = sncosmo.get_source(source_name)

    # Create model with dust effects
    dust = sncosmo.CCM89Dust()
    model = sncosmo.Model(
        source=source,
        effects=[dust],
        effect_names=['host'],
        effect_frames=['rest']
    )

    # Set parameters
    model.update({
        'z': z,
        'hostebv': hostebv,
        'hostr_v': hostr_v,
    })

    # SALT2/SALT3 parameters
    if source_name.startswith(("salt2", "salt3")):
        model.update({'x1': x1, 'c': c})

    # Calculate magnitudes
    mag1 = model.bandmag(band, 'ab', phase1)
    mag2 = model.bandmag(band, 'ab', phase2)

    return mag2 - mag1


def main():

    # Parameters for demonstration
    source_name = "salt2"
    band = "sdss::r"
    z = 0.1
    phase1 = 0.0
    phase2 = 20.0
    hostebv = 0.1

    # Standard candle with moderate host extinction
    dm = mag_diff_between_phases(
        source_name=source_name,
        band=band,
        z=z,
        phase1=phase1,
        phase2=phase2,
        hostebv=hostebv,
        hostr_v=2.5,  # typical SN Ia host value
    )
    print(f"Magnitude diff for a SN Ia at z {z} with host E(B-V) {hostebv} between phase {phase1} and {phase2}: Δm = {dm:+.3f} mag")


    print("\n--- Impact of host E(B-V) on Δm(0→20d) when changing host extinction---")

    # Demonstrate extinction impact on color evolution
    print("\n--- Impact of host E(B-V) on Δm(0→20d) ---")
    for hostebv in [0.0, 0.1, 0.3, 0.5]:
        dm = mag_diff_between_phases(
            band="sdss::r",
            phase1=0.0,
            phase2=20.0,
            hostebv=hostebv
        )
        print(f"  hostebv={hostebv:.1f}: Δm = {dm:+.3f}")

    # Wavelength-dependent extinction: bluer bands dim more
    print("\n--- Band-dependent Δm with hostebv=0.2 ---")
    for band in ["sdss::u", "sdss::g", "sdss::r", "sdss::i"]:
        dm = mag_diff_between_phases(band=band, phase1=0.0, phase2=20.0, hostebv=0.2)
        print(f"  {band:12s}: Δm = {dm:+.3f}")


if __name__ == "__main__":
    main()