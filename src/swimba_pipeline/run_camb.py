import typing as T

import camb
import numpy as np


@T.overload
def run_camb(params: camb.CAMBparams) -> T.Tuple[np.ndarray, camb.CAMBparams]: ...
@T.overload
def run_camb(
    *, Omega_b: float, Omega_cdm: float, h: float, ns: float
) -> T.Tuple[np.ndarray, camb.CAMBparams]: ...
def run_camb(
    params: T.Optional[camb.CAMBparams] = None,
    *,
    Omega_b: float = 0,
    Omega_cdm: float = 0,
    h: float = 0,
    ns: float = 0,
) -> T.Tuple[np.ndarray, camb.CAMBparams]:
    if params is None:
        params = camb.CAMBparams()
        # set accuracy of the calculation
        params.set_accuracy(
            AccuracyBoost=3.0,
            lSampleBoost=3.0,
            lAccuracyBoost=1.0,
            DoLateRadTruncation=True,
        )

        # set value of the cosmological parameters
        params.set_cosmology(
            H0=h * 100.0,
            ombh2=Omega_b * h**2,
            omch2=Omega_cdm * h**2,
            mnu=0,
            omk=0,
            neutrino_hierarchy="degenerate",
            num_massive_neutrinos=0,
            nnu=3.046,
            tau=None,
        )

        # set the value of the primordial power spectrum parameters
        params.InitPower.set_params(
            As=2.13e-9,
            ns=ns,
            pivot_scalar=0.05,
            pivot_tensor=0.05,
        )

        # set redshifts, k-range and k-sampling
        params.set_matter_power(
            redshifts=[0],
            kmax=200,
            k_per_logint=20,
        )

    # compute results
    results: camb.CAMBdata = camb.get_results(params)
    k, zs, Pkmm = results.get_matter_power_spectrum(
        minkh=2e-5,
        maxkh=200,
        npoints=400,
        var1=7,
        var2=7,
        have_power_spectra=True,
        params=None,
    )

    return np.transpose([k, Pkmm[0, :]]), params
