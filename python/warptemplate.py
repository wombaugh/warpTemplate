import os
import re
import pickle
import random
import logging
from typing import List, Dict, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import sncosmo
from sncosmo.fitting import DataQualityError
from astropy.table import Table

from pymongo.database import Database
from scipy.interpolate import RectBivariateSpline as Spline2d
from scipy.interpolate import make_smoothing_spline

from sncosmo import TimeSeriesSource

from ampel.ztf.util.ZTFIdMapper import ZTFIdMapper
from ampel.ztf.view.ZTFFPTabulator import ZTFFPTabulator



def get_warpedTimeSeriesModel( name: str, original_template_name: str, warpdata: dict, 
                               z: None | float = None, hostr_v: None | float = 3.1, 
                                original_template_version: None | str = None, version=None) -> sncosmo.Model:
    """
        Create a sncosmo Model based on a warped TimeSeriesSource and using warp data from warplibrary.        
    """

    ws = WarpedTimeSeriesSource(
        phase=warpdata['corrmodel']['phase'],
        wave=warpdata['corrmodel']['wave'],
        flux=warpdata['corrmodel']['flux'],
        original_template_name=original_template_name,
        original_template_version=original_template_version,
        time_spline_degree=3,
        name=name,
        version=version
    )

    dust = sncosmo.CCM89Dust()
    wm = sncosmo.Model(source=ws,
                        effects=[dust],
                        effect_names=['host'],
                        effect_frames=['rest'])
    if hostr_v is not None:
        wm.set(hostr_v = hostr_v)
    if z is not None:
        wm.set(z=z)

    return wm


class WarpedTimeSeriesSource( TimeSeriesSource ):
    """
    Load existing, named TimeSeriesSource and warp according to input (phase,wave,flux).
    """

    def __init__(self, phase, wave, flux, 
                 original_template_name, original_template_version=None,
                 cut_negative_flux=True,
                 time_spline_degree=3, name=None, version=None):
        
        self.name = name
        self.version = version
        self._phase = phase
        self._wave = wave
        self._parameters = np.array([1.])
        self._original_source = sncosmo.get_source( original_template_name, original_template_version)
        self._warp_spline = Spline2d(phase, wave, flux, kx=time_spline_degree,
                                    ky=3)
        self._zero_before = True # Assume this should always be the case here

        # Create phases to use with the warped template
        self._phase = self._original_source._phase[( 
                ( self._original_source._phase >= phase.min() ) & 
                ( self._original_source._phase <= phase.max() )
                 )
        ]
        self._wave = self._original_source._wave[( 
                ( self._original_source._wave >= min(wave) ) & 
                ( self._original_source._wave <= max(wave) )
                 )
        ]
        # Corrected flux
        flux = ( 
            self._original_source._flux( self._phase, self._wave ) *
            self._warp_spline( self._phase, self._wave )
        )
        if cut_negative_flux:
            flux[flux<0] = 0


        self._model_flux = Spline2d(self._phase, self._wave, flux, 
                                    kx=time_spline_degree, ky=3)




def get_db_table( name, database, tabulators ):
    """
    For ZTF name, get photopoints and then tables. 
    """

    # Name
    if isinstance(name, int):    
        # Assuming this is already a DB stock
        stock = int
    elif re.search('ZTF', name):
        stock = ZTFIdMapper.to_ampel_id(name)
    else:
        raise ValueError(f"Cannot parse {name}" )

    # Obtain photopoints
    dps = [dp for dp in database.t0.find({'stock':stock})]

    # Convert to table(s)
    ftables = [
        tabulator.get_flux_table(dps) for tabulator in tabulators
    ]
    if len(ftables)>1:
        raise NotImplementedError("Debug appending two tabulator tables.")
    return ftables.pop(0)



def get_ztftable_from_ampel(ztfid: str, dbhandle: Database, include_sigma: float = 5., **kwarg) -> Table:
    """
    Given a ZTF name and a local AMPEL DB:
    - Retrieve available photometric data.
    - Reject outliers. 
    - Return an astropy table useful e.g. for sncosmo. 

    Parameters:
    - ztfid: str : ZTF name, e.g. ZTF18aaayemw
    - dbhandle: Database : AMPEL MongoDB handle
    - inclusion_sigma: float : Sigma threshold for outlier rejection.
    - kwarg: dict : Additional arguments added to table meta.

    tab = get_ztftable_from_ampel('ZTF22aaa', dbhandle, inclusion_sigma=5., z=0.03, type='SN Ia')

    """

    # Load photopoints from AMPEL DB
    tabulators = [
        ZTFFPTabulator(inclusion_sigma=include_sigma)
        ]
    tab = get_db_table(ztfid, database=dbhandle, tabulators=tabulators)
    tab.sort('time')

    tab.meta = {
            'object_id':ztfid,
            **kwarg
        }

    return tab




def get_template_correction( 
    tab: Table, templatename: str, z: float, 
    max_chidof: float =50., min_bands: int = 2, min_point_band: int =5, pull_cut: float = 10,
    rv: float =3.1, max_phases: None | list[float] = None, require_phasecoverage: bool = True,
    spline_lam: float =0.1,
    plot_dir: None | str = None, plot_label: str='ZTF',
) -> dict:
    """
    Attempt to calculate correction coefficients for this template to the datatable.
    Note that we will create a template with the same mean color as the SN.

    - Fit the template to the data, iteratively rejecting flux and time outliers.
    - If fit not good enough (bands, datapoints, chi), end process.
    - Extract model prediction at the specific phases and wavelengths of each band. 
    - Divide this with the observe data to create a correction vector 
        V( restPhase, restWave )
        where restPhase are restframe phases of the observation in the template time
        and restWave the effective wavelength of the observed band shifted to z=0.
    - Make sure that these end at 1 at wavelength ends.
    - Add buffer vectors of 1 at template min and max phases. 
    - Create 2d python interpolation object based on the above.
    - Interpolate from object to the phases of the template.
    - Apply same reddening to template as was fitted.
    - Divide template with interpolation.

    if require_phasecoverage: 
    - Do not extrapolate phases beyond what the supernova template included. If false
    will use max_phases or template range (whichver is smaller). (Check not done per band).
    """

    if max_phases is None:
        max_phases = [-20.,100.]


    # Fit template 
    if re.search( 'salt', templatename ):
        raise ValueError('SALT templates not incorporated')
    dust = sncosmo.CCM89Dust()
    m = sncosmo.Model(source=templatename,
                      effects=[dust],
                      effect_names=['host'],
                      effect_frames=['rest'])
    m.set(hostr_v = rv)
    m.set(z=z)
    fitprop = ['t0', 'amplitude', 'hostebv']
        
    # run the fit
    mdict: dict = {'model':templatename,'dps_init':len(tab)}
    try:
        result, fitted_model = sncosmo.fit_lc(
            tab, m,
            fitprop,  # parameters of model to vary
        )
        mdict.update( {
            result['param_names'][k]:result['parameters'][k] 
            for k in range(len(result['parameters'])) 
            }
        )

        # Look for outliers in time. 
        phases = (tab['time']-mdict['t0'])/(1+z)
        iGood = ( 
            ( max_phases[0] < phases ) & (phases < max_phases[1] ) & 
            ( phases > m.mintime() ) & 
            ( phases < m.maxtime() )
        )

        # Look for mag outliers in flux
        # Well, if we start far away we will cut away the real data to look like the model 
        result, fitted_model = sncosmo.fit_lc(
                tab[iGood], fitted_model,
                fitprop
        )
        iNorm = np.abs( 
            (tab['flux']-fitted_model.bandflux(tab['band'], tab['time'],zp=25,zpsys='ab')) / tab['fluxerr']
            ) < pull_cut 
        mdict['dps_fcut'] = sum(iNorm)
        iTot = iGood & iNorm

        # Final fit
        result, fitted_model = sncosmo.fit_lc(
                tab[iTot], fitted_model,
                fitprop
        )
        mdict.update( {
            result['param_names'][k]:result['parameters'][k] 
            for k in range(len(result['parameters'])) 
            }
        )


        # Final outlier selection 
        phases = (tab['time']-mdict['t0'])/(1+z)
        iGood = ( 
            ( max_phases[0] < phases ) & (phases < max_phases[1] ) & 
            ( phases > m.mintime() ) & 
            ( phases < m.maxtime() )
        )
        mdict['dps_tcut'] = sum(iGood)
        iNorm = np.abs( 
            (tab['flux']-fitted_model.bandflux(tab['band'], tab['time'],zp=25,zpsys='ab')) / tab['fluxerr']
            ) < pull_cut 
        mdict['dps_fcut'] = sum(iNorm)
        iTot = iGood & iNorm
        mdict['dps_allcut'] = sum(iTot)
        
    except (RuntimeError, ValueError, KeyError, DataQualityError):
        mdict['success'] = False
        return mdict
    if result.ndof < 1:
        mdict['success'] = False
        return mdict       
    mdict['success'] = True 

    mdict.update( {k:result[k] for k in ['success', 'chisq', 'ndof', 'errors']} )
    mdict['absmag'] = fitted_model.source_peakabsmag(band='bessellb',magsys='ab')
    mdict['chidof'] = result.chisq / result.ndof 

    # Inspect whether fit is good enough for template construction
    mdict['lceval'] = {
        band: sum(tab['band'][iTot]==band)
        for band in set( tab['band'][iTot] )
    }
    if (len([count for count in mdict['lceval'].values()
            if count>min_point_band])) < min_bands:
        mdict['success'] = False
        return mdict         
    if mdict['chidof'] > max_chidof:
        mdict['success'] = False
        return mdict

    # At this point we have a model which should be ok for template building. 

    # Plot the model
    if plot_dir:
        fig = plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)

    # Create the frac
    startphase = max( m.mintime(), max_phases[0] )
    endphase = min( m.maxtime(), max_phases[1] )
    corr_frac = tab['flux'] / fitted_model.bandflux(tab['band'], tab['time'],zp=25,zpsys='ab')        
    mdict['corrdata'] = {}
    for band in set( tab['band'][iTot] ):
        iBand = (tab['band'][iTot]==band)
        band_corr, band_err = list(corr_frac[iTot][iBand]), list(tab['fluxerr'][iTot][iBand])
        band_phase = list( (tab['time'][iTot][iBand]-mdict['t0'])/(1+z) )
        bandfunc = sncosmo.get_bandpass(band)
        rest_wave = bandfunc.wave_eff / (1+z)
        # Add limits to ensure correction goes to 1 at edges if so set
        if not require_phasecoverage:
            # Add inital cadenced buffer starting at startphase and ending at first phase point, and same at end.
            buffercadence = 3.
            pre_phases = np.arange( startphase, min(band_phase), buffercadence )
            if len(pre_phases)==0:
                pre_phases = [startphase]
            post_phases = np.arange( max(band_phase)+buffercadence, endphase, buffercadence )
            if len(post_phases)==0:
                post_phases = [endphase]

            pre_err = np.linspace(0.01, band_err[0], len(pre_phases))
            post_err = np.linspace(band_err[-1], 0.01, len(post_phases))

            band_phase = [*pre_phases, *band_phase, *post_phases]
            band_corr = [1]*len(pre_phases) + band_corr + [1]*len(post_phases)
            band_err = list(pre_err) + band_err + list(post_err)
#            band_phase = [startphase, *band_phase, endphase]
#            band_corr = [1, *band_corr, 1]
#            band_err = [0.01, *band_err, 0.01]

        # Interpolate to model phases
        if len(band_phase)<5:
            # Not enough points to make a spline
            continue
        spl = make_smoothing_spline( band_phase, band_corr, 
                                    w=1.0 / np.array(band_err) ** 2, lam=spline_lam
                                    )
        dspl = make_smoothing_spline( band_phase, band_err, 
                                     w=1.0 / np.array(band_err) ** 2, lam=spline_lam
                                    )
        # Have to access source _phase directly
        tphase = m.source._phase[                  
            (m.source._phase>=startphase) & (m.source._phase<=endphase)   
        ]

        finterp = spl(tphase)


        mdict['corrdata'][band] = {
            'wave': rest_wave, 'phase': band_phase, 
            'frac': band_corr, 'err': band_err,
            'tphase':tphase, 'tcorr':finterp, 'terr':dspl(tphase)
        }

        if plot_dir:
            plt.plot(band_phase, band_corr, 'o', label=f"{band} at {rest_wave}")
            plt.plot(tphase, finterp, "-.", label=f"Interp: {band}")
            plt.legend()



    # Create vectors for the interpolation
    # Wavelength limits 
    wave = [m.minwave()]
    phase: list[np.ndarray] = []
    flux: list[np.ndarray] = []
    # Assuming ztf bands, doing this to ensure order
    for band in ['ztfg', 'ztfr', 'ztfi']:
        if band not in mdict['corrdata']:
            continue
        wave.append( mdict['corrdata'][band]['wave'] )
        if len(phase)==0:
            # Have to start initiating 
            phase = mdict['corrdata'][band]['tphase']
            flux = [np.ones( len(phase) )]
            flux.append( mdict['corrdata'][band]['tcorr'] )
        else:
            if sum( phase - mdict['corrdata'][band]['tphase'] )>0:
                raise ValueError('incompatible phase error')
            flux.append( mdict['corrdata'][band]['tcorr'] )
    wave.append( m.maxwave() )
    flux.append( np.ones( len(phase) ) )
    flux2d = np.array(flux).transpose()
    corrmodel = Spline2d(phase, wave, flux2d , kx=3, ky=3)

    # One could apply reddening according to the fitted model to the flux
    # Makes the model very dependent on the fitted ebv, so not doing this for now.
#    flux2d = fitted_model.effects[0].propagate(np.array(wave), flux2d)
    # Add as potential init param to WarpTimeSeriesSource
    mdict['corrmodel'] = {'phase':phase,'wave':wave,'flux':flux2d}


    if plot_dir:
        # Todo: Make a unified plot with data, uncorrected and corrected model.

        # Finish correction model plot
        plt.subplot(1,2,2)
        plotwave = np.arange( 3500, 7000, 500 )
        for plotphase in [-10,-5,0,5,10,15,20]:
            plt.plot(plotwave, corrmodel(plotphase, plotwave)[0], 
                     label=f'@ {plotphase}'
                    )
        plt.legend()
        fig.savefig(plot_dir + f"/{plot_label}_{templatename}_warpmodel.png") 
        plt.close(fig)
        
        # Corrected model
        fig = sncosmo.plot_lc(tab[iTot], model=fitted_model, errors=result.errors)
        fig.savefig(plot_dir + f"/{plot_label}_{templatename}_warpfit.png") 
        plt.close(fig)
        
    return mdict



class WarpfitTemplateLoader:
    def __init__(
        self,
        warpcoeffs_dir: str,
        logger: Optional[logging.Logger] = None
    ):
        self.warpcoeffs_dir = warpcoeffs_dir
        self._cache: Dict[str, list] = {}

        if logger is None:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(message)s"
            )
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

    # -------------------------
    # Internal: load with cache
    # -------------------------
    def _load_coeffs(self, fitclass: str) -> list:
        key = re.sub(r"/", "", fitclass)

        if key in self._cache:
            self.logger.debug(f"Cache hit for fitclass={key}")
            return self._cache[key]

        filepath = os.path.join(
            self.warpcoeffs_dir,
            f"warpcoeffs_{key}.pkl"
        )

        self.logger.info(f"Loading warpcoeffs from {filepath}")

        if not os.path.exists(filepath):
            self.logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(filepath)

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self._cache[key] = data
        return data

    def get_templates(
        self,
        fitclass: str,
        exclude_input: Optional[list] = None,
        template_selection: Union[int, str] = 1,
        snbasis_selection: Union[int, str] = 1,
        require_good_templatefit: bool = False,
        random_seed: Optional[int] = None,   # <-- NEW
    ) -> List[Dict]:
    
        exclude_input = exclude_input or []
    
        # -------------------------
        # Local RNG (reproducible)
        # -------------------------
        rng = random.Random(random_seed)
    
        if random_seed is not None:
            self.logger.info(f"Using random seed: {random_seed}")
    
        template_collection = self._load_coeffs(fitclass)
    
        # -------------------------
        # Filter SN bases
        # -------------------------
        valid_snbases = []
        for sndict in template_collection:
    
            sn_name = sndict.get("ztfid")
    
            if sn_name in exclude_input:
                self.logger.debug(f"Excluded SN basis: {sn_name}")
                continue
    
            if require_good_templatefit and not sndict.get("good_fit", False):
                self.logger.debug(f"Rejected (bad fit): {sn_name}")
                continue
    
            valid_snbases.append(sndict)
    
        if not valid_snbases:
            self.logger.warning("No valid SN bases after filtering")
            return []
    
        self.logger.info(f"{len(valid_snbases)} SN bases available after filtering")
    
        # -------------------------
        # Select SN bases
        # -------------------------
        if snbasis_selection == "all":
            selected_snbases = valid_snbases
    
        elif isinstance(snbasis_selection, int):
            selected_snbases = rng.choices(
                valid_snbases,
                k=snbasis_selection
            )
    
        else:
            raise ValueError("snbasis_selection must be int or 'all'")
    
        results = []
    
        # -------------------------
        # Loop SN bases
        # -------------------------
        for sndict in selected_snbases:
            sn_name = sndict["ztfid"]
    
            possible_templates = []
    
            for warpfit in sndict["warpmodels"]:
    
                template_sn = warpfit.get("model")
    
                if template_sn in exclude_input:
                    self.logger.debug(
                        f"Excluded template {template_sn} (basis {sn_name})"
                    )
                    continue
    
                try:
                    model = get_warpedTimeSeriesModel(
                        name=f"{sn_name}_{template_sn or 'tpl'}",
                        original_template_name=template_sn,
                        warpdata=warpfit["mdict"],
                        z=float(warpfit["z"]),
                        original_template_version=None,
                    )
                except Exception as e:
                    self.logger.error(
                        f"Model construction failed for {sn_name}: {e}"
                    )
                    continue
    
                possible_templates.append({
                    "basis_sn": sn_name,
                    "model": model,
                    "template_prob": warpfit["draw_prob"],
                })
    
            if not possible_templates:
                self.logger.debug(f"No valid templates for SN {sn_name}")
                continue
    
            # -------------------------
            # Template selection
            # -------------------------
            if template_selection == "all":
                selected_templates = possible_templates
    
            elif isinstance(template_selection, int):
    
                if template_selection > 0:
                    weights = [tpl["template_prob"] for tpl in possible_templates]
    
                    selected_templates = rng.choices(
                        possible_templates,
                        weights=weights,
                        k=template_selection
                    )
        
                else:
                    selected_templates = rng.choices(
                        possible_templates,
                        k=abs(template_selection)
                    )
    
            else:
                raise ValueError("template_selection must be int or 'all'")
    
            results.extend(selected_templates)
    
        self.logger.info(f"Returning {len(results)} templates")
    
        return results

    # -------------------------
    # Optional: cache control
    # -------------------------
    def clear_cache(self):
        self.logger.info("Clearing warpcoeff cache")
        self._cache.clear()

