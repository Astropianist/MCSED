""" SED fitting class using emcee for parameter estimation

.. moduleauthor:: Greg Zeimann <gregz@astro.as.utexas.edu>

"""
import logging
import sfh
import dust_abs
import dust_emission
import metallicity
import cosmology
import emcee
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner
import time
from scipy.integrate import simps
from scipy.interpolate import interp1d
from astropy.constants import c as clight
import numpy as np
from scipy.misc import derivative
from itertools import cycle
import os.path as op

plt.ioff() 

import seaborn as sns
sns.set_context("talk") # options include: talk, poster, paper
sns.set_style("ticks")
sns.set_style({"xtick.direction": "in","ytick.direction": "in",
               "xtick.top":True, "ytick.right":True,
               "xtick.major.size":12, "xtick.minor.size":4,
               "ytick.major.size":12, "ytick.minor.size":4,
               })
color_palette_arr = sns.color_palette('dark')
color_palette = cycle(tuple(color_palette_arr))

def partial_derivative(func, var=0, point=[],dx=1e-6):
    ''' Partial derivative of function func given variable number var
        at point point with spacing dx used for computation 
        Note: This version expects the argument of the function to be a single array '''
    args = np.copy(point)
    def wraps(x):
        args[var] = x
        return func(args)
    return derivative(wraps, point[var], dx = dx)

class Mcsed:
    def __init__(self, filter_matrix, ssp_spectra,
                 emlinewave, ssp_emline, ssp_ages, ssp_met, wave, 
                 sfh_class, dust_abs_class, dust_em_class, met_class=None,
                 nfreeparams=None, t_birth=None, SSP=None, lineSSP=None, 
                 data_fnu=None, data_fnu_e=None, 
                 data_emline=None, data_emline_e=None, emline_dict=None,
                 use_emline_flux=None, linefluxCSPdict=None,
                 data_absindx=None, data_absindx_e=None, absindx_dict=None,
                 use_absorption_indx=None, absindxCSPdict=None,
                 fluxwv=None, indsort=None, fluxfn=None, vararr=None,
                 spectrum=None, redshift=None, Dl=None, filter_flag=None, 
                 input_params=None, true_fnu=None, true_spectrum=None, 
                 sigma_m=0.1, nwalkers=40, nsteps=1000, 
                 chi2=None, tauISM_lam=None, tauIGM_lam=None, indsort=None):
        ''' Initialize the Mcsed class.

        Init
        ----
        filter_matrix : numpy array (2 dim)
            The filter_matrix has rows of wavelength and columns for each
            filter (can be much larger than the filters used for fitting)
        ssp_spectra : numpy array (3 dim)
            single stellar population spectrum for each age in ssp_ages
            and each metallicity in ssp_met 
        emlinewave : numpy array (1 dim)
            Rest-frame wavelengths of requested emission lines (emline_dict)
            Corresponds to ssp_emline
        ssp_emline : numpy array (3 dim)
            Emission line SSP grid spanning emlinewave, age, metallicity
            Only includes requested emission lines (from emline_dict)
            Only used for calculating model emission line strengths
            Spectral units are ergs / s / cm2 at 10 pc
        ssp_ages : numpy array (1 dim)
            ages of the SSP models
        ssp_met : numpy array (1 dim)
            metallicities of the SSP models
            assume a grid of values Z, where Z_solar = 0.019
        wave : numpy array (1 dim)
            wavelength for SSP models and all model spectra
        sfh_class : str
            Converted from str to class in initialization
            This is the input class for sfh.  Each class has a common attribute
            which is "sfh_class.get_nparams()" for organizing the total model_params.
            Also, each class has a key function, sfh_class.evaluate(t), with
            the input of time in units of Gyrs
        dust_abs_class : str 
            Converted from str to class in initialization
            This is the input class for dust absorption.
        dust_em_class : str
            Converted from str to class in initialization
            This is the input class for dust emission.
        met_class : str
            Converted from str to class in initialization
            This is the input class for stellar metallicity
        nfreeparams : int
            number of free model parameters
        t_birth : float
            Age of the birth cloud in Gyr
            set from the value provided in config.py
        SSP : numpy array (2 dim)
            Grid of SSP spectra at current guess of stellar metallicity
            (set from ssp_spectra)
        lineSSP : numpy array (2 dim)
            Grid of emission line fluxes at each age in the SSP grid
            (set from ssp_emline)
        data_fnu : numpy array (1 dim)
            Photometry for data.  Length = (filter_flag == True).sum()
        data_fnu_e : numpy array (1 dim)
            Photometric errors for data
        data_emline : Astropy Table (1 dim)
            Emission line fluxes in units ergs / cm2 / s
        data_emline_e : Astropy Table (1 dim)
            Emission line errors in units ergs / cm2 / s
        emline_dict : dictionary
            Keys are emission line names (str)
            Values are a two-element tuple:
                (rest-frame wavelength in Angstroms (float), weight (float))
            emline_list_dict defined in config.py, containing only the 
            emission lines that were also provided in the input file
            (i.e., only the measurements that will be used to constrain the model)
        use_emline_flux : bool
            If emline_dict contains emission lines, set to True. Else, False
        linefluxCSPdict : dict
            Emission-line fluxes for current SED model
        data_absindx : Astropy Table (1 dim)
            Absorption line indices
        data_absindx_e : Astropy Table (1 dim)
            Absorption line index errors
        absindx_dict : dict
            absorption_index_dict defined in config.py, containing only 
            measurements that were also provided in the input file
            (i.e., only the measurements that will be used to constrain the model)
        use_absorption_indx : bool
            True, if index measurements were included in the input file and
            should be used in the model selection
        absindxCSPdict : dict
            Absorption line index measurements for current SED model
        fluxwv : numpy array (1 dim)
            wavelengths of photometric filters
        indsort: numpy array (1 dim)
            Array to sort the photometric filter wavelengths
        fluxfn : numpy array (2 dim)
            flux densities of modeled photometry at various parameter configurations
        vararr : numpy array (1 dim)
            Array of values for a given variable (a linspace)
        spectrum : numpy array (1 dim)
            current SED model (same length as self.wave) 
        redshift : float
            Redshift of the source
        Dl : float
            Luminosity distance of the galaxy (in units of 10 pc)
        filter_flag : numpy array (1 dim)
            Length = filter_matrix.shape[1], True for filters matching data
        input_params : list
            input parameters for modeling.  Intended for testing fitting
            procedure.
        true_fnu : numpy array (1 dim)
            True photometry for test mode.  Length = (filter_flag == True).sum()
        true_spectrum : numpy array (1 dim)
            truth model spectrum in test model (realized from input_params)
        sigma_m : float
            Fractional error expected from the models.  This is used in
            the log likelihood calculation.  No model is perfect, and this is
            more or less a fixed parameter to encapsulate that.
        nwalkers : int
            The number of walkers for emcee when fitting a model
        nsteps : int
            The number of steps each walker will make when fitting a model
        chi2 : dict
            keys: 'dof', 'chi2', 'rchi2'
            Track the degrees of freedom (accounting for data and model parameters)
            and the chi2 and reduced chi2 of the current fit
        tauISM_lam : numpy array (1 dim)
            Array of effective optical depths as function of wavelength 
            for MW dust correction
        tauIGM_lam : numpy array (1 dim)
            Array of effective optical depths as function of wavelength 
            for IGM gas correction
        '''
        # Initialize all argument inputs
        self.filter_matrix = filter_matrix
        self.ssp_spectra = ssp_spectra
        self.emlinewave = emlinewave
        self.ssp_emline = ssp_emline
        self.ssp_ages = ssp_ages
        self.ssp_met = ssp_met
        self.wave = wave
        self.dnu = np.abs(np.hstack([0., np.diff(2.99792e18 / self.wave)]))
        self.sfh_class = getattr(sfh, sfh_class)()
        self.dust_abs_class = getattr(dust_abs, dust_abs_class)()
        self.dust_em_class = getattr(dust_emission, dust_em_class)()
        self.met_class = getattr(metallicity, 'stellar_metallicity')()
        self.param_classes = ['sfh_class', 'dust_abs_class', 'met_class',
                              'dust_em_class']
        self.nfreeparams = nfreeparams
        self.t_birth = t_birth
        self.SSP = None
        self.lineSSP = None
        self.data_fnu = data_fnu
        self.data_fnu_e = data_fnu_e
        self.data_emline = data_emline
        self.data_emline_e = data_emline_e
        self.emline_dict = emline_dict
        self.use_emline_flux = use_emline_flux
        self.linefluxCSPdict = None
        self.data_absindx = data_absindx
        self.data_absindx_e = data_absindx_e
        self.absindx_dict = absindx_dict
        self.use_absorption_indx = use_absorption_indx
        self.absindxCSPdict = None
        self.fluxwv = fluxwv
        self.indsort = indsort
        self.fluxfn = fluxfn
        self.vararr = vararr
        self.spectrum = None
        self.redshift = redshift
        if self.redshift is not None:
            self.set_new_redshift(self.redshift)
        self.Dl = Dl
        self.filter_flag = filter_flag
        self.input_params = input_params
        self.true_fnu = true_fnu
        self.true_spectrum = true_spectrum
        self.sigma_m = sigma_m
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.chi2 = chi2
        self.tauISM_lam = tauISM_lam
        self.tauIGM_lam = tauIGM_lam

        # Set up logging
        self.setup_logging()

    def set_new_redshift(self, redshift):
        ''' Setting redshift

        Parameters
        ----------
        redshift : float
            Redshift of the source for fitting
        '''
        self.redshift = redshift
        # Need luminosity distance to adjust spectrum to distance of the source
        self.Dl = cosmology.Cosmology().luminosity_distance(self.redshift)
        self.sfh_class.set_agelim(self.redshift)

    def setup_logging(self):
        '''Setup Logging for MCSED

        Builds
        -------
        self.log : class
            self.log.info() is for general print and self.log.error() is
            for raise cases
        '''
        self.log = logging.getLogger('mcsed')
        if not len(self.log.handlers):
            # Set format for logger
            fmt = '[%(levelname)s - %(asctime)s] %(message)s'
            fmt = logging.Formatter(fmt)
            # Set level of logging
            level = logging.INFO
            # Set handler for logging
            handler = logging.StreamHandler()
            handler.setFormatter(fmt)
            handler.setLevel(level)
            # Build log with name, mcsed
            self.log = logging.getLogger('mcsed')
            self.log.setLevel(logging.DEBUG)
            self.log.addHandler(handler)

    def remove_waverange_filters(self, wave1, wave2, restframe=True):
        '''Remove filters in a given wavelength range

        Parameters
        ----------
        wave1 : float
            start wavelength of masked range (in Angstroms)
        wave2 : float
            end wavelength of masked range (in Angstroms)
        restframe : bool
            if True, wave1 and wave2 correspond to rest-frame wavelengths
        '''
        wave1, wave2 = np.sort([wave1, wave2])
        if restframe:
            wave_factor = 1. + self.redshift
        else:
            wave_factor = 1.
        loc1 = np.searchsorted(self.wave, wave1 * wave_factor)
        loc2 = np.searchsorted(self.wave, wave2 * wave_factor)
        # account for the case where indices are the same
        if (loc1 == loc2):
            loc2+=1
        maxima = np.max(self.filter_matrix, axis=0)
        try:
            newflag = np.max(self.filter_matrix[loc1:loc2, :], axis=0) < maxima * 0.1
        except ValueError:
            return
        maximas = np.max(self.filter_matrix[:, self.filter_flag], axis=0)
        newflags = np.max(self.filter_matrix[loc1:loc2, self.filter_flag], axis=0) < maximas * 0.1
        self.filter_flag = self.filter_flag * newflag
        if self.true_fnu is not None:
            self.true_fnu = self.true_fnu[newflags]
        self.data_fnu = self.data_fnu[newflags]
        self.data_fnu_e = self.data_fnu_e[newflags]


    def get_filter_wavelengths(self):
        '''Get central wavelengths of photometric filters 
        '''
        wave_avg = np.dot(self.wave, self.filter_matrix[:, self.filter_flag])
        return wave_avg

    def get_filter_fluxdensities(self):
        '''Convert a spectrum to photometric fluxes for a given filter set.
        The photometric fluxes will be in the same units as the spectrum.
        The spectrum is in microjanskies(lambda) such that
        the photometric fluxes will be in microjanskies.

        Returns
        -------
        f_nu : numpy array (1 dim)
            Photometric flux densities for an input spectrum
        '''
        f_nu = np.dot(self.spectrum, self.filter_matrix[:, self.filter_flag])
        return f_nu


    def measure_absorption_index(self):
        '''
        measure absorption indices using current spectrum
        '''
        self.absindxCSPdict = {}
        if self.use_absorption_indx:
            # convert the spectrum from units of specific frequency to specific wavelength
            wave = self.wave.copy()
            factor = clight.to('Angstrom/s').value / wave**2.
            spec = self.spectrum * factor

            for indx in self.absindx_dict.keys():
                wht, wave_indx, wave_blue, wave_red, unit = self.absindx_dict[indx]

                # select appropriate data ranges for blue/red continuum and index
                sel_index = np.array([False]*len(wave))
                sel_index[np.argmin(abs(wave-wave_indx[0])):np.argmin(abs(wave-wave_indx[1]))] = True
                if abs(np.argmin(abs(wave-wave_indx[0]))-np.argmin(abs(wave-wave_indx[1])))<2:
                    sel_index[np.argmin(abs(wave-wave_indx[0])):np.argmin(abs(wave-wave_indx[0]))+2] = True
                sel_blue = np.array([False]*len(wave))
                sel_blue[np.argmin(abs(wave-wave_blue[0])):np.argmin(abs(wave-wave_blue[1]))] = True
                if abs(np.argmin(abs(wave-wave_blue[0]))-np.argmin(abs(wave-wave_blue[1])))<2:
                    sel_blue[np.argmin(abs(wave-wave_blue[0])):np.argmin(abs(wave-wave_blue[0]))+2] = True
                sel_red = np.array([False]*len(wave))
                sel_red[np.argmin(abs(wave-wave_red[0])):np.argmin(abs(wave-wave_red[1]))] = True
                if abs(np.argmin(abs(wave-wave_red[0]))-np.argmin(abs(wave-wave_red[1])))<2:
                    sel_red[np.argmin(abs(wave-wave_red[0])):np.argmin(abs(wave-wave_red[0]))+2] = True

                # estimate continuum in the index:
                fw_blue  = np.dot(spec[sel_blue][0:-1], np.diff(wave[sel_blue])) 
                fw_blue /= np.diff(wave[sel_blue][[0,-1]])
                fw_red   = np.dot(spec[sel_red][0:-1],  np.diff(wave[sel_red]))  
                fw_red  /= np.diff(wave[sel_red][[0,-1]])
                cont_waves = [np.median(wave_blue), np.median(wave_red)]
                cont_fw    = [fw_blue, fw_red]
                coeff = np.polyfit( cont_waves, cont_fw, 1)
                cont_index = coeff[0] * wave[sel_index] + coeff[1]

                # flux ratio of index and continuum
                spec_index = spec[sel_index] / cont_index

                if unit==0: # return measurement in equivalent width (Angstroms)
                    value = np.dot( 1. - spec_index[0:-1], np.diff(wave[sel_index]) )

                if unit==1: # return measurement in magnitudes
                    integral = np.dot( spec_index[0:-1], np.diff(wave[sel_index]) )
                    value = -2.5 * np.log10( integral / np.diff(wave[sel_index][[0,-1]]) ) 

                if unit==2: # return measurement as a flux density ratio (red / blue)
                    value = fw_red / fw_blue

                self.absindxCSPdict[indx] = float(value)


    def set_class_parameters(self, theta):
        ''' For a given set of model parameters, set the needed class variables
        related to SFH, dust attenuation, ect.

        Input
        -----
        theta : list
            list of input parameters for sfh, dust attenuation, 
            stellar metallicity, and dust emission
        '''
        start_value = 0
        ######################################################################
        # STAR FORMATION HISTORY
        self.sfh_class.set_parameters_from_list(theta, start_value)
        start_value += self.sfh_class.get_nparams()

        ######################################################################
        # DUST ATTENUATION
        self.dust_abs_class.set_parameters_from_list(theta, start_value)
        start_value += self.dust_abs_class.get_nparams()

        ######################################################################
        # STELLAR METALLICITY 
        self.met_class.set_parameters_from_list(theta, start_value)
        start_value += self.met_class.get_nparams()

        ######################################################################
        # DUST EMISSION
        self.dust_em_class.set_parameters_from_list(theta, start_value)
        start_value += self.dust_em_class.get_nparams()


    def get_ssp_spectrum(self):
        '''
        Calculate SSP for an arbitrary metallicity (self.met_class.met) given a
        model grid for a range of metallicities (self.ssp_met)

        if left as a free parameter, stellar metallicity (self.met_class.met)
        spans a range of log(Z / Z_solar)

        the SSP grid of metallicities (self.ssp_met) assumes values of Z
        (as opposed to log solar values)

        Returns
        -------
        SSP : 2-d array
            Single stellar population models for each age in self.ages
        lineSSP : 2-d array
            Single stellar population line fluxes for each age in self.ages

        '''
        if self.met_class.fix_met:
            if self.SSP is not None:
                return self.SSP, self.lineSSP
        Z = np.log10(self.ssp_met)
        Zsolar = 0.019
        z = self.met_class.met + np.log10(Zsolar)
        X = Z - z
        wei = np.exp(-(X)**2 / (2. * 0.15**2))
        wei /= wei.sum()
        self.SSP = np.dot(self.ssp_spectra, wei)
        if self.use_emline_flux:
            self.lineSSP = np.dot(self.ssp_emline, wei)
        else:
            self.lineSSP = self.ssp_emline[:,:,0]
        return self.SSP, self.lineSSP

    def build_csp(self, sfr=None):
        '''Build a composite stellar population model for a given star
        formation history, dust attenuation law, and dust emission law.

        In addition to the returns it also modifies a lineflux dictionary

        Returns
        -------
        csp : numpy array (1 dim)
            Composite stellar population model (micro-Jy) at self.redshift
        mass : float
            Mass for csp given the SFH input
        '''
        # Collapse for metallicity
        SSP, lineSSP = self.get_ssp_spectrum()

        # Need star formation rate from observation back to formation
        if sfr is None:
            sfr = self.sfh_class.evaluate(self.ssp_ages)
        ageval = 10**self.sfh_class.age # Gyr

        # Treat the birth cloud and diffuse component separately
        age_birth = self.t_birth 

        # Get dust-free CSPs, properly accounting for ages
        # ageval sets limit on ssp_ages that are useable in model calculation
        # age_birth separates birth cloud and diffuse components
        sel = (self.ssp_ages > age_birth) & (self.ssp_ages <= ageval)
        sel_birth = (self.ssp_ages <= age_birth) & (self.ssp_ages <= ageval)
        sel_age = self.ssp_ages <= ageval

        # The weight is the linear time between ages of each SSP
        weight = np.diff(np.hstack([0, self.ssp_ages])) * 1e9 * sfr
        weight_orig = weight.copy()
        weight_birth = weight.copy()
        weight_age = weight.copy()
        weight[~sel] = 0
        weight_birth[~sel_birth] = 0
        weight_age[~sel_age] = 0

        # Cover the two cases where ssp_ages contains ageval and when not
        # A: index of last acceptable SSP age
        A = np.nonzero(self.ssp_ages <= ageval)[0][-1]
        # indices of SSP ages that are too old
        select_too_old = np.nonzero(self.ssp_ages >= ageval)[0]
        if len(select_too_old):
            # B: index of first SSP that is too old
            B = select_too_old[0]
            # only adjust weight if ageval falls between two SSP age gridpoints
            if A != B:
                lw = ageval - self.ssp_ages[A]
                wei = lw * 1e9 * np.interp(ageval, self.ssp_ages, sfr)
                if ageval > age_birth:
                    weight[B] = wei
                if ageval <= age_birth:
                    weight_birth[B] = wei
                weight_age[B] = wei

        # Cover two cases where ssp_ages contains age_birth and when not
        # A: index of last acceptable SSP age
        A = np.nonzero(self.ssp_ages <= age_birth)[0][-1]
        # indices of SSP ages that are too old
        select_too_old = np.nonzero(self.ssp_ages >= age_birth)[0]
        if (len(select_too_old)>0): 
            # B: index of first SSP that is too old
            B = select_too_old[0]
            if A != B:
                lw = age_birth - self.ssp_ages[A]
                wei = lw * 1e9 * np.interp(age_birth, self.ssp_ages, sfr)
                if ageval > age_birth:
                    weight[B] = weight_age[B] - wei
                if ageval >= age_birth:
                    weight_birth[B] = wei
                else:
                    weight_birth[B] = weight_age[B]

        # Finally, do the matrix multiplication using the weights
        spec_dustfree = np.dot(self.SSP, weight)
        spec_birth_dustfree = np.dot(self.SSP, weight_birth)
        linespec_dustfree = np.dot(self.lineSSP, weight_birth)
        mass = np.sum(weight_age)

        # Need to correct spectrum for dust attenuation
        Alam = self.dust_abs_class.evaluate(self.wave)
        spec_dustobscured = spec_dustfree * 10**(-0.4 * Alam)

        # Correct the corresponding birth cloud spectrum separately
        Alam_birth = Alam / self.dust_abs_class.EBV_old_young
        spec_birth_dustobscured = spec_birth_dustfree * 10**(-0.4 * Alam_birth)

        # Combine the young and old components
        spec_dustfree += spec_birth_dustfree
        spec_dustobscured += spec_birth_dustobscured

        # Compute attenuation for emission lines
        Alam_emline = (self.dust_abs_class.evaluate(self.emlinewave,new_wave=True)
                       / self.dust_abs_class.EBV_old_young)
        linespec_dustobscured = linespec_dustfree * 10**(-0.4*Alam_emline)

        if self.dust_em_class.assume_energy_balance:
            # Bolometric luminosity of dust attenuation (for energy balance)
            L_bol = (np.dot(self.dnu, spec_dustfree) - np.dot(self.dnu, spec_dustobscured)) 
            dust_em = self.dust_em_class.evaluate(self.wave)
            L_dust = np.dot(self.dnu,dust_em)
            mdust_eb = L_bol/L_dust 
            spec_dustobscured += mdust_eb * dust_em
        else:
            spec_dustobscured += self.dust_em_class.evaluate(self.wave)

        # Redshift the spectrum to the observed frame
        csp = np.interp(self.wave, self.wave * (1. + self.redshift),
                        spec_dustobscured * (1. + self.redshift))

        # Correct for ISM and/or IGM (or neither)
        if self.tauIGM_lam is not None:
            csp *= np.exp(-self.tauIGM_lam)
        if self.tauISM_lam is not None:
            csp *= np.exp(-self.tauISM_lam)

        # Update dictionary of modeled emission line fluxes
        linefluxCSPdict = {}
        if self.use_emline_flux:
            for emline in self.emline_dict.keys():
                indx = np.argmin(np.abs(self.emlinewave 
                                        - self.emline_dict[emline][0]))
                # flux is given in ergs / s / cm2 at 10 pc
                flux = linespec_dustobscured[indx]
                # Correct flux from 10pc to redshift of source
                linefluxCSPdict[emline] = linespec_dustobscured[indx] / self.Dl**2
        self.linefluxCSPdict = linefluxCSPdict

        # Correct spectra from 10pc to redshift of the source
        if self.dust_em_class.assume_energy_balance:
            return csp / self.Dl**2, mass, mdust_eb
        else:
            return csp / self.Dl**2, mass

    def lnprior(self):
        ''' Simple, uniform prior for input variables

        Returns
        -------
        0.0 if all parameters are in bounds, -np.inf if any are out of bounds
        '''
        flag = True
        for par_cl in self.param_classes:
            flag *= getattr(self, par_cl).prior()
        if not flag:
            return -np.inf
        else:
            return 0.0

    def lnlike(self):
        ''' Calculate the log likelihood and return the value and stellar mass
        of the model as well as other derived parameters

        Returns
        -------
        log likelihood, mass, sfr10, sfr100, fpdr, mdust_eb : (all float)
            The log likelihood includes a chi2_term and a parameters term.
            The mass comes from building of the composite stellar population
            The parameters sfr10, sfr100, fpdr, mdust_eb are derived in get_derived_params(self)
        '''
        if self.dust_em_class.assume_energy_balance:
            self.spectrum, mass, mdust_eb = self.build_csp()
        else:
            self.spectrum, mass = self.build_csp()
            mdust_eb = None

        sfr10,sfr100,fpdr = self.get_derived_params()

        # likelihood contribution from the photometry
        model_y = self.get_filter_fluxdensities()
        inv_sigma2 = 1.0 / (self.data_fnu_e**2 + (model_y * self.sigma_m)**2)
        chi2_term = -0.5 * np.sum((self.data_fnu - model_y)**2 * inv_sigma2)
        parm_term = -0.5 * np.sum(np.log(1 / inv_sigma2))

        # calculate the degrees of freedom and store the current chi2 value
        if not self.chi2:
            dof_wht = list(np.ones(len(self.data_fnu)))

        # likelihood contribution from the absorption line indices
        self.measure_absorption_index()
        if self.use_absorption_indx:
            for indx in self.absindx_dict.keys():
                unit = self.absindx_dict[indx][-1]
                # if null value, ignore it (null = -99)
                if (self.data_absindx['%s_INDX' % indx]+99 > 1e-10):
                    indx_weight = self.absindx_dict[indx][0]
                    model_indx = self.absindxCSPdict[indx]
                    if unit == 1: # magnitudes
                        model_err = 2.5*np.log10(1.+self.sigma_m)
                    else:
                        model_err = model_indx * self.sigma_m
                    obs_indx = self.data_absindx['%s_INDX' % indx]
                    obs_indx_e = self.data_absindx_e['%s_Err' % indx]
                    sigma2 = obs_indx_e**2. + model_err**2.
                    chi2_term += (-0.5 * (model_indx - obs_indx)**2 /
                                  sigma2) * indx_weight
                    parm_term += -0.5 * np.log(indx_weight * sigma2)
                    if not self.chi2:
                        dof_wht.append(indx_weight) 

        # likelihood contribution from the emission lines
        if self.use_emline_flux:
            # if all lines have null line strengths, ignore 
            if not min(self.data_emline) == max(self.data_emline) == -99:
                for emline in self.emline_dict.keys():
                    if self.data_emline['%s_FLUX' % emline] > -99: # null value
                        emline_wave, emline_weight = self.emline_dict[emline]
                        model_lineflux = self.linefluxCSPdict[emline]
                        model_err = model_lineflux * self.sigma_m
                        lineflux  = self.data_emline['%s_FLUX' % emline]
                        elineflux = self.data_emline_e['%s_ERR' % emline]
                        sigma2 = elineflux**2. + model_err**2.
                        chi2_term += (-0.5 * (model_lineflux - lineflux)**2 /
                                      sigma2) * emline_weight
                        parm_term += -0.5 * np.log(emline_weight * sigma2)
                        if not self.chi2:
                            dof_wht.append(emline_weight)

        # record current chi2 and degrees of freedom
        if not self.chi2:
            self.chi2 = {}
            dof_wht = np.array(dof_wht)
            npt = ( sum(dof_wht)**2. - sum(dof_wht**2.) ) / sum(dof_wht) + 1
            self.chi2['dof'] = npt - self.nfreeparams 
        self.chi2['chi2']  = -2. * chi2_term
        self.chi2['rchi2'] = self.chi2['chi2'] / (self.chi2['dof'] - 1.)

        return (chi2_term + parm_term, mass,sfr10,sfr100,fpdr,mdust_eb)

    def lnprob(self, theta):
        ''' Calculate the log probabilty and return the value and stellar mass 
        (as well as derived parameters) of the model

        Returns
        -------
        log prior + log likelihood, [mass,sfr10,sfr100,fpdr,mdust_eb]: (all floats) 
            The log probability is just the sum of the logs of the prior and
            likelihood.  The mass comes from the building of the composite
            stellar population. The other derived parameters are calculated in get_derived_params()
        '''
        self.set_class_parameters(theta)
        lp = self.lnprior()
        if np.isfinite(lp):
            lnl,mass,sfr10,sfr100,fpdr,mdust_eb = self.lnlike()
            if not self.dust_em_class.fixed:
                if self.dust_em_class.assume_energy_balance:
                    return lp + lnl, np.array([mass,sfr10,sfr100,fpdr,mdust_eb])
                else:
                    return lp + lnl, np.array([mass, sfr10, sfr100, fpdr])
            else:
                return lp + lnl, np.array([mass, sfr10, sfr100])
        else:
            if not self.dust_em_class.fixed:
                if self.dust_em_class.assume_energy_balance:
                    return -np.inf, np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
                else:
                    return -np.inf, np.array([-np.inf, -np.inf, -np.inf, -np.inf])
            else:
                return -np.inf, np.array([-np.inf, -np.inf, -np.inf])

    def get_param_names(self):
        ''' Grab the names of the parameters for plotting

        Returns
        -------
        names : list
            list of all parameter names
        '''
        names = []
        for par_cl in self.param_classes:
            names.append(getattr(self, par_cl).get_names())
        names = list(np.hstack(names))
        return names

    def get_params(self):
        ''' Grab the the parameters in each class

        Returns
        -------
        vals : list
            list of all parameter values
        '''
        vals = []
        for par_cl in self.param_classes:
            vals.append(getattr(self, par_cl).get_params())
        vals = list(np.hstack(vals))
        self.nfreeparams = len(vals)
        return vals

    def get_param_lims(self):
        ''' Grab the limits of the parameters for making mock galaxies

        Returns
        -------
        limits : numpy array (2 dim)
            an array with parameters for rows and limits for columns
        '''
        limits = []
        for par_cl in self.param_classes:
            limits.append(getattr(self, par_cl).get_param_lims())
        limits = np.array(sum(limits, []))
        return limits

    def get_derived_params(self):
        ''' These are not free parameters in the model, but are instead
        calculated from free parameters
        '''
        # Lookback times for past 10 and 100 Myr (avoid t=0 for log purposes)
        t_sfr100 = np.linspace(1.0e-9,0.1,num=251)
        t_sfr10 = np.linspace(1.0e-9,0.01,num=251)
        # Time-averaged SFR over the past 10 and 100 Myr 
        sfrarray = self.sfh_class.evaluate(t_sfr100)
        sfr100 = simps(sfrarray,x=t_sfr100)/(t_sfr100[-1]-t_sfr100[0])
        sfrarray = self.sfh_class.evaluate(t_sfr10)
        sfr10 = simps(sfrarray,x=t_sfr10)/(t_sfr10[-1]-t_sfr10[0])

        if self.dust_em_class.fixed:
            fpdr = None
        else:
            if self.dust_em_class.assume_energy_balance:
                umin,gamma,qpah = self.dust_em_class.get_params()
            else:
                umin,gamma,qpah,mdust = self.dust_em_class.get_params()
            umax = 1.0e6
            fpdr = gamma*np.log(umax/100.) / ((1.-gamma)*(1.-umin/umax) + gamma*np.log(umax/umin))

        return sfr10,sfr100,fpdr

    def get_init_walker_values(self, kind='ball', num=None):
        ''' Before running emcee, this function generates starting points
        for each walker in the MCMC process.

        Returns
        -------
        pos : np.array (2 dim)
            Two dimensional array with Nwalker x Ndim values
        '''
        # We need an initial guess for emcee so we take it from the model class
        # parameter values and deltas
        init_params, init_deltas, init_lims = [], [], []
        for par_cl in self.param_classes:
            init_params.append(getattr(self, par_cl).get_params())
            init_deltas.append(getattr(self, par_cl).get_param_deltas())
            if len(getattr(self, par_cl).get_param_lims()):
                init_lims.append(getattr(self, par_cl).get_param_lims())
        theta = list(np.hstack(init_params))
        thetae = list(np.hstack(init_deltas))
        theta_lims = np.vstack(init_lims)
        if num is None:
            num = self.nwalkers
        if kind == 'ball':
            pos = emcee.utils.sample_ball(theta, thetae, size=num)
        else:
            pos = (np.random.rand(num)[:, np.newaxis] *
                   (theta_lims[:, 1]-theta_lims[:, 0]) + theta_lims[:, 0])
        return pos

    def fnu_vs_var(self,param_class,var,theta,numeval=300):
        self.set_class_parameters(theta)
        wv = self.get_filter_wavelengths()
        self.indsort = np.argsort(wv) #Want it in order of wavelength from short to long
        self.fluxwv = wv
        fnu = []
        lims = getattr(getattr(self,param_class),var+'_lims')
        rng = abs(lims[1]-lims[0])
        vararr = np.linspace(lims[0]+0.02*rng,lims[1]-0.02*rng,numeval)
        for i in range(numeval):
            setattr(getattr(self,param_class),var,self.vararr[i])
            if self.dust_em_class.assume_energy_balance:
                self.spectrum, mass, mdust_eb = self.build_csp()
            else:
                self.spectrum, mass = self.build_csp()
            fnu.append(self.get_filter_fluxdensities())
        return vararr, np.array(fnu)

    def get_var_num(self,param_class,var):
        ''' Get position of desired variable in the theta array '''
        if param_class is 'sfh_class': 
            return self.sfh_class.get_param_nums(var)
        elif param_class is 'dust_abs_class':
            return self.sfh_class.get_nparams() + self.dust_abs_class.get_param_nums(var)
        elif param_class is 'met_class':
            return ( self.sfh_class.get_nparams() + self.dust_abs_class.get_nparams()
                    + self.met_class.get_param_nums(var) )
        elif param_class is 'dust_em_class':
            return ( self.sfh_class.get_nparams() + self.dust_abs_class.get_nparams()
                    + self.met_class.get_nparams() + self.dust_em_class.get_param_nums(var) )
        else:
            print("What param class is this??")
            return -99

    def like_part_der_single(self,param_class,var,theta,dx=1.0e-6):
        var_num = self.get_var_num(param_class,var)
        assert var_num>=0, "Either param_class and/or var doesn't match MCSED specs"
        return partial_derivative(lnprob,var=var_num,point=theta,dx=dx)

    def like_part_der_mult(self,param_class,var,theta,numeval=200,dx=1.0e-6):
        thetamod = np.copy(theta)
        var_num = self.get_var_num(param_class,var)
        assert var_num>=0, "Either param_class and/or var doesn't match MCSED specs"
        lims = getattr(getattr(self,param_class),var+'_lims')
        rng = abs(lims[1]-lims[0])
        vararr = np.linspace(lims[0]+0.02*rng,lims[1]-0.02*rng,numeval)
        partder = np.zeros(numeval)
        for i in range(numeval):
            thetamod[var_num] = vararr[i]
            partder[i] = partial_derivative(lnprob,var=var_num,point=theta,dx=dx)
        return vararr, partder

    def plot_Fnu_vs_var(self,vararr,fnuarr,filt_names,varname,varlabel,id,
                        max_in_one=3,imgtype='png',imgdir=''):
        n = len(fnuarr[0])
        assert len(filt_names)==n
        assert len(vararr)==len(fnuarr[:,0])
        nplots = int(np.ceil(float(n)/max_in_one))
        dim1, dim2 = int(np.sqrt(nplots)), int(np.sqrt(nplots))
        if dim1*dim2<nplots: dim2+=1
        if dim1*dim2<nplots: dim1+=1
        fig, axes = plt.subplots(dim1,dim2,sharex=True)
        index = 0
        for i in range(dim1):
            for j in range(dim2):
                for k in range(max_in_one):
                    axes[i,j].semilogy(vararr,fnuarr[:,index],color=color_palette.next(),label=filt_names[index])
                    if index>=n-1: break
                    index+=1
                axes[i,j].legend(loc='best',size='small')
                axes[i,j].minorticks_on()
                if index>=n-1: break
            if index>=n-1: break
        fig.add_subplot(111,frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.xlabel(varlabel)
        plt.ylabel(r"Flux (${\rm{\mu}}$Jy)")
        plt.tight_layout()
        imgname = op.join(imgdir,"%s_Fnu_vs_%s.%s"%(id,varname,imgtype))
        fig.savefig(imgname,bbox_inches='tight',dpi=300)

    def plot_like_der_step(self,param_class,var,theta,varlabel,id,numeval=200,
                           imgtype='png',imgdir=''):
        multrange = np.geomspace(1.0e-8,1.0e-2,numeval)
        partder = np.zeros(numeval)
        lims = getattr(getattr(self,param_class),var+'_lims')
        rng = abs(lims[1]-lims[0])
        dx = multrange*rng
        for j in range(numeval):
            partder[j] = self.like_part_der_single(param_class,var,theta,dx[j])
        
        fig, ax = plt.subplots()
        ax.set_yscale('symlog')
        cond = ~np.isnan(partder)
        ax.plot(multrange[cond],partder[cond],color=color_palette.next())
        ax.set_xlabel(r"$d$%s$/ \Delta$%s$_{\mathrm{tot}}$"%(varlabel,varlabel))
        ax.set_ylabel(r"$d\log \mathcal{L}/d$%s"%(varlabel))
        ax.minorticks_on()
        plt.tight_layout()
        imgname = op.join(imgdir,"%s_dlogLd%s_step.%s"%(id,var,imgtype))
        fig.savefig(imgname,bbox_inches='tight',dpi=300)

    def plot_like_der_var(self,param_class,var,theta,varlabel,id,numeval=200,
                          imgtype='png',imgdir=''):
        vararr, partder = self.like_part_der_mult(param_class,var,theta,numeval=numeval)
        fig, ax = plt.subplots()
        ax.set_yscale('symlog')
        cond = ~np.isnan(partder)
        ax.plot(vararr[cond],partder[cond],color=color_palette.next())
        ax.set_xlabel(varlabel)
        ax.set_ylabel(r"$d\log \mathcal{L}/d$%s"%(varlabel))
        ax.minorticks_on()
        imgname = op.join(imgdir,"%s_dlogLd%s_vs_%s.%s"%(id,var,var,imgtype))
        fig.savefig(imgname,bbox_inches='tight',dpi=300)