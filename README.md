# MCSED
The official documentation is hosted at [RTFM](https://mcsed.readthedocs.io/en/latest/index.html).

## Background
MCSED models the optical, near-infrared and infrared spectral energy distribution (SED) of galactic systems.  In light of the fact that there are so many such codes already publicly available, we describe the motivation for MCSED and highlight areas in which this code stands out from the crowd.  First of all, galaxies over cosmic time span a wide range of parameters related to their baryonic content including total stellar mass, gas and stellar metallcity, dust mass and distribution, and star formation history.  This large variation for the totality of all galaxies makes it extremely difficult to develope a general enough SED fitting code to work for all systems.  Instead, most codes are taylored to work best for specific populations.  

MCSED targets galaxies at cosmic noon (z ~ 1-3) that are selected via their emission lines either in the rest-frame optical or ultraviolet wavelengths.  These sources are drawn from the 3DHST survey (http://3dhst.research.yale.edu/Home.html) as well as the HETDEX survey (http://hetdex.org/).  Initial SED fitting efforts revealed that these systems are forming stars at a rate of 1-1000 solar masses per year and have total stellar masses of 10^8-10^11 ([Bowman et al. 2019](https://ui.adsabs.harvard.edu/abs/2019ApJ...875..152B/abstract)).  

## Installation
To acquire and install this code, simply move to a directory where you would like it stored and type:

        git clone https://github.com/wpb-astro/MCSED.git

A directory called "MCSED" will be created containing all of the necessary files for the program.  This is a python based code and does require a few standard python based packages.  All of the packages required can be found in the Anaconda distribution environment.  To install Anaconda, see:
https://docs.anaconda.com/anaconda/install/

## How to Run MCSED
The primary script is run_mcsed_fit.py, which can be called from the command line with input arguments.  To view the input arguments, simply type:

        python run_mcsed.py -h

And you will find a help menu like this.

        -f FILENAME, --filename FILENAME: File to be read for galaxy data

        -o OUTPUT_FILENAME, --output_filename OUTPUT_FILENAME: Output filename for given run

        -p, --parallel: Running in parallel?

        -t, --test: Test mode with mock galaxies

        -tf TEST_FIELD, --test_field TEST_FIELD: Test filters will match the given field

        -no NOBJECTS, --nobjects NOBJECTS: Number of test objects

        -s SSP, --ssp SSP: SSP models, default fsps

        -i ISOCHRONE, --isochrone ISOCHRONE: Isochrone for SSP model, e.g. padova

        -sfh SFH, --sfh SFH: Star formation history, e.g. constant

        -dl DUST_LAW, --dust_law DUST_LAW: Dust law, e.g. calzetti

        -de, --dust_em: Dust emission class, e.g., DL07 (Draine & Li 2007)
                        or False (if dust emission should be ignored)

        -aeb, --assume_energy_balance: If true, normalization of dust IR emission based on attenuation amount

        -z METALLICITY, --metallicity METALLICITY: Fixed metallicity for SSP models (0.019 is solar), 
                                                   or False if free parameter

        -nw NWALKERS, --nwalkers NWALKERS: Number of walkers for EMCEE

        -ns NSTEPS, --nsteps NSTEPS: Number of steps for EMCEE

        -lu LOGU, --logU LOGU: Ionization Parameter for nebular gas

        -ism, --ISM_correct_coords: If a coordinate system is given, MW dust correction will be performed

        -igm, --IGM_correct: If selected, Madau statistical IGM correction will be done (affecting wavelengths up to rest-frame Lyman-alpha)

All of the available options for MCSED are found in [config.py](https://github.com/wpb-astro/MCSED/blob/master/config.py).  Here we break down the most important of those: 

        ssp = 'fsps'          
        isochrone = 'padova' 
        sfh = 'binned_lsfr'
        dust_law = 'calzetti'
        dust_em = 'DL07'   
        metallicity = 0.0077

The current version of MCSED includes the single stellar population (SSP) models from FSPS using Padova isochrones and users can easily substitute their own SSP grid. To build a composite stellar population the user can set the type of star formation history as well as dust attenuation law to go from instrinsic emission to observed flux. We include options for six star formation histories and four dust attenuation laws (and one extinction law). Another key feature in SED fitting is the metallicity of the SSPs.  We offer two options: fixed metallicity or treating stellar metallicity as a free model parameter.

To run MCSED for a set of objects, simply input a file that has the format: 

        Field     ID     z
        COSMOS   18945  2.188
        COSMOS   13104  2.292

The call would look like:

        python run_mcsed.py -f PATH/FILENAME 

Users can take advantage of multiple available cores by including the -p option in the command line call, which will initialize the parallel fitting mode. This call will distribute the total number of objects across (N-i) cores, where N is the total number of cores on the machine, and i is the number of cores which should not be used in the calculation (specified by the "reserved_cores" keyword in config.py). This mode is extremely useful when fitting large galaxy samples.

The output files are stored in a directory called "outputs". Several output files are available and can be turned on/off via the "output_dict" dictionary options in config.py. These include: a summary table of best-fit model parameters (and associated confidence intervals); full posterior distributions for model parameters; the best-fit SED model; modeled and observed photometric fluxes, emission lines, and absorption line indices; an age-weighted plot of the SSP spectra used in the fitting; and a summary diagnostic figure (example shown below).
 
<p align="center">
  <img src="example_triangle.png" width="650"/>
</p>

## Authors

* Greg Zeimann, Hobby Eberly Telescope, UT Austin, grzeimann@gmail.com or gregz@astro.as.utexas.edu
* William P. Bowman, Penn State University, wpb.astro@gmail.com or bowman@psu.edu
* Gautam Nagaraj, Penn State University, gxn75@psu.edu

## Dependencies

* emcee, tested with version '2.1.0', currently there are errors using '2.2.1'
* corner, tested with version '2.0.1'
* seaborn, tested with version '0.8.1'
* astropy, tested with version '2.0.6'
* matplotlib, tested with version '2.1.2'
* scipy, tested with version '1.0.0'
* dustmaps (if a correct for foreground Milky Way dust extinction is desired)

