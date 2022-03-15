# LSM-type_fitting
A stripped down version of ECIF specifically for batch-fitting mixed ionic-electronic conductor (MIEC) impedance spectra

The custom circuit applies for a symmetric electrode|MIEC|substrate|MIEC|electrode geometry, when electrode and MIEC electronic resistance is negligible, MIEC ionic resistance is non-negligible, and the substrate is electron-blocking. Two versions are included:

(1) CCfit: MIEC chemical capacitance is modeled by a pure capacitor

(2) CCfit_CPE: MIEC chemical capacitance is modeled by a constant phase element


To use:

-Install all required Python libraries, listed in top of files.

-Run "python CCfit.py" from command line.

-Enter initial guesses as prompted for first impedance file.

-Select impedance files to batch-fit. Files should be tab-delimited, with frequency, Z', and Z" columns, with no headers. Batch fitting will use results from previous file as initial guess for the subsequent file.

-If extracting temperature and pressure from a log file, select it when prompted. Otherwise, leave empty.

-Fitting routine will save .png images of the correlation plots and impedance spectra fits as each fitting is completed. Results are saved at the end.
