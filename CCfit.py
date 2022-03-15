import numpy as np
from scipy import optimize
from tkinter.filedialog import askopenfilenames, askopenfilename
import pandas as pd
from datetime import datetime
import sys
import os
import pytz
import matplotlib.pyplot as plt

#This code fits complex circuit with 6 element, 6 parameter model
#r0 = substrate resistance
#rs = surface resistance
#cs = surface capacitance
#ri = film resistance
#ci = film capacitance
#ce = electronic capacitance

# define impedance model
def model(parameters, x):
    r0, rs, cs, ri, ci, ce = parameters
    w=2j*np.pi*x
    zd=1/(w*ce)
    za=rs/(1+rs*cs*w)
    a=np.sqrt(w*ri*ci)
    return (ri*zd*np.tanh(a)+zd*za*a)/((ri+za*zd*a*a/ri)*np.tanh(a)+(za+zd)*a)+r0

# define lower and upper bound for parameters
def bound():
    return [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]

# define residual function for fitting
def res_vec(parameters, x, y):
    y_cal = model(parameters, x)
    z = y-y_cal
    modulus = np.sqrt(y.real ** 2 + y.imag ** 2)
    a = z.real/modulus
    b = z.imag/modulus
    res = np.concatenate((a, b), axis=None)
    return res

# least-square fitting
def custom_fitting(freq, data, parameters):
    x_data = np.array(freq)
    y_data = np.array(data)
    parameters = np.array(parameters)
    fitting_result = optimize.least_squares(res_vec, parameters, jac='3-point', bounds=(bound()), args=(x_data, y_data))
    
    #split real and imaginary residuals for bootstrap
    length = int(len(fitting_result.fun) / 2)
    real_error = np.std(fitting_result.fun[:length])
    imag_error = np.std(fitting_result.fun[length:])
    real_mean = np.mean(fitting_result.fun[:length])
    imag_mean = np.mean(fitting_result.fun[length:])
    modulus = np.sqrt(data.real**2 + data.imag**2)
    
    sample = 100
    total_results = np.zeros((sample,6))
    total_results[0] = fitting_result.x
    i = 1
    while i<sample:
        real_residuals = np.random.normal(real_mean, real_error, size=length)*modulus
        imag_residuals = np.random.normal(imag_mean, imag_error, size=length)*modulus
        real_boot = y_data.real + real_residuals
        imag_boot = y_data.imag + imag_residuals
        boot_data = real_boot + 1j*imag_boot
        boot_fit = optimize.least_squares(res_vec, total_results[i-1], jac='3-point', bounds=(bound()), args=(x_data, boot_data))
        total_results[i] = boot_fit.x
        i += 1
    mean = np.mean(total_results, axis=0)
    std = np.std(total_results, axis=0)
    corr = np.corrcoef(np.transpose(total_results))
    cov = np.cov(np.transpose(total_results))
    return mean, std, corr, cov

#extract data from text file
def fileread(file):
    expData = np.array(pd.read_csv(file, delimiter='\t', header=None))
    freq = expData[:, 0]
    data = expData[:, 1] + expData[:, 2]*1j
    return freq, data

#extract temperature and pressure from PO2 file based on impedance file modification date
def temp_pressure(file_list, PO2_file):
    if sys.platform.startswith('linux'):
        linux_flag = True
    else:
        linux_flag = False
    time_list = list()
    for file in file_list:
        if linux_flag: #Fix for linux reading file modification time in UTC
            cst = pytz.timezone('US/Central')
            dst = cst.localize(datetime.fromtimestamp(os.path.getmtime(file))).dst()
            offset = 6 * 60 * 60 - dst.total_seconds()
        else:
            offset = 0   
        time_list.append(os.path.getmtime(file) + offset)
    data = np.array(pd.read_csv(PO2_file, delimiter='\t'))
    date = data[:, 0]
    time = data[:, 1]
    tstamp = np.zeros(date.size)
    for i in range(tstamp.size):
        t = datetime.strptime(date[i] + ' ' + time[i], '%m/%d/%Y %I:%M:%S %p')
        tstamp[i] = datetime.timestamp(t)
    temp_all = data[:, 2]
    PO2_all = data[:, 3]
    temp_select = np.zeros((len(time_list), 1))
    PO2_select = np.zeros((len(time_list), 1))
    for i in range(len(time_list)):
        ind = 0
        while time_list[i] > tstamp[ind]:
            ind += 1
        temp_select[i] = temp_all[ind - 1]
        PO2_select[i] = PO2_all[ind - 1]
    return np.concatenate((temp_select, PO2_select), axis=1)

#batch impedance fitting
def batch(parameters):
    cwd = os.getcwd()
    variables = 'r0, rs, cs, ri, ci, ce'
    files = askopenfilenames(title="Choose impedance files", filetypes=[("", "*.txt")])
    path = os.path.dirname(files[0])
    os.mkdir(path + '/fit_figures')
    os.mkdir(path + '/fit_figures' + '/Nyquist')
    os.mkdir(path + '/fit_figures' +'/Bode')
    os.mkdir(path + '/fit_figures' +'/Correlation')
    os.mkdir(path + '/fit_results')
    PO2_file = askopenfilename(title="Choose PO2 file", filetypes=[("", "*.txt")])
    count = len(files)
    results = np.zeros((count,6))
    error = np.zeros((count,6))
    file_list = list('')
    count = 0
    for file in files:
        freq, data = fileread(file)
        fit, err, corr, cov = custom_fitting(freq, data, parameters)
        
        model_fit = model(fit, freq)
        theta_data = np.arcsin(data.imag / np.abs(data)) * 180 / np.pi
        theta_model = np.arcsin(model_fit.imag / np.abs(model_fit)) * 180 / np.pi
        parameters = fit #use previous fit for next guess
        
        file_list.append(file)
        results[count] = parameters
        error[count] = err
        count += 1
        print('{}/{} complete'.format(count, len(files)))
        fig_name = str(count) + '.png'
        
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        plt.plot(data.real, -data.imag, 'ok')
        plt.plot(model_fit.real, -model_fit.imag, '-r')
        ax.legend(['Exp data', 'Fit'])
        plt.savefig(path + '/fit_figures/' + '/Nyquist/' + fig_name)
        plt.close()
        
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.plot(np.log10(freq), np.log10(np.abs(data)), 'ok')
        ax1.plot(np.log10(freq), np.log10(np.abs(model_fit)), '-r')
        ax1.legend(['Exp data', 'Fit'])
        ax1.set_ylabel('log(|Z|) \Ohm')
        
        ax2.plot(np.log10(freq), theta_data, 'ok')
        ax2.plot(np.log10(freq), theta_model, '-r')
        ax2.set_xlabel('log(f) \Hz')
        ax2.set_ylabel('theta \deg')
        plt.savefig(path + '/fit_figures/' + '/Bode/' + fig_name)
        plt.close()
        
        plt.imshow(corr)
        plt.clim(-1, 1)
        plt.colorbar()
        plt.tick_params(axis='y', labelleft=False, left=False) 
        plt.tick_params(axis='x', labelbottom=False, bottom=False)
        plt.savefig(path + '/fit_figures/' + '/Correlation/' + fig_name)
        plt.close()
        
        cov_name = str(count) + '.csv'
        np.savetxt(path + '/fit_results/' + cov_name, results, delimiter=',', 
                   newline='\n', header=variables)

    os.chdir(path + '/fit_results')
    np.savetxt('fit_report.csv', results, delimiter=',', newline='\n', header=variables)
    np.savetxt('fit_error.csv', error, delimiter=',', newline='\n', header=variables)
            
    f = open('fit_files.csv','w')
    for i in range(count):
        f.write("%s\n" % file_list[i])
    f.close()
    
    if PO2_file!='':
        np.savetxt('temp_pressure.csv', temp_pressure(file_list, PO2_file), 
                   delimiter=',', newline='\n', header='T [C], P [atm]')
    else:
        print('No PO2 file selected')
        
    os.chdir(cwd)

def input_check(var):
    if var=='':
        return 1.0
    else:
        return float(var)
    
#for running from command line: specify initial guesses and run batch fitting
if __name__=='__main__':
    print('Input initial guesses for the following:')
    print('Blanks are interpreted as 1')
    r0 = input_check(input('r0 (substrate) = '))
    rs = input_check(input('rs (surface) = '))
    cs = input_check(input('cs (surface) = '))
    ri = input_check(input('ri (film) = '))
    ci = input_check(input('ci (film) = '))
    ce = input_check(input('ce (electronic) = '))
    param = np.array([r0, rs, cs, ri, ci, ce], dtype=float)
    batch(param)
    
