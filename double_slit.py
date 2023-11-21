
import numpy as np
import LT.box as B
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
from sklearn.linear_model import LinearRegression
from numpy.fft import fft

#prelab
#graph intensity from a single slit as function of distance from center
I0 = B.Parameter(1., 'I0') # Max. intensity, initial value set to 1
#x in m
def intsingle(x):
    x0=0
    lam= 500/(10**9) #m
    D=0.01/1000 #m
    L=500/(1000) #m
    theta=(x-x0)/L
    phi=(2*np.pi/lam)*D*np.sin(theta)
    I = I0()*((np.sin(phi/2))/(phi/2))**2
    return I

plt.plot(np.linspace(-100/1000, 100/1000,1000),intsingle(np.linspace(-100/1000, 100/1000,1000)),label="Single Slit")
#plt.ylim(-1e-14+1,0.5e-14+1)
plt.title("Intensity of a Single Slit vs. Distance from center")
plt.ylabel("Intensity")
plt.xlabel("Distance from center (m)")


def intdouble(x):
    x0=0
    lam= 500/(10**9) #m
    D=0.01/1000
    L=500/(1000)
    S=0.02/1000
    k = 2.*np.pi/lam                # lam is the wavelength
    phi = k*D*np.sin((x-x0)/L)  # L is the distance between the double slit and the detector slit
    psi = k*S*np.sin((x-x0)/L)
    I = I0()*(np.sin(phi/2.)/(phi/2.))**2 * np.cos(psi/2.)**2
    return I


plt.plot(np.linspace(-100/1000, 100/1000,1000),intdouble(np.linspace(-100/1000, 100/1000,1000)),label="Double Slit")
plt.title("Intensity of a Double Slit vs. Distance from center")
plt.ylabel("Intensity")
plt.xlabel("Distance from center (m)")
#plt.ylim(-8e-14+1,0.5e-14+1)

plt.legend()


#%%

#changing them a bit, making some parameters.
L= 500/1000 #m
I0 = B.Parameter(1, 'I0') # Max. intensity, initial value set to 1
D = B.Parameter(0.356/1000, 'D')  # slit width in m, initial value set to 0.1mm
x0 = B.Parameter(4.75/1000, 'x0') # location of maximum in position, initial value set to 0.01 radian, you need to adjust this to your data


#x in m
lam= 670/(10**9) #m


def intsingle(x): 
    ang=(x-x0())/L
    phi=(2*np.pi/lam)*D()*np.sin(ang)
    I = I0()*((np.sin(phi/2))/(phi/2))**2
    return I
def theta(x):
    return (x-x0())/L
#plt.plot(np.linspace(9.99, 10.01,1000),intsingle(np.linspace(9.99, 10.01,1000)))
#plt.title("Intensity vs. Distance from center")
#plt.ylabel("Intensity")
#plt.xlabel("Distance from center (m)")
#%%

D = B.Parameter(1.e-4, 'D')  # slit width in m, initial value set to 0.1mm
S = B.Parameter(3.e-4, 'S')  # slit separation in m, initial value set to 0.3mm
x0 = B.Parameter(1.e-2, 'x0') # location of maximum in position, initial value set to 0.01 radian, you need to adjust this to your data
I0 = B.Parameter(1., 'I0') # Max. intensity, initial value set to 1
def intdouble(x):
    L= 500/1000 #m
    k = 2.*np.pi/lam                # lam is the wavelength
    phi = k*D()*np.sin((x-x0())/L)  # L is the distance between the double slit and the detector slit
    psi = k*S()*np.sin((x-x0())/L)
    I = I0()*(np.sin(phi/2.)/(phi/2.))**2 * np.cos(psi/2.)**2
    return I



#%%
#importing data


f=B.get_file("leftslit.dat")

z=(B.get_data(f,"XL"))/1000 #m
V=B.get_data(f,"YL")

D.set(0.356/1000)
I0.set(1)
x0.set(4.75000001/1000)
angle=theta(z)

B.plot_exp(angle, V/np.max(V),dy=dV)
plt.title("Intensity of Left Slit vs. Angle from Center")
plt.ylabel("Intensity (V)")
plt.xlabel("Angle")
plt.show()
F = B.genfit(intsingle, [I0,x0,D], x = z, y = V)
plt.show()

# Gaussian function
# def gaussian(x, amplitude, mean, stddev):
#     return amplitude * np.exp(-(x - mean)**2 / (2 * stddev**2))

# # Fitting the Gaussian function to the data with adjusted initial guess and bounds
# initial_guess = [551, 0, 0.005]  # Initial guess for amplitude, mean, and standard deviation
# bounds = ([7, -0.01, 0], [600, 0.01, 0.01])  # Lower and upper bounds for parameters
# params, covariance = curve_fit(gaussian, z, V, p0=initial_guess, bounds=bounds)

# # Extracting fitted parameters
# amplitude, mean, stddev = params

# # Generating the fitted curve
# x_fit = np.linspace(-0.001, 0.001, 100)
# y_fit = gaussian(x_fit, amplitude, mean, stddev)

# # Plotting the data with error bars and the fitted curve
# plt.errorbar(z, V, yerr=dV, fmt='o', label='Data with Error Bars')
# plt.plot(z, V, label='Fitted Gaussian Curve', color='red')

#angle[19]=0.000000000000000000000000000000000000000000000000000000001


B.plot_exp(angle, V/np.max(V), dy=dV, color='purple')
plt.plot(angle,V, color='red')
plt.title("Left Slit Intensity vs. Angle")
plt.ylabel("Intensity")
plt.xlabel("Angle")

#%%
#voltage uncertainty
#dV=0.02/0.569
#uncertainty in screw: 
dx=0.02/1000
dtheta=(1/L)*dx
dphi=(((2*np.pi)/lam)*D()*np.cos(theta(z)))*dtheta
phi=((2*np.pi)/lam)*D()*np.sin(theta(z))
dV=np.abs((4*I0())*(((-2)*(phi**(-3))*((np.sin(phi/2))**2))+((np.sin(phi/2))*(np.cos(phi/2))*(phi**(-2)))))*dphi

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-(x - mean)**2 / (2 * stddev**2))

# Fitting the Gaussian function to the data with adjusted initial guess and bounds
initial_guess = [551, 0, 0.005]  # Initial guess for amplitude, mean, and standard deviation
bounds = ([7, -0.01, 0], [600, 0.01, 0.01])  # Lower and upper bounds for parameters
params, covariance = curve_fit(gaussian, z, V, p0=initial_guess, bounds=bounds)

# Extracting fitted parameters
amplitude, mean, stddev = params

# Generating the fitted curve
x_fit = np.linspace(-0.001, 0.001, 100)
y_fit = gaussian(x_fit, amplitude, mean, stddev)

# Plotting the data with error bars and the fitted curve
plt.errorbar(z, V/np.max(V), yerr=dV, label='Data with Error Bars')
plt.plot(z, V/np.max(V), label='Fitted Gaussian Curve', color='red')


B.plot_exp(z, V/np.max(V),dy=dV, color='purple')
# plt.plot(angle,V)
plt.title("Intensity of Left Slit vs. Position")
plt.ylabel("Intensity (V)")
plt.xlabel("Position (m)")
# F = B.genfit(intsingle, [I0,x0,D], x = z, y = V)

#%%
f=B.get_file("rightslit.dat")

z=(B.get_data(f,"XR"))/1000 #m
V=B.get_data(f,"YR")
# z=z[2:41]
# V=V[2:41]


# ang=(x-x0())/L
# phi=(2*np.pi/lam)*D()*np.sin(ang)
# I = I0()*((np.sin(phi/2))/(phi/2))**2


# V=V/0.4380000000000000000001


D.set(0.075/1000)
I0.set(1)
x0.set(5.7500001/1000)

# th = np.linspace(0, 0.01, 1000) # create an array of 1000 equally spaced values between tmin and tmax
# plt.plot(z,V)
# plt.show()

# B.plot_line(th, intsingle(th))  # this should plot the calculated line with the new parameter values


# F = B.genfit(intsingle, [I0,x0,D], x = z, y = V)


#voltage uncertainty
#dV=0.02/0.569
#uncertainty in screw: 
dx=0.02/1000
dtheta=(1/L)*dx
dphi=(((2*np.pi)/lam)*D()*np.cos(theta(z)))*dtheta
phi=((2*np.pi)/lam)*D()*np.sin(theta(z))
dV=np.abs((4*I0())*(((-2)*(phi**(-3))*((np.sin(phi/2))**2))+((np.sin(phi/2))*(np.cos(phi/2))*(phi**(-2)))))*dphi


B.pl.title("Right Slit Intensity vs. Position")
B.pl.ylabel("Intensity (V)")
B.pl.xlabel("Position (m)")
B.plot_exp(z, V/np.max(V),dy=dV)
B.pl.plot(angle,V/np.max(V), dV)

# F = B.genfit(intsingle, [I0,x0,D], x = z, y = V)
#%%
angle=theta(z)
B.plot_exp(angle, V/np.max(V),dy=dV)
plt.title("Right Slit Intensity vs. Angle")
plt.ylabel("Intensity (V)")
plt.xlabel("Angle")
#%%
plt.plot(z,dV,".")
plt.ylim(0,0.01)
#%%

# #finding width differently. 
# #left
# 0.0005 - 0.0095 = 0.009

# #right
# 0.00125-0.009=0.007749999999999999

#%%
f=B.get_file("doubleslit.dat")

z=(B.get_data(f,"XD"))/1000
V=(B.get_data(f,"YD"))
angle=theta(z)
plt.plot(z,V/np.max(V))
plt.show()


th = np.linspace(0, 0.01, 1000) # create an array of 1000 equally spaced values between tmin and tmax
D.set(0.08/1000)  # change the value of D to 0.08 mm
S.set(0.45e-3)  # change the value of S to 0.31 mm
I0.set(1.8560001)
x0.set(5.25000001/1000)
# B.plot_line(th, intdouble(th))  # this should plot the calculated line with the new parameter values

D.set(0.04/1000)  # change the value of D to 0.08 mm
S.set(0.31e-3)  # change the value of S to 0.31 mm
I0.set(1.8560001)
x0.set(5.25000001)
#i_fit = B.genfit(intdouble, [D,S,x0], x=z, y = V, y_err = dV) # assuming th_exp, I_exp, dI_exp


B.plot_exp(z, V/np.max(V))
# i_fit = B.genfit(intdouble, [D,S,x0,I0], x=z, y = V) # assuming th_exp, I_exp, dI_exp
plt.title("Double Slit Intensity vs. Position")
plt.ylabel("Intensity (V)")
plt.xlabel("Position (m)")

#%%
angle=theta(z)
B.plot_exp(angle, V/np.max(V))
plt.title("Intensity of Double Slit vs. Angle from Center")
plt.ylabel("Intensity (V)")
plt.xlabel("Angle from Center (rad)")

B.plot_line(angle, V/np.max(V))
#%%                                                                            # are your exp. data
dx=0.02/1000
dtheta=(1/L)*dx
k=(2*np.pi)/lam
dphi=k*D()*np.cos(theta(z))*dtheta
dpsi=k*S()*np.cos(theta(z))*dtheta
psi=k*S()*np.sin(theta(z))
phi=k*D()*np.sin(theta(z))
didphi=(4*I0()*(np.cos(psi/2))**2)*(((-8)*phi**(-3))*((np.sin(phi/2))**2)+((np.sin(phi/2))*(np.cos(phi/2))*(4*phi**(-2))))
didpsi=(4*I0())*(4*phi**(-2))*((np.sin(phi/2))**2)*((np.cos(psi/2))*(-np.sin(psi/2)))
dI=np.sqrt(((didphi)**2*(dphi)**2)+((didpsi)**2*(dpsi)**2))
#%%
fourier=fft(np.sqrt(intdouble(z)))