import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import integrate

df_vel = pd.read_excel("NozzleVelocityProfile.xls",skiprows = 1)
del df_vel['Distance']
df_vel_20psi = df_vel.iloc[[0,1,2]]
df_vel_dist = np.array(df_vel_20psi.mean().index.tolist())/1000
df_dist_temp = np.sort(-df_vel_dist)
df_dist_new = np.append(df_dist_temp, df_vel_dist)


df_vel_20psi_mean = 340*((2/0.4)*(((np.array(df_vel_20psi.mean().tolist()) + 407.3016)/407.3016)**(0.4/1.4) - 1))**0.5
df_vel_20psi_mean_sort = np.sort(df_vel_20psi_mean)
df_vel_20psi_mean = np.append(df_vel_20psi_mean_sort, df_vel_20psi_mean)

df_vel_40psi = df_vel.iloc[[3,4,5]]
df_vel_40psi_mean = 340*((2/0.4)*(((np.array(df_vel_40psi.mean().tolist()) + 407.3016)/407.3016)**(0.4/1.4) - 1))**0.5
df_vel_40psi_mean_sort = np.sort(df_vel_40psi_mean)
df_vel_40psi_mean = np.append(df_vel_40psi_mean_sort, df_vel_40psi_mean)

df_vel_50psi = df_vel.iloc[[6,7,8]]
df_vel_50psi_mean = 340*((2/0.4)*(((np.array(df_vel_50psi.mean().tolist()) + 407.3016)/407.3016)**(0.4/1.4) - 1))**0.5
df_vel_50psi_mean_sort = np.sort(df_vel_50psi_mean)
df_vel_50psi_mean = np.append(df_vel_50psi_mean_sort, df_vel_50psi_mean)

df_vel_30psi = df_vel.iloc[[9,10,11]]
df_vel_30psi_mean = 340*((2/0.4)*(((np.array(df_vel_30psi.mean().tolist()) + 407.3016)/407.3016)**(0.4/1.4) - 1))**0.5
df_vel_30psi_mean_sort = np.sort(df_vel_30psi_mean)
df_vel_30psi_mean = np.append(df_vel_30psi_mean_sort, df_vel_30psi_mean)

print (df_vel_20psi_mean)
centre_20_psi = max(df_vel_20psi_mean)
print (df_vel_30psi_mean)
centre_30_psi = max(df_vel_30psi_mean)
print (df_vel_40psi_mean)
centre_40_psi = max(df_vel_40psi_mean)
print (df_vel_50psi_mean)
centre_50_psi = max(df_vel_50psi_mean)
centre_list = [centre_20_psi,centre_30_psi,centre_40_psi,centre_50_psi]

dist_array = np.arange(-40/1000,40/1000, 1/10000)
vel20psi = []
vel40psi = []
vel50psi = []
vel30psi = []

def normpdf(factor,mu, sigma,x):
    n_pdf = factor*(1/(2*sigma**2*math.pi)**0.5)*math.exp(-(x-mu)**2/(2*sigma**2))
    return n_pdf

for item in dist_array:
    factor_20 = 1.25
    mu_20 = 0
    sigma_20 = 0.005
    n_pdf_20 = normpdf(factor_20, mu_20, sigma_20, item)
    vel20psi.append(n_pdf_20)
    factor_40 = 2.0
    mu_40 = 0
    sigma_40 = 0.005
    n_pdf_40 = normpdf(factor_40, mu_40, sigma_40, item)
    vel40psi.append(n_pdf_40)
    factor_50 = 2.2
    mu_50 = 0
    sigma_50 = 0.005
    n_pdf_50 = normpdf(factor_50, mu_50, sigma_50, item)
    vel50psi.append(n_pdf_50)
    factor_30 = 1.75
    mu_30 = 0
    sigma_30 = 0.005
    n_pdf_30 = normpdf(factor_30, mu_30, sigma_30, item)
    vel30psi.append(n_pdf_30)

plt.figure()
plt.plot(df_dist_new, df_vel_20psi_mean)
plt.plot(dist_array, vel20psi)
plt.xlabel('Distance (m)', fontsize = 25)
plt.ylabel('Velocity (m/s)', fontsize = 25)
plt.title('Velocity Profile at 10 cm below Nozzle under 20 PSIG', fontsize = 25)
plt.legend(['Measured Data', 'Function Fit'], fontsize = 25)
plt.annotate('1.25*N(0, 0.005)', [0.005, 100], fontsize = 25)
plt.show()

plt.figure()
plt.plot(df_dist_new, df_vel_40psi_mean)
plt.plot(dist_array, vel40psi)
plt.xlabel('Distance (m)', fontsize = 25)
plt.ylabel('Velocity (m/s)', fontsize = 25)
plt.title('Velocity Profile at 10 cm below Nozzle under 40 PSIG', fontsize = 25)
plt.legend(['Measured Data', 'Function Fit'], fontsize = 25)
plt.annotate('2.0*N(0, 0.005)', [0.005, 100], fontsize = 25)
plt.show()

plt.figure()
plt.plot(df_dist_new, df_vel_50psi_mean)
plt.plot(dist_array, vel50psi)
plt.xlabel('Distance (m)', fontsize = 25)
plt.ylabel('Velocity (m/s)', fontsize = 25)
plt.title('Velocity Profile at 10 cm below Nozzle under 50 PSIG', fontsize = 25)
plt.legend(['Measured Data', 'Function Fit'], fontsize = 25)
plt.annotate('2.2*N(0, 0.005)', [0.005, 100], fontsize = 25)
plt.show()

plt.figure()
plt.plot(df_dist_new, df_vel_30psi_mean)
plt.plot(dist_array, vel30psi)
plt.xlabel('Distance (m)', fontsize = 25)
plt.ylabel('Velocity (m/s)', fontsize = 25)
plt.title('Velocity Profile at 10 cm below Nozzle under 30 PSIG', fontsize = 25)
plt.legend(['Measured Data', 'Function Fit'], fontsize = 25)
plt.annotate('1.75*N(0, 0.005)', [0.005, 100], fontsize = 25)
plt.show()

x_20_psi = lambda x : 2*math.pi*x*(1.25*(1/(2*0.005**2*math.pi)**0.5)*math.exp(-(x-0)**2/(2*0.005**2)))
q_20_psi = integrate.quad(x_20_psi, 0, 0.02)
x_40_psi = lambda x : 2*math.pi*x*(2.0*(1/(2*0.005**2*math.pi)**0.5)*math.exp(-(x-0)**2/(2*0.005**2)))
q_40_psi = integrate.quad(x_40_psi, 0, 0.02)
x_50_psi = lambda x : 2*math.pi*x*(2.2*(1/(2*0.005**2*math.pi)**0.5)*math.exp(-(x-0)**2/(2*0.005**2)))
q_50_psi = integrate.quad(x_50_psi, 0, 0.02)
x_30_psi = lambda x : 2*math.pi*x*(1.75*(1/(2*0.005**2*math.pi)**0.5)*math.exp(-(x-0)**2/(2*0.005**2)))
q_30_psi = integrate.quad(x_30_psi, 0, 0.02)
q_all = np.array([q_20_psi[0],q_30_psi[0], q_40_psi[0], q_50_psi[0]])
outlet_area = np.pi*(8.5/2000)**2
vel_all = q_all/outlet_area
print (vel_all)
print (q_all)

plt.figure()
plt.scatter([20,30,40,50],q_all)
plt.xlabel('Pressure (PSIG)', fontsize = 25)
plt.ylabel('Flow Rate (m^3/s)', fontsize = 25)
plt.title('Flow Rate of Nozzle versus Pressure', fontsize = 25)
plt.show()

plt.figure()
plt.scatter(centre_list, q_all)
plt.xlabel('Centre Line Velocity (m/s)', fontsize = 25)
plt.ylabel('Flow Rate (m^3/s)', fontsize = 25)
plt.title('Flow Rate of Nozzle versus Centre Line Velocity', fontsize = 25)
plt.show()


#print (df_vel_20psi_mean)
#print (df_vel_40psi_mean)
#print (df_vel_50psi_mean)

'''for item in dist_array:
    vel20psi.append(vel_20_psi_fit[0]*item**2 + vel_20_psi_fit[1]*item + vel_20_psi_fit[2])
    vel40psi.append(vel_40_psi_fit[0]*item**2 + vel_40_psi_fit[1]*item + vel_40_psi_fit[2])
    vel50psi.append(vel_50_psi_fit[0]*item**2 + vel_50_psi_fit[1]*item + vel_50_psi_fit[2])'''

#print (vel20psi)
#print (vel_40_psi_fit)
#print (vel_50_psi_fit)
#print (df_vel_20psi_mean)

