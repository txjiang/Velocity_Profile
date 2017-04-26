import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import integrate

df_vel = pd.read_excel("VelocityAlex.xls")

df_vel_array = (2*5.202*df_vel/0.002297)**0.5*0.3048

writer = pd.ExcelWriter('outputVelocity_Alex.xlsx', engine='xlsxwriter')
df_vel_array.to_excel(writer,sheet_name='Sheet1')
writer.save()