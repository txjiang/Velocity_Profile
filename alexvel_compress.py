import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import integrate

df_vel = pd.read_excel("VelocityAlex.xls")

df_vel_array = 340*((2/0.4)*(((df_vel + 407.3016)/407.3016)**(0.4/1.4) - 1))**0.5

writer = pd.ExcelWriter('outputVelocity_compress_Alex.xlsx', engine='xlsxwriter')
df_vel_array.to_excel(writer,sheet_name='Sheet1')
writer.save()

