import numpy as np 
import matplotlib.pyplot as plt 


# Conver this to SI units
# V_CM = 500mV
# V_SYBT = 300mV
# f_sw = 3.125MHz
# T_int = 3/390625s
# C_int = 5pF

V_CM = 0.5              # V
V_STBY = 0.3            # V
f_sw = 3.125 * 10**6    # Hz
T_int = 3/390625        # s
C_int = 5 * 10**(-12)    # F

coeff = (V_CM - V_STBY) * f_sw * T_int / C_int
print(coeff)

filename = "../data/super_res/impedance.npy"
arr = np.load(filename) / coeff 



plt.imshow(arr, cmap='Greys')
plt.colorbar()

# whole_mean = np.mean(myimage)
# whole_std = np.std(myimage)


# # np.save("../log/raw_impedance_array_image", myimage)
# plt.figure(1)
# plt.imshow(myimage, cmap='Greys', vmin=(whole_mean-(5*whole_std)), vmax=(whole_mean+(5*whole_std)))
# plt.xlabel("Column [px]")
# plt.ylabel("Row [px]")


plt.show()