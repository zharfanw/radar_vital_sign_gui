from scipy.io import savemat
import numpy as np
import datetime
# import the time module
import time

datanya = np.array([])
dat = 0
while(dat < 10):
    datanya = np.append(datanya,np.array([dat]))
    dat = dat + 1

# print("Try Save MATFILE")
# mdic = {"rec_breath": datanya , "label": "experiment"}
# print(mdic)
# filenamenya = "rec_1.mat"
# savemat(filenamenya, mdic)
seconds = round(time.time_ns())

print("Seconds since epoch =", seconds)	