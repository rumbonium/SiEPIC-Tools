import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import os
import numpy as np

def cartesian_product(arrays):
    la = len(arrays)
    dtype = np.find_common_type([a.dtype for a in arrays], [])
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)

def straightWaveguide(wavelength, width, thickness, angle):
    #load regression
    LR_straight = joblib.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'R_straight_update.joblib'))
    #

    # Santize the input
    if type(wavelength) is np.ndarray:
        wavelength = np.squeeze(wavelength)
    else:
        wavelength = np.array([wavelength])
    if type(width) is np.ndarray:
        width = np.squeeze(width)
    else:
        width = np.array([width])
    if type(thickness) is np.ndarray:
        thickness = np.squeeze(thickness)
    else:
        thickness = np.array([thickness])
    if type(angle) is np.ndarray:
        angle = np.squeeze(angle)
    else:
        angle = np.array([angle])

    # Run through regression
    INPUT  = cartesian_product([wavelength,width,thickness,angle])

    OUTPUT = LR_straight.predict(INPUT)
    
    return OUTPUT

if __name__ == "__main__":
    print(straightWaveguide(1.550, 0.45, 0.22, 90))