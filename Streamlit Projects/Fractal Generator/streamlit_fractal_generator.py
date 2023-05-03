import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

#This script allows the user to choose between two different fractals -
# the Mandelbrot set or the Julia set - using a dropdown menu. It then allows the user to adjust the
# number of iterations and the value of the complex parameter c (for the Julia set).
# The script generates the fractal using the specified parameters and displays it on a Streamlit page using st.pyplot().

def mandelbrot(c, maxiter):
    z = c
    n = 0
    while abs(z) <= 2 and n < maxiter:
        z = z**2 + c
        n += 1
    if n == maxiter:
        return 0
    else:
        return np.abs(z)

def julia(c, maxiter):
    z = complex(0.0, 0.0)
    n = 0
    while np.abs(z) <= 2.0 and n < maxiter:
        z = z*z + c
        n += 1
    return n

st.title("Fractal Generator")
st.set_option('deprecation.showPyplotGlobalUse', False)
method = st.sidebar.selectbox("Select a fractal:", ["Mandelbrot", "Julia"])

if method == "Mandelbrot":
    maxiter = st.sidebar.slider("Max iterations:", 10, 500, 50)
    xs = np.linspace(-2, 2, 1000)
    ys = np.linspace(-2, 2, 1000)
    xv, yv = np.meshgrid(xs, ys)
    cv = xv + 1j * yv
    fractal = np.zeros_like(cv)
    for i in range(len(xs)):
        for j in range(len(ys)):
            fractal[i,j] = mandelbrot(cv[i,j], maxiter)
    plt.imshow(np.real(fractal.T), extent=[-2, 2, -2, 2], cmap="magma")
    plt.axis("off")
    st.pyplot()

elif method == "Julia":
    maxiter = st.sidebar.slider("Max iterations:", 10, 500, 50)
    real = st.sidebar.slider("Real component of c:", -2.0, 2.0, 0.0)
    imag = st.sidebar.slider("Imaginary component of c:", -2.0, 2.0, 0.0)
    c = complex(real, imag)
    xs = np.linspace(-2, 2, 1000)
    ys = np.linspace(-2, 2, 1000)
    xv, yv = np.meshgrid(xs, ys)
    cv = xv + 1j * yv
    fractal = np.zeros_like(cv, dtype=int)
    for i in range(len(xs)):
        for j in range(len(ys)):
            fractal[i,j] = julia(cv[i,j], maxiter)
    plt.imshow(np.real(fractal.T), extent=[-2, 2, -2, 2], cmap="magma")
    plt.axis("off")
    st.pyplot()
