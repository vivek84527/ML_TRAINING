import numpy as np

print(np.pi)

print(np.linspace(-5, 5, 11))

print(np.linspace(-np.pi, np.pi, 5))

print("--Transcendental functions:---")
a = np.arange(1, 6)  # [0, 1, 2, 3, 4, 5]

print(np.sin(a))  # array([ 0., 0.84147098,  0.90929743,  0.14112001, -0.7568025 ])

print(np.log(a))  # array([ -inf,  0. ,  0.69314718,  1.09861229,  1.38629436])
print(np.exp(a))  # array([  1. ,   2.71828183,   7.3890561 ,  20.08553692,  54.59815003])
