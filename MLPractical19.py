import matplotlib.pyplot as plt

# If you make multiple lines with
#  one plot command, the kwargs
# apply to all those lines, e.g.:
x1 = [1, 2, 3]
y1 = [1, 2, 3]

x2 = [1, 2, 3]
y2 = [1, 4, 9]

plt.plot(x1, y1, x2, y2, linewidth=7)
plt.axis([0, 4, 0, 10])
plt.show()
