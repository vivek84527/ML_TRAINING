#   1) Univariate Histogram Plot.
#   2) Univariate Density Plots.
#   3) Univariate Box and Whisker Plots.



#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import pandas

filename = 'indians-diabetes.data.csv'

hnames = ['preg', 'plas', 'pres', 'skin',
        'test', 'mass', 'pedi', 'age', 'class']

df = pandas.read_csv(filename, names=hnames)

print( df )

df.hist()

plt.show()
