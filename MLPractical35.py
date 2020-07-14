import pandas
import matplotlib.pyplot as plt
filename = 'indians-diabetes.data.csv'

headingnames = ['preg', 'plas', 'pres', 'skin', 'test',
                'mass', 'pedi', 'age', 'class']

dataframe = pandas.read_csv(filename, names=headingnames)

dataframe.plot(kind='density', subplots=True,
               layout=(3, 3), sharex=False)

plt.show()
