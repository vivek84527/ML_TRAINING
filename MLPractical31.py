import pandas

filename = 'indians-diabetes.data.csv'

hnames = ['preg', 'plas', 'pres',
          'skin', 'test','mass',
          'pedi',  'age', 'class']

dataframe = pandas.read_csv(filename,  names=hnames)

print( "pandas Data : " , dataframe.shape  )

print( dataframe  )

print( "\n\n"  )

print( type(dataframe)  )


a = 5
print( type(a) )

a = 12.5
print( type(a) )

a = "iitk"
print( type(a) )

a = [1,2,3,4]
print( type(a) )


a = (1,2,3,4)
print( type(a) )

print( dataframe.dtypes )
