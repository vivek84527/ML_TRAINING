import pandas as pd
import numpy as np

dates = pd.date_range('20190101', periods=6, freq="D")
# "M"=Month
# "Y"=Year
# "D"=Day
print(dates)  # It contains 6 dates as array

# dates =  DatetimeIndex(['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04',
#               '2019-01-05', '2019-01-06'],
#              dtype='datetime64[ns]', freq='D')

print(dates[0])
print("\n\n", np.random.randn(6, 4))

df = pd.DataFrame(np.random.randn(6, 4),
                  index=dates,
                  columns=['A', 'B', 'C', 'D']
                  )

print("\n\n", df)

df['E'] = df['A'].apply(lambda x: 1 if (x > 0) else 0)

print(df)

print(df.groupby('E').size())

print("Headings in Dataframe : ", df.columns)

print("Row Headings in Dataframe : ", df.index)

print("Values in Dataframe : \n", df.values)

print(df.dtypes)

print(df.dtypes)
print(df.head())  # BY default 5 value
print(df.tail(3))  # last 3 row

print(df.sample(3))  # BY default 1

print(df.describe())

print(df.describe(include="all"))

print(df.T)

print(df.sort_values(by='B', ascending=True))

print("original values : \n ", df)

print(df.sort_values(by='B', ascending=True))

print("original values : \n ", df)

# del df["E"]
# print( df )


# Selecting a single column, which yields a Series, equivalent to df.A
print(df.A)
print()
print(df['A'])

print(df['2019-01-01':'2019-01-03'])
print(df[0:3])  # print( First three rows)

print(df.loc[dates[0]])

print(df.loc[:, ['A', 'B']])

print(df.loc['20190102':'20190104', ['A', 'C']])

print(df.loc['20190102', ['A', 'B']])

# For getting fast access to a scalar
#  (equiv to the prior method)
print(df.at[dates[0], 'A'])

# For getting fast access to a scalar
#  (equiv to the prior method)
print(df.at[dates[0], 'A'])
# Select via the position of the passed integers

print(df.iloc[3])

print(df.iloc[3:5, 0:2])

print(df.iloc[[1, 2, 4], [0, 2]])
print(df.iloc[1:3, :])

# _________________________________________

print("\n\n\n")

# Boolean Indexing
print(df.A)

print("\n\n")
print(df.A > 0)

print("\n\n")
print(df[df.A > 0])

#                   A         B         C         D
#   2019-01-04  0.194764 -0.877930 -2.115444 -1.564107
#   2019-01-05  0.960763 -0.913735  0.077043 -2.103443

print(df["B"][df.A > 0])

print("\n\n")
# Selecting values from a DataFrame where a boolean condition is met.
print(df > 0)

print("\n\n")
print(df[df > 0])

#                   A         B         C        D
#   2019-01-01       NaN       NaN  1.088617      NaN
#   2019-01-02  0.248885  1.077837  0.459089      NaN
#   2019-01-03  1.281457  0.434085       NaN      NaN
#   2019-01-04       NaN  0.455839  1.189213      NaN
#   2019-01-05       NaN       NaN       NaN      NaN
#   2019-01-06       NaN  0.016767  2.130660  0.98654
