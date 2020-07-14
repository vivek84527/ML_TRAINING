import numpy as np

print("-------------------------------")
ddarr = np.array([[1, 2, 3], [4, 5, 6]])

print("ddarr.ndim = ", ddarr.ndim)  # ddarr.ndim =  2

print("ddarr.shape = ", ddarr.shape)
# ddarr.shape =  (2L, 3L)

print("ddarr.size = ", ddarr.size)  # ddarr.size =  6

print("len(ddarr) = ", len(ddarr))

print("ddarr.dtype = ", ddarr.dtype)
# ddarr.dtype =  int64

print(ddarr)  # [ [1 2 3]
#     [4 5 6] ]

print("***************************")
print("ddarr[0,1] = ", ddarr[0, 1])  # 2  [ row-id ,coloum-id ]
print("ddarr[0] = ", ddarr[0])  # [1 2 3] #Get entire 0 row

print("ddarr[ : , 0 ]=", ddarr[:, 0])  # [1 4]   #From all rows, get 0 coloum

print("ddarr[1 , : ]=", ddarr[1, :])
