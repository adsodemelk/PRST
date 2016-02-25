from scipy import io
dict_a = {'a':[3, 9, 17, 15, 19]}
io.savemat('example.mat', dict_a)

mat = io.loadmat('example.mat')
