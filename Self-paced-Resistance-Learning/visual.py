import scipy.io
import matplotlib.pyplot as plt

mat_data = scipy.io.loadmat('/home/wangyl/Code/PSCL/Self-paced-Resistance-Learning/tire_symmetric_ResNet18_200epoch40_test_accuracy_0.2.mat')

variables = mat_data.keys()
print(variables)

var = mat_data['test_accuracy']
print(var)
# plt.plot(var)
# plt.savefig('accuracy_curve.png')
# plt.show()