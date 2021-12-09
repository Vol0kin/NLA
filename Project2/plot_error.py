import numpy as np
import matplotlib.pyplot as plt

k = [1, 5, 10, 25, 100]
deer = [0.2874376427561434, 0.19901972666376574, 0.15674559980240926, 0.10377117208321583, 0.04469922197099497]
new_york = [0.48494116028696377, 0.3986024244749012, 0.3489472704514646, 0.2748494438102985, 0.1489300464821571]
woman = [0.2968443026775804, 0.13539210819522046, 0.08922561780928832, 0.05214604965875523, 0.017056281816051417]
dog = [0.3682365550318945, 0.14937470940209863, 0.1016774430637014, 0.05107826613886749, 0.008193054137424124]

plt.plot(k, deer, 'bo-', label='Deer')
plt.plot(k, new_york, 'ro-', label='New York')
plt.plot(k, woman, 'go-', label='Woman')
plt.plot(k, dog, 'o-', color='orange', label='Dog')

plt.legend()
plt.xlabel('Rank of the approximation')
plt.ylabel('Relative error (Frobenius norm)')
plt.xticks(k)
plt.show()