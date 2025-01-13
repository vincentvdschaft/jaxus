# importing libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import *


# creating data
x1 = np.array([0, 1, 2, 3])
y1 = np.array([5, 2, 8, 6])

# creating plot
fig = plt.figure()
ax = fig.subplots()
plt.subplots_adjust(left=0.3, bottom=0.25)
(p,) = ax.plot(x1, y1, color="blue", marker="o")


# defining function to add line plot
def add(val):
    print("Button clicked.")
    x2 = np.array([0, 1, 2, 3])
    y2 = np.array([10, 2, 0, 12])
    ax.plot(x2, y2, color="green", marker="o")
    plt.draw()


# defining button and add its functionality
axes = plt.axes([0.21, 0.000001, 0.3, 0.075])
bnext = RadioButtons(axes, "Add", activecolor="yellow")
bnext.on_clicked(add)
plt.show()
