import matplotlib.pyplot as plt
with open('Lorentz.txt') as fp:
    p = [float(numeric_string) for numeric_string in ((fp.readline()).strip()).split()]
    x = [float(numeric_string) for numeric_string in ((fp.readline()).strip()).split()]
    y = [float(numeric_string) for numeric_string in ((fp.readline()).strip()).split()]
    z = [float(numeric_string) for numeric_string in ((fp.readline()).strip()).split()]
fig = plt.figure()

ax = fig.add_subplot(projection='3d')

ax.plot(x,y,z)

plt.show()