import numpy as np
import matplotlib.pyplot as plt
import scipy

omega_b = 0.5
omega_c = -1.25
omega_p = 0.75
v_b = 0.1

def transformation_matrices(t, omega_b,omega_c,omega_p,v_b):
    T_ab = np.array([
        [np.cos(omega_b * t), -np.sin(omega_b * t), 0, 0],
        [np.sin(omega_b * t), np.cos(omega_b * t), 0, 1 + v_b * t],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    T_bc = np.array([
        [np.cos(omega_c * t), -np.sin(omega_c * t), 0, 4],
        [np.sin(omega_c * t), np.cos(omega_c * t), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    T_cd = np.array([
        [1, 0, 0, 1.5],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    T_dp = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, np.cos(omega_p*t)],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    T_ap = np.matmul(np.matmul(np.matmul(T_ab, T_bc), T_cd), T_dp)
    return T_ap
x_translations = []
y_translations = []

for t in np.arange(1, 80):
    T_ap = transformation_matrices(t, omega_b, omega_c, omega_p, v_b)
    x_translations.append(T_ap[0, 3])
    y_translations.append(T_ap[1, 3])

x_translations = np.array(x_translations)
y_translations = np.array(y_translations)

time = np.arange(1, 80)

plt.figure(figsize=(10, 6))
plt.plot(x_translations, y_translations, marker='o', linestyle='-')
plt.title('Movement of the Artistic Arm (Point P)')
plt.xlabel('X Translation')
plt.ylabel('Y Translation')
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
plt.grid(True)
plt.plot(time, x_translations, marker='o', label='X Translations')
plt.plot(time, y_translations, marker='o', label='Y Translations')
plt.title('X and Y Translations Over Time')
plt.xlabel('Time (t)')
plt.ylabel('Translation')
plt.legend()
plt.show()
plt.show()


spline_x = scipy.interpolate.CubicSpline(time, x_translations, bc_type='natural')
spline_y = scipy.interpolate.CubicSpline(time, y_translations, bc_type='natural')

time_smooth = np.linspace(1, 80, 400)
x_spline_smooth = spline_x(time_smooth)
y_spline_smooth = spline_y(time_smooth)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(time, x_translations, 'o', label='Original X Points')
plt.plot(time_smooth, x_spline_smooth, label='Cubic Spline X', color='b')
plt.title('Cubic Spline Interpolation for X Translations')
plt.xlabel('Time (t)')
plt.ylabel('X Translation')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(time, y_translations, 'o', label='Original Y Points')
plt.plot(time_smooth, y_spline_smooth, label='Cubic Spline Y')
plt.title('Cubic Spline Interpolation for Y Translations')
plt.xlabel('Time (t)')
plt.ylabel('Y Translation')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(x_spline_smooth, y_spline_smooth)
plt.title('Movement of the Artistic Arm (Point P)')
plt.xlabel('X Translation')
plt.ylabel('Y Translation')
plt.grid(True)
plt.show()


#Printing out the spline functions for each segment
"""
def print_spline_function(spline, spline_name):
    print(f"Spline Function for {spline_name}:")
    for i in range(len(spline.c[0]) - 1):
        a, b, c, d = spline.c[:, i]
        t_i = spline.x[i]
        print(f"Segment {i+1} [{t_i}, {spline.x[i + 1]}]:")
        print(f"{spline_name}({t}) = ({a:.4f}) * (t - {t_i})^3 + ({b:.4f}) * (t - {t_i})^2 + ({c:.4f}) * (t - {t_i}) + ({d:.4f})\n")

print_spline_function(spline_x, "x")
print_spline_function(spline_y, "y")
"""

spline_x = scipy.interpolate.CubicSpline(time, x_translations, bc_type='natural')
spline_y = scipy.interpolate.CubicSpline(time, y_translations, bc_type='natural')

# Generate more points for a smoother curve
time_smooth = np.linspace(1, 80, 400)


vx = spline_x(time_smooth, 1)  #1st derivate of spline
vy = spline_y(time_smooth, 1)  #1st derivative

# Plotting
plt.figure(figsize=(12, 6))

#Velocity component x
plt.subplot(1, 2, 1)
plt.plot(time_smooth, vx, label='Velocity vx')
plt.title('Velocity Component vx')
plt.xlabel('Time (t)')
plt.ylabel('Velocity vx')
plt.legend()

#Velocity component y
plt.subplot(1, 2, 2)
plt.plot(time_smooth, vy, label='Velocity vy')
plt.title('Velocity Component vy')
plt.xlabel('Time (t)')
plt.ylabel('Velocity vy')
plt.legend()

plt.tight_layout()
plt.show()


omega_p = 0
v_b = 0

x_translations = []
y_translations = []

for t in np.arange(1, 80):
    T_ap = transformation_matrices(t, omega_b, omega_c, omega_p, v_b)
    x_translations.append(T_ap[0, 3])  # X Translation
    y_translations.append(T_ap[1, 3])  # Y Translation

x_translations = np.array(x_translations)
y_translations = np.array(y_translations)

plt.plot(x_translations, y_translations)
plt.title('Pattern of P with vB = 0 and ωP = 0')
plt.xlabel('X Translation')
plt.ylabel('Y Translation')
plt.grid(True)
plt.show()

spline_x = scipy.interpolate.CubicSpline(time, x_translations, bc_type='natural')
spline_y = scipy.interpolate.CubicSpline(time, y_translations, bc_type='natural')

time_smooth = np.linspace(1, 80, 400)
x_spline_smooth = spline_x(time_smooth)
y_spline_smooth = spline_y(time_smooth)

plt.figure(figsize=(10, 6))
plt.plot(x_spline_smooth, y_spline_smooth)
plt.title('Pattern of the arm with wP = 0 and vB = 0 with natural cubic splines')
plt.xlabel('X Translation')
plt.ylabel('Y Translation')
plt.grid(True)
plt.show()


omega_p = 0.75
x_translations = []
y_translations = []

for t in np.arange(1, 80):
    T_ap = transformation_matrices(t, omega_b, omega_c, omega_p, v_b)
    x_translations.append(T_ap[0, 3])
    y_translations.append(T_ap[1, 3])

x_translations = np.array(x_translations)
y_translations = np.array(y_translations)

plt.plot(x_translations, y_translations)
plt.title('Pattern of P with vB = 0')
plt.xlabel('X Translation')
plt.ylabel('Y Translation')
plt.grid(True)
plt.show()
