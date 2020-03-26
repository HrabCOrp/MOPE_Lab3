import random, math, numpy
from scipy.stats import t, f

# Вигляд рівняння регресії
print("\nВигляд рівняння регресії: y = b0 + b1*x1 + b2*x2 + b3*x3\n")

N = 4
m = 3

min_x1 = -20
max_x1 = 15
min_x2 = 10
max_x2 = 60
min_x3 = 15
max_x3 = 35

mean_Xmax = (max_x1 + max_x2 + max_x3) / 3
mean_Xmin = (min_x1 + min_x2 + min_x3) / 3

max_y = 200 + mean_Xmax
min_y = 200 + mean_Xmin

# нормовані значення факторів (з фіктивним нульовим фактором x0=1)
x_coded = numpy.array([[1, -1, -1, -1],
                       [1, -1, 1, 1],
                       [1, 1, -1, 1],
                       [1, 1, 1, -1]])

while True:
    # значення функції відгуку
    y_matrix = numpy.random.uniform(low=min_y, high=max_y, size=(N, m))

    # натуралізовані значення факторів
    x_matrix = numpy.array([[min_x1, min_x2, min_x3],
                            [min_x1, max_x2, max_x3],
                            [max_x1, min_x2, max_x3],
                            [max_x1, max_x2, min_x3]])

    sumY_matrix = numpy.zeros(N)
    meanY_matrix = numpy.zeros(N)

    for i in range(N):
        for j in range(m):
            sumY_matrix[i] += y_matrix[i][j]
        # Середні значення функції відгуку за рядками
        meanY_matrix[i] = sumY_matrix[i] / m

    # Критерій Кохрена (перша статистична перевірка)

    # Дисперсія
    sigma = numpy.zeros(N)
    for i in range(N):
        for j in range(m):
            sigma[i] += pow((y_matrix[i][j] - meanY_matrix[i]), 2)
        sigma[i] = sigma[i] / m

    f1 = m - 1
    f2 = N
    q = 0.05

    q1 = q / f1
    fisher_value = f.ppf(q=1 - q1, dfn=f2, dfd=(f1 - 1) * f2)
    Gt = fisher_value / (fisher_value + f1 - 1)
    print
    Gp = max(sigma) / sum(sigma)
    print("Критерій Кохрена")
    if (Gp < Gt):
        print("{} < {} => Дисперсія однорідна.\n".format(Gp.__round__(3), Gt.__round__(3)))
        break
    else:
        print("{} > {}Дисперсія неоднорідна. Збільшуємо m.\n".format(Gp.__round__(3), Gt.__round__(3)))
        m = m + 1

mx1 = numpy.sum(x_matrix, axis=0)[0] / N
mx2 = numpy.sum(x_matrix, axis=0)[1] / N
mx3 = numpy.sum(x_matrix, axis=0)[2] / N

my = numpy.sum(meanY_matrix) / N

a1 = 0
a2 = 0
a3 = 0
a11 = 0
a22 = 0
a33 = 0
a12 = 0
a13 = 0
a23 = 0
for i in range(len(x_matrix)):
    a1 += x_matrix[i][0] * meanY_matrix[i] / len(x_matrix)
    a2 += x_matrix[i][1] * meanY_matrix[i] / len(x_matrix)
    a3 += x_matrix[i][2] * meanY_matrix[i] / len(x_matrix)

    a11 += x_matrix[i][0] ** 2 / len(x_matrix)
    a22 += x_matrix[i][1] ** 2 / len(x_matrix)
    a33 += x_matrix[i][2] ** 2 / len(x_matrix)

    a12 += x_matrix[i][0] * x_matrix[i][1] / len(x_matrix)
    a13 += x_matrix[i][0] * x_matrix[i][2] / len(x_matrix)
    a23 += x_matrix[i][1] * x_matrix[i][2] / len(x_matrix)

a21 = a12
a31 = a13
a32 = a23

determinant = numpy.linalg.det([[1, mx1, mx2, mx3],
                                [mx1, a11, a12, a13],
                                [mx2, a12, a22, a32],
                                [mx3, a13, a23, a33]])

determinant0 = numpy.linalg.det([[my, mx1, mx2, mx3],
                                 [a1, a11, a12, a13],
                                 [a2, a12, a22, a32],
                                 [a3, a13, a23, a33]])

determinant1 = numpy.linalg.det([[1, my, mx2, mx3],
                                 [mx1, a1, a12, a13],
                                 [mx2, a2, a22, a32],
                                 [mx3, a3, a23, a33]])

determinant2 = numpy.linalg.det([[1, mx1, my, mx3],
                                 [mx1, a11, a1, a13],
                                 [mx2, a12, a2, a32],
                                 [mx3, a13, a3, a33]])

determinant3 = numpy.linalg.det([[1, mx1, mx2, my],
                                 [mx1, a11, a12, a1],
                                 [mx2, a12, a22, a2],
                                 [mx3, a13, a23, a3]])

# коефіцієнти рівняння регресії
b0 = determinant0 / determinant
b1 = determinant1 / determinant
b2 = determinant2 / determinant
b3 = determinant3 / determinant

print("Нормовані значення факторів:\n", x_coded)
print("Натуралізовані значення факторів:\n", x_matrix)
print("Значення функції відгуку\n", y_matrix)
print("\nРівняння регресії: y = {} + {}*x1 + {}*x2 + {}*x3\n".format(b0.__round__(3), b1.__round__(3), b2.__round__(3),
                                                                   b3.__round__(3)))
# Перевірка
y1 = b0 + b1 * x_matrix[0][0] + b2 * x_matrix[0][1] + b3 * x_matrix[0][2]
y2 = b0 + b1 * x_matrix[1][0] + b2 * x_matrix[1][1] + b3 * x_matrix[1][2]
y3 = b0 + b1 * x_matrix[2][0] + b2 * x_matrix[2][1] + b3 * x_matrix[2][2]
y4 = b0 + b1 * x_matrix[3][0] + b2 * x_matrix[3][1] + b3 * x_matrix[3][2]

print("Зробимо перевірку по рядках: \n{}\n{}\n{}\n{}\n".format(y1, y2, y3, y4))
print("Порівняємо із середніми значеннями:\n", meanY_matrix)

# Критерій Стьюдента (Друга статистична перевірка)

f3 = f1 * f2

# Оцінка генеральної дисперсії відтворюваності
Sb = sum(sigma) / N
Sbs_2 = Sb / (N * m)
Sbs = math.sqrt(Sbs_2)

# Оцінки коефіцієнтів
beta0 = (meanY_matrix[0] + meanY_matrix[1] + meanY_matrix[2] + meanY_matrix[3]) / N
beta1 = (-meanY_matrix[0] - meanY_matrix[1] + meanY_matrix[2] + meanY_matrix[3]) / N
beta2 = (-meanY_matrix[0] + meanY_matrix[1] - meanY_matrix[2] + meanY_matrix[3]) / N
beta3 = (-meanY_matrix[0] + meanY_matrix[1] + meanY_matrix[2] - meanY_matrix[3]) / N

T0 = abs(beta0) / Sbs
T1 = abs(beta1) / Sbs
T2 = abs(beta2) / Sbs
T3 = abs(beta3) / Sbs

# Перевірка значущості коефіцієнтів рівняння регресії (b0, b1, b2, b3)
# Якщо відповідний T > Tt => коеф. значущий, інакше => незначущий (рівний 0)
Tt = t.ppf((1 + (1 - q)) / 2, f3)
b_0 = b0 if T0 > Tt else 0
b_1 = b1 if T1 > Tt else 0
b_2 = b2 if T2 > Tt else 0
b_3 = b3 if T3 > Tt else 0
beta_matrix = numpy.array([b_0, b_1, b_2, b_3])

# Рівняння регресії після другої статистичної перевірки (незначущі коефіцієнти рівні 0)
print("\nКритерій Стьюдента. Отримане рівняння регресії: y = {} + {}*x1 + {}*x2 + {}*x3\n".format(b_0.__round__(3),
                                                                                                b_1.__round__(3),
                                                                                                b_2.__round__(3),
                                                                                                b_3.__round__(3)))
y_1 = b_0 + b_1 * x_matrix[0][0] + b_2 * x_matrix[0][1] + b_3 * x_matrix[0][2]
y_2 = b_0 + b_1 * x_matrix[1][0] + b_2 * x_matrix[1][1] + b_3 * x_matrix[1][2]
y_3 = b_0 + b_1 * x_matrix[2][0] + b_2 * x_matrix[2][1] + b_3 * x_matrix[2][2]
y_4 = b_0 + b_1 * x_matrix[3][0] + b_2 * x_matrix[3][1] + b_3 * x_matrix[3][2]
y_list = [y_1, y_2, y_3, y_4]

# Критерій Фішера (Третя статистична перевірка)

print("Критерій Фішера")

d = len(beta_matrix[numpy.array(beta_matrix) != 0])
f4 = N - d

# Дисперсія адекватності
Sad = m / (N - d) * sum([(y_list[i] - meanY_matrix[i]) ** 2 for i in range(len(meanY_matrix))])
Fp = Sad / Sbs_2
Ft = f.ppf(1 - q, f4, f3)
if Fp > Ft:
    print("Рівняння регресії неадекватне оригіналу при q = {}\n{} > {}".format(q.__round__(3), Fp.__round__(3),
                                                                               Ft.__round__(3)))
else:
    print("Рівняння регресії адекватне оригіналу при q = {}\n{} < {}".format(q.__round__(3), Fp.__round__(3),
                                                                             Ft.__round__(3)))
