import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Linear Regression Model
linear_regressor = LinearRegression()
linear_regressor.fit(x, y)

# Polynomial Regression Model
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)

# This linear regression model will be a multiple linear regression model trained with the new matrix
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# Visualizing the Linear Regression Results
# The real results
plt.scatter(x, y, color="red")
# The linear regression line will take the predicted values from the trained linear regression created above
plt.plot(x, linear_regressor.predict(x), color="blue")
plt.title("Linear Regression Model")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualizing the Polynomial Regression Results
plt.scatter(x, y, color="red")
plt.plot(x, lin_reg_2.predict(x_poly), color="blue")
plt.title("Polynomial Regression Model")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Make the curve more smooth
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x,y, color="red")
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color="blue")
plt.title("Polynomial Regression Model Smooth")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
