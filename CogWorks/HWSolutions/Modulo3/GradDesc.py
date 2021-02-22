# Necessary Imports
import numpy as np;
import matplotlib.pyplot as plt;
from bwsi_grader.cogworks.gradient_learning import grade_polygrad;
from bwsi_grader.cogworks.gradient_learning import grade_gradient_descent;
from bwsi_grader.cogworks.gradient_learning import grade_loss_and_gradient;



# Function to get the derivative of a function at a value x
def poly_grad(coefs, x):
    # Base cases
    if (len(coefs) == 0 or len(coefs) == 1): return float(0);
    if (len(coefs) == 2): return float(coefs[1]);
    # Updating coefficients
    co = np.array(coefs);
    N = np.size(co);
    mult = np.linspace(0,N-1,N);
    co = co * mult;
    # Finding powers of x
    var = np.zeros(N);
    var = var + x;
    pw = np.concatenate((np.array([0]),np.linspace(0,N-2,N-1)));
    var = var ** pw;
    # Multiplying and summing coefficents and powers of x
    return float(np.sum(var * co));
grade_polygrad(poly_grad);



# Function that returns the values of x on a gradient descent
def grad_descent(poly, step_size=0.1, iterations=10, x=100.):
    # Initialzing list
    x_list = [];
    x_list.append(x);
    # Updating x and adding to list
    for _ in range(iterations):
        df = poly_grad(poly,x);
        x -= step_size * df;
        x_list.append(x);
    # Returning final list
    return x_list;
grade_gradient_descent(grad_descent);



# Testing out different step sizes for the same functions
x_list = grad_descent([15, 0, 2], 0.1, 10, 100);
x_list2 = grad_descent([15, 0, 2], 0.4, 10, 100);
x_list3 = grad_descent([15, 0, 2], 0.6, 10, 3);
x_list = [round(x,3) for x in x_list];
x_list2 = [round(x,3) for x in x_list2];
x_list3 = [round(x,3) for x in x_list3];
print("L(x) = 15x^2 + 2 (Convex)");
print("0.1:",x_list,"\nDist from Min:",abs(x_list[-1]));
print("0.4:",x_list2,"\nDist from Min:",abs(x_list2[-1]));
print("0.6:",x_list3,"\nDist from Min:",abs(x_list3[-1]));
# Testing out a non convex function with multiple relative mininums
def f(x): return 1 + x - (x ** 2) - (x ** 3) + (x ** 4);
x_list4 = grad_descent([1, 1, -1, -1, 1], x=2, step_size=0.03);
x_list4 = [round(x,3) for x in x_list4];
print("\nL(x) = x^4 - x^3 - x^2 + x + 1 (Non-Convex)");
print("0.03:",x_list4,);
print("Mininums at around (1,1) and (-0.7,0.4) but the gradient doesnt realize that \nthere is a better mininum so it stays around 1.");



# Movie sales prediction using gradient descent
# Model will be f(X, A_bias, A_prod, A_prom, A_book) = A_bias + (X_prod * A_prod) + (X_prom * A_prom) + (X_book * A_book)
# Can also be written as X * A = [1,X_prod,X_prom,X_book] * [A_bias,A_prod,A_prom,A_book]
print("\nModel: X * A");
# Loss function will be average squared error
print("Loss Function: 1/N * \sum{(y - X * A)^2}");
# Derivative of Loss = Gradient: [ 1/N * \sum{2*(y-X*A)*-1}, 1/N * \sum{2*(y-X*A)*-X_prod}, 1/N * \sum{2*(y-X*A)*-X_prom}, 1/N * \sum{2*(y-X*A)*-X_book} ]
# [1,X_prod,X_prod,X_prom,X_book] = X -> 1/N * \sum{-2(y-X*A)X} -> -2/N * \sum{(y-X*A)X}
print("Derivative of Loss Function: -2/N * \sum{(y-X*A)X}")



# Function to calculate both loss and gradient change for this specific movie example
def loss_and_gradient(X, A, y):
    N = np.size(y);
    # Loss Function: 1/N * \sum_1^N{(y-A*X)^2}
    model = np.dot(X,A);
    loss = np.sum((y - model) ** 2) / N;
    # Gradient Descent -2/N * \sum_1^N{(y-A*X)X}
    gradient = np.sum(np.resize((y - model), (N,1)) * X, axis=0) * -2/N;
    return (loss,gradient);
grade_loss_and_gradient(loss_and_gradient);



# Initializing Data and Parameters
data = {
    'Box Office Sales': [85.1, 106.3, 50.2, 130.6, 54.8, 30.3, 79.4, 91.0, 135.4, 89.3],
    'Production Costs': [8.5, 12.9, 5.2, 10.7, 3.1, 3.5, 9.2, 9.0, 15.1, 10.2],
    'Promotion Costs': [5.1, 5.8, 2.1, 8.4, 2.9, 1.2, 3.7, 7.6, 7.7, 4.5],
    'Book Sales': [4.7, 8.8, 15.1, 12.2, 10.6, 3.5, 9.7, 5.9, 20.8, 7.9]
};
N = len(data['Box Office Sales']);
y = np.array(data['Box Office Sales']);
X = np.vstack([np.ones(N), data['Production Costs'], data['Promotion Costs'], data['Book Sales']]).T;
# The four model parameters, random numbers distributed near 0.
A = np.random.normal(size=(4,));
Ai = np.copy(A);


# Iterating through and updating parameters and recording losses
losses = [];
step_size = 0.001;
iterations = 400;
losses = list();
for _ in range(iterations):
    loss, grad = loss_and_gradient(X, A, y);
    A -= step_size * grad;
    losses.append(loss);
print("Initial Weights:",Ai)
print("Final Weights:",A);
print("Initial Loss:",losses[0]);
print("Final Loss",losses[-1]);



# Visualization of change in loss
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(losses)
ax.set_yscale('log')
ax.set_ylabel("Loss")
ax.set_xlabel("Iterations");
ax.grid(True)
# Visualization of weights for each X
fig, ax = plt.subplots(figsize=(8,4))
ax.barh(np.arange(4), A[::-1])
ax.barh(np.arange(4), Ai[::-1])
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(['Bias', 'Production', 'Promotion', 'Book'][::-1])
ax.set_xlabel("Initial Weights vs Final Weights");
plt.show();
