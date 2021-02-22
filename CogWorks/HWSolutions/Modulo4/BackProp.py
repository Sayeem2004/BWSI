# Necessary Imports
import numpy as np;
import matplotlib.pyplot as plt;
from bwsi_grader.cogworks.autograd import grade_op;
from bwsi_grader.cogworks.autograd import grade_arithmetic;
from bwsi_grader.cogworks.autograd import grade_backprop;
from bwsi_grader.cogworks.autograd import grade_op_backprop;



# Parent Class to Add, Multiple, Subtract, Divide, and Power
class Operation(object):
    def partial_a(self): # Computes the partial derivative of this operation with respect to a: d(op)/da
        return NotImplementedError;
    def partial_b(self): # Computes the partial derivative of this operation with respect to b: d(op)/db
        return NotImplementedError;
    def __call__(self, a, b): # Computes the forward pass of this operation: op(a, b) -> output
        return NotImplementedError;
    def backprop(self, grad):
        self.a.backprop(self.partial_a() * grad) # Backprop: d(op)/da * dF/d(op) -> dF/da
        self.b.backprop(self.partial_b() * grad) # Backprop: d(op)/db * dF/d(op) -> dF/db
    def null_gradients(self):
        for attr in self.__dict__:
            var = getattr(self, attr);
            if hasattr(var, 'null_gradients'):
                var.null_gradients();



# Subclasses of Operation
class Add(Operation):
    def __repr__(self): return "+";
    def __call__(self, a, b): # Adds two Number instances
        self.a = a; self.b = b;
        return a.data + b.data;
    def partial_a(self): # Returns d(a + b)/da
        return 1;
    def partial_b(self): # Returns d(a + b)/db
        return 1;
class Multiply(Operation):
    def __repr__(self): return "*";
    def __call__(self, a, b): # Multiplies two Number instances
        self.a = a; self.b = b;
        return a.data * b.data;
    def partial_a(self): # Returns d(a * b)/da
        return self.b.data;
    def partial_b(self): # Returns d(a * b)/db
        return self.a.data;
class Subtract(Operation):
    def __repr__(self): return "-";
    def __call__(self, a, b): # Subtracts two Number instances
        self.a = a; self.b = b;
        return a.data - b.data;
    def partial_a(self): # Returns d(a - b)/da
        return 1;
    def partial_b(self): # Returns d(a - b)/db
        return -1;
class Divide(Operation):
    def __repr__(self): return "/";
    def __call__(self, a, b): # Divides two Number instances
        self.a = a; self.b = b;
        return a.data / b.data;
    def partial_a(self): # Returns d(a / b)/da
        return 1 / self.b.data;
    def partial_b(self): # Returns d(a / b)/db
        return -self.a.data / (self.b.data ** 2);
class Power(Operation):
    def __repr__(self): return "**";
    def __call__(self, a, b): # Exponentiates two Number instances.
        self.a = a; self.b = b;
        return a.data ** b.data;
    def partial_a(self): # Returns d(a ** b)/da
        return self.b.data * (self.a.data ** (self.b.data-1));
    def partial_b(self): # Returns d(a ** b)/db
        return (self.a.data ** self.b.data) * np.log(self.a.data);



# Class to store values and to create links to run back propogation through.
class Number(object):
    def __repr__(self): return "Number({})".format(self.data)
    def __init__(self, obj, *, creator=None):
        assert isinstance(obj, (Number, int, float, np.generic));
        self.data = obj.data if isinstance(obj, Number) else obj;
        self._creator = creator;
        self.grad = None;
    @property
    def creator(self): # Number.creator is a read-only property
        return self._creator;
    @staticmethod
    def _op(Op, a, b):
        if (not isinstance(a,Number)): a = Number(a);
        if (not isinstance(b,Number)): b = Number(b);
        f = Op(); ret = f(a,b);
        ret = Number(ret,creator=f);
        return ret;
    def __add__(self, other): return self._op(Add, self, other);
    def __radd__(self, other): return self._op(Add, other, self);
    def __mul__(self, other): return self._op(Multiply, self, other);
    def __rmul__(self, other): return self._op(Multiply, other, self);
    def __truediv__(self, other): return self._op(Divide, self, other);
    def __rtruediv__(self, other): return self._op(Divide, other, self);
    def __sub__(self, other): return self._op(Subtract, self, other);
    def __rsub__(self, other): return self._op(Subtract, other, self);
    def __pow__(self, other): return self._op(Power, self, other);
    def __rpow__(self, other): return self._op(Power, other, self);
    def __neg__(self): return -1 * self;
    def __eq__(self, value):
        if isinstance(value, Number): value = value.data;
        return self.data == value;
    def backprop(self, grad=1):
        if (self.grad is None): self.grad = grad;
        else: self.grad += grad;
        if (self._creator is not None): self._creator.backprop(grad);
    def null_gradients(self):
        self.grad = None
        if self._creator is not None: self._creator.null_gradients()



# Checking the classes above
grade_op(student_Number=Number, student_Add=Add, student_Multiply=Multiply);
grade_arithmetic(student_Number=Number, student_Subtract=Subtract, student_Divide=Divide, student_Power=Power);
grade_backprop(student_Number=Number, student_Add=Add, student_Multiply=Multiply);
grade_op_backprop(student_Number=Number, student_Subtract=Subtract, student_Divide=Divide, student_Power=Power);



# Initializing data
data = {
    'Box Office Sales': [85.1, 106.3, 50.2, 130.6, 54.8, 30.3, 79.4, 91.0, 135.4, 89.3],
    'Production Costs': [8.5, 12.9, 5.2, 10.7, 3.1, 3.5, 9.2, 9.0, 15.1, 10.2],
    'Promotion Costs': [5.1, 5.8, 2.1, 8.4, 2.9, 1.2, 3.7, 7.6, 7.7, 4.5],
    'Book Sales': [4.7, 8.8, 15.1, 12.2, 10.6, 3.5, 9.7, 5.9, 20.8, 7.9]
};
# Transforming into a (10,4) list
data_set = list(zip(*[data[key] for key in data]));
# Random parameters initialized as Number class to back propogation works.
def create_model(num_params):
    return tuple(Number(np.random.rand()) for _ in range(num_params));
# Performs one iteration of supervised learning using back propogation to get the change in paramater values
def train_epoch(model, data_set, loss_fn, lr=0.001):
    # Compute the mean error over the dataset
    mean_loss = sum(loss_fn(sample, model) for sample in data_set) / len(data_set);
    # Compute gradients for our parameters
    mean_loss.null_gradients();
    mean_loss.backprop();
    for param in model:
        # Param_new = param_old - step-size * d(L)/d(param)
        param.data -= lr*param.grad;
    return mean_loss.data;


# The L2 loss (squared error) of for a single data point
def l2_loss(truth, model):
    l = truth[0] - model[0] - sum(truth[i]*model[i] for i in range(1, len(model)));
    return l**2 if l.data > 0 else (-1*l)**2;
# Creating and testing out a model (Can't use numpy because Number object)
model = create_model(4);
losses = [];
for _ in range(1000):
    losses.append(train_epoch(model, data_set, l2_loss));
# Visualizing the change in the loss
fig, ax = plt.subplots(figsize=(8,4));
ax.plot(losses);
ax.set_yscale('log');
ax.set_ylabel('Loss');
ax.set_xlabel('Iterations');
ax.grid(True);
# Visualizing importance of each parameter
fig, ax = plt.subplots(figsize=(8,4));
ax.barh(np.arange(4), [model[i].data for i in range(len(model))]);
ax.set_yticks([0, 1, 2, 3]);
ax.set_yticklabels(['Bias', 'Production', 'Promotion', 'Book']);
ax.set_xlabel('Learned Weights');
print("L2 Loss Function");
print("Initial Loss:",losses[0],"Final Loss:",losses[-1]);
print("Parameters:",[Number(round(mod.data,3)) for mod in model]);
print("\n");



# However since we are using back propogation we can calculate derivatives of more loss functions, such as absolute error.
def l1_loss(truth, model):
    l = truth[0] - model[0] - sum(truth[i]*model[i] for i in range(1, len(model)));
    return l if l.data > 0 else (-1*l);
# Creating and Iterating through model with new loss function
model = create_model(4);
losses = [];
for _ in range(1000):
    the_loss = train_epoch(model, data_set, l1_loss);
    losses.append(the_loss);
# Visualizing the change in the loss of the new loss function
fig, ax = plt.subplots(figsize=(8,4));
ax.plot(losses);
ax.set_yscale('log');
ax.set_ylabel('Loss');
ax.set_xlabel('Training Step');
ax.grid(True);
# Visualizing importance of each parameter
fig, ax = plt.subplots(figsize=(8,4));
ax.barh(np.arange(4), [model[i].data for i in range(len(model))]);
ax.set_yticks([0, 1, 2, 3]);
ax.set_yticklabels(['Bias', 'Production', 'Promotion', 'Book']);
ax.set_xlabel('Learned Weights');
print("L1 Loss Function");
print("Initial Loss:",losses[0],"Final Loss:",losses[-1]);
print("Parameters:",[Number(round(mod.data,3)) for mod in model]);
print("\n");


# We can also add an arbitrary amount of data
data = {
    'Box Office Sales': [85.1, 106.3, 50.2, 130.6, 54.8, 30.3, 79.4, 91.0, 135.4, 89.3],
    'Production Costs': [8.5, 12.9, 5.2, 10.7, 3.1, 3.5, 9.2, 9.0, 15.1, 10.2],
    'Promotion Costs': [5.1, 5.8, 2.1, 8.4, 2.9, 1.2, 3.7, 7.6, 7.7, 4.5],
    'Book Sales': [4.7, 8.8, 15.1, 12.2, 10.6, 3.5, 9.7, 5.9, 20.8, 7.9],
    'Random1': [np.random.rand() for _ in range(10)],
    'Random2': [np.random.rand() for _ in range(10)],
    'Random3': [np.random.rand() for _ in range(10)]
};
# Creating and iterating through the model
data_set = list(zip(*[data[key] for key in data]));
model = create_model(7);
losses = [];
for _ in range(1000):
    the_loss = train_epoch(model, data_set, l2_loss);
    losses.append(the_loss);
# Visualizing change in the loss with arbitrary data points
fig, ax = plt.subplots(figsize=(8,4));
ax.plot(losses);
ax.set_yscale('log');
ax.set_ylabel('Loss');
ax.set_xlabel('Training Step');
ax.grid(True);
# Visualizing importance of each parameter
fig, ax = plt.subplots(figsize=(8,4));
ax.barh(np.arange(7), [model[i].data for i in range(len(model))]);
ax.set_yticks([0, 1, 2, 3, 4, 5, 6]);
ax.set_yticklabels(['Bias','Production', 'Promotion', 'Book', 'Random1', 'Random2', 'Random3']);
ax.set_xlabel('Learned Weights');
print("Random Data");
print("Initial Loss:",losses[0],"Final Loss:",losses[-1]);
print("Parameters:",[Number(round(mod.data,3)) for mod in model]);
print("\n");
# plt.show();
