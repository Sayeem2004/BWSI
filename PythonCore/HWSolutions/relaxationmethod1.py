from bwsi_grader.python.relaxation_method import grader1;

def relaxation_method1(func, xo, num_it):
    lt = [];
    lt.append(xo);
    for i in range(num_it): lt.append(func(lt[-1]));
    return lt;

grader1(relaxation_method1);
