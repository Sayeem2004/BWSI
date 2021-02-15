from bwsi_grader.python.relaxation_method import grader2;

def relaxation_method2(func, xo, tol, max_it):
    lt = [];
    lt.append(xo);
    for i in range(max_it-1):
        if (len(lt) < 3): lt.append(func(lt[-1]));
        else:
            err = error(lt[-1],lt[-2],lt[-3]);
            if (err < tol): break;
            else: lt.append(func(lt[-1]));
    return lt;

def error(x,y,z):
    top = (x-y)*(x-y);
    bottom = 2*y - z - x;
    if (bottom == 0): return abs(top/1e-14);
    else: return abs(top/bottom);

grader2(relaxation_method2);
