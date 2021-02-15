from bwsi_grader.python.three_five import grader;

def student_func(x):
    if x%15 == 0: return "threefive";
    elif x%5 == 0: return "five";
    elif x%3 == 0: return "three";
    else: return x;

grader(student_func);
