from bwsi_grader.python.palindrome import grader;

def student_func(x):
    return x.replace(" ","").lower() == x[::-1].replace(" ","").lower();

grader(student_func);
