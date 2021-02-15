from collections import Counter;
from bwsi_grader.python.odds_and_ends import grade_file_parser;

def get_most_popular_foods(file_path):
    count = Counter();
    types = set();
    fin = open(file_path,"r");
    lt = fin.read().split("\n");
    lt = [l.split(", ") for l in lt];
    for l in lt:
        for d in l:
            s = "".join(d.split(": "));
            count[s] += 1;
            types.add(d.split(": ")[1]);
    dt = {};
    for s in types:
        string = "";
        mx = 0;
        for c in count:
            if (s in c and count[c] > mx):
                string = c[:c.index(s)];
                mx = count[c];
            if (s in c and count[c] == mx):
                if (c[:c.index(s)] < string):
                    string = c[:c.index(s)];
        dt[s] = string;
    return dt;

grade_file_parser(get_most_popular_foods);
