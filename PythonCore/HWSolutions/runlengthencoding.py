from bwsi_grader.python.run_length_encoding import encoder_grader;

def run_length_encoder(in_string):
    i = 0;
    out_list = [];
    while (i < len(in_string)):
        if (i == 0):
            out_list.append(in_string[i]);
            i += 1; continue;
        else:
            if (in_string[i] != in_string[i-1]):
                out_list.append(in_string[i]);
                i += 1; continue;
            else:
                q = 1;
                while (i < len(in_string) and in_string[i] == in_string[i-1]):
                    q += 1; i += 1;
                out_list.append(in_string[i-1]);
                out_list.append(q);
                continue;
    return out_list;

encoder_grader(run_length_encoder);
