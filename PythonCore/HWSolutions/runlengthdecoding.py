from bwsi_grader.python.run_length_encoding import decoder_grader;

def run_length_decoder(in_list):
    out_string = "";
    i = 0;
    while (i < len(in_list)):
        if (i == 0):
            out_string += in_list[i];
            i += 1; continue;
        else:
            if (in_list[i] == in_list[i-1]):
                out_string += in_list[i] * (in_list[i+1]-1);
                i += 2; continue;
            else:
                out_string += in_list[i];
                i += 1; continue;
    return out_string;

decoder_grader(run_length_decoder)
