from bwsi_grader.python.ciphers import grade_cesar_cipher;

def encode_caesar(string, shift_amt):
    original = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    new = original[shift_amt:] + original[:shift_amt];
    encode = "";
    for c in string:
        if (c not in original): encode += c;
        else: encode += new[original.index(c)];
    return encode;

grade_cesar_cipher(encode_caesar);
