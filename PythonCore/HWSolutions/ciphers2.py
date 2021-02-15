from bwsi_grader.python.ciphers import grade_keyword_cipher;

def encode_keyword(string, keyword):
    key = "";
    alp = "abcdefghijklmnopqrstuvwxyz";
    for i,c in enumerate(keyword):
        if (keyword.index(c) == i and c in alp): key += c;
    for c in alp:
        if c not in key: key += c;
    encode = "";
    for c in string.lower():
        if (c not in alp): encode += c;
        else: encode += key[alp.index(c)];
    return encode;

grade_keyword_cipher(encode_keyword);
