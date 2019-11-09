def encode(solution, lower_bound, up_bound, length, mode):
    """Encode a solution with binary or float code.
    Args:
        solution (float or int): a single variable to encode.
        length (int): code length.
        mode (binary or real): encode mode, default binary. 
    """
    if mode == "binary":
        precision = (up_bound - lower_bound) / (2 ** length - 1)
        code_dec = int((solution - lower_bound) / precision)
        code_bin = bin(code_dec)
        code = str(code_bin).lstrip("0b").zfill(length)
    elif mode == 'real':
        code = solution
    else:
        raise ValueError("Unkonw code mode.")
    return code


def decode(code, lower_bound, up_bound, length, mode):
    """Decode a code to a valid solution.
    Args:
        code (binary or real): a single variable to decode.
        length (int): code length.
        mode (binary or real): code type, default binary. 
    """
    if mode == "binary":
        precision = (up_bound - lower_bound) / (2 ** length - 1)
        code_dec = int(code, 2)
        solution = lower_bound + precision * code_dec
    elif mode == 'real':
        solution = code
    else:
        raise ValueError("Unkonw code mode.")
    return solution