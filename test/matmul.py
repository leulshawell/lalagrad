import numpy as np
import lala as l

def  matmul_int32():
    a1 = l.rand(3, 3)
    a2 = l.rand(3, 3)

    b1 = np.rand(3, 3)
    b2 = np.rand(3, 3)

    assert (a1 * a2).tolist() == (b1 * b2).tolist()



if __name__ == "__main__":
    matmul_int32()
