import numpy as np
import cupy as cp
import time

def test_mat_mul():
    SIZE_A = 50000000
    SIZE_B = 5

    a = np.random.rand(SIZE_B, SIZE_A)
    b = np.random.rand(SIZE_A, SIZE_B)

    
    a_cp = cp.array(a)
    b_cp = cp.array(b)
    c_start = time.time()
    result_cp = cp.matmul(a_cp, b_cp)
    c_end = time.time()

    result_np = np.matmul(a, b)
    n_end = time.time()

    print(c_end - c_start)
    print(n_end - c_end)
    

    print(result_np)
    print(result_cp)

test_mat_mul()