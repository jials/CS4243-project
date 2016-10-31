import numpy as np
import numpy.linalg as la

def find_homography(plane1, plane2):
    num_points = len(plane1)
    M = np.zeros([num_points * 2, 9])
    for i, j in zip(range(0, num_points * 2, 2), range(0, num_points)):
        point = np.append(plane1[j], [1])
        M[i, :] = np.append(np.append(point, [0, 0, 0]), plane2[j][0] * -point)
        M[i+1, :] = np.append(np.append([0, 0, 0], point), plane2[j][1] * -point)

    U, S, VT = la.svd(M)
    min_index = np.argmin(S)
    # if the smallest value is bigger than threshold
    if S[min_index] > 0.000001:
        min_index = -1

    H = np.array(VT[min_index, :])
    H = np.resize(H, (3, 3))
    H = H / H[2, 2]  # TODO: normalise with h_33
    return H