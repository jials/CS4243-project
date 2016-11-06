import numpy as np
import numpy.linalg as la

def find_homography(plane1, plane2):
    num_points = len(plane1)
    M = np.zeros([num_points * 2, 9])
    for i, j in zip(range(0, num_points * 2, 2), range(0, num_points)):
        point = np.append(plane1[j], [1])  # make it 3-dimensional
        M[i, :] = np.append(np.append(point, [0, 0, 0]), plane2[j][0] * -point)
        M[i+1, :] = np.append(np.append([0, 0, 0], point), plane2[j][1] * -point)

    U, S, VT = la.svd(M)
    min_index = -1
    for idx, element in enumerate(S):
        if np.abs(element) < 0.0001:
            min_index = idx

    H = np.array(VT[min_index, :])
    H = np.resize(H, (3, 3))
    H = H / H[2, 2]  # TODO: normalise with h_33
    return H