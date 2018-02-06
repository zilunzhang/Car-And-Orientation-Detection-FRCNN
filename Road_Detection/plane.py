from scipy.optimize import leastsq
import numpy as np
import random
from matplotlib import pylab
from mpl_toolkits import mplot3d

# np.random.seed(1)

def residual(coeffs, X):
    plane = coeffs[0:3]
    distance = np.sum(plane * X, axis=1) + coeffs[3]
    return distance / np.linalg.norm(plane)

# initial guess of fitted line
def fitted_plane(p0, X):
    return leastsq(residual, p0, args=(X))[0]



# https://stackoverflow.com/questions/38754668/plane-fitting-in-a-3d-point-cloud

def PCA(data, correlation = False, sort = True):
    mean = np.mean(data, axis=0)

    data_adjust = data - mean

    #: the data is transposed due to np.cov/corrcoef syntax
    if correlation:

        matrix = np.corrcoef(data_adjust.T)

    else:
        matrix = np.cov(data_adjust.T)

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    if sort:
        #: sort eigenvalues and eigenvectors
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:,sort]

    return eigenvalues, eigenvectors

def best_fitting_plane(points, equation=True):

    w, v = PCA(points)

    #: the normal of the plane is the last eigenvector
    normal = v[:,2]

    #: get a point from the plane
    point = np.mean(points, axis=0)


    if equation:
        a, b, c = normal
        d = -(np.dot(normal, point))
        return a, b, c, d

    else:
        return point, normal



def run_ransac(data, estimate, is_inlier, sample_size, goal_inliers, max_iterations, stop_at_goal=True):
    best_ic = 0
    best_model = None
    for i in range(max_iterations):
        indices = np.random.choice(data.shape[0], sample_size)
        s = np.take(data, indices, axis=0)
        m = estimate(s)
        ic = 0
        for j in range(len(data)):
            if is_inlier(m, data[j], 40):
                ic += 1

        if ic > best_ic:
            best_ic = ic
            best_model = m
            if ic > goal_inliers and stop_at_goal:
                break

    return best_model


def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz

def estimate(xyzs):
    axyz = augment(xyzs[:3])
    return np.linalg.svd(axyz)[-1][-1, :]

def is_inlier(coeffs, xyz, threshold):
    return np.abs(coeffs.dot(augment([xyz]).T)) < threshold


def plot_plane(a, b, c, d, min_x, max_x, min_y, max_y):
    xx, yy = np.mgrid[min_x:max_x, min_y:max_y]
    return xx, yy, (-d - a * xx - b * yy) / c


if __name__ == '__main__':
    # load coordinates
    # road_coor = np.loadtxt('../left/3d_coor2.npy')
    path = '../left/3d_coor2'
    road_coor = np.loadtxt(path + '.out', delimiter=',')

    road_coor = road_coor.astype(float)

    road_coor[:, 2] = road_coor[:, 2] - 5

    fig = pylab.figure()

    ax = mplot3d.Axes3D(fig)

    n = road_coor.shape[0]
    max_iterations = 100
    goal_inliers = n * 0.4

    # test data
    # xyzs = np.random.random((n, 3)) * 10
    # xyzs[:50, 2:] = xyzs[:50, :1]

    # ax.scatter3D(xyzs.T[0], xyzs.T[1], xyzs.T[2])

    ax.scatter3D(road_coor.T[0], road_coor.T[1], road_coor.T[2])

    # RANSAC
    # m = run_ransac(road_coor, estimate, is_inlier, 20, goal_inliers, max_iterations)
    # a, b, c, d = m

    a, b, c, d = best_fitting_plane(road_coor)

    # p0 = [0.506645455682, -0.185724560275, -1.43998120646, 1.37626378129]
    # a, b, c, d = fitted_plane(p0, xyzs)

    xx, yy, zz = plot_plane(a, b, c, d, -40, 40, -10, 40)
    counter0 = 10
    counter1 = 40
    ax.plot_surface(xx[counter0:, :counter1], yy[counter0:, :counter1], zz[counter0:, :counter1], color=(0, 1, 0, 0.5)), pylab.show()
