import numpy as np
import matplotlib.pyplot as plt
import cv2


def get_seg(box_matrix, matrix_3d):

    threshold = 1
    seg_matrix = np.matrix((box_matrix.shape[0], 12))

    x_list = np.arange(0, matrix_3d.shape[1], 1)
    y_list = np.arange(0, matrix_3d.shape[0], 1)
    x_vector = np.asarray(x_list).reshape(1, (len(x_list)))
    y_vector = np.asarray(y_list).reshape((len(y_list), 1))
    x_matrix = np.tile(x_list, (y_vector.shape[0], 1))
    y_matrix = np.tile(y_list.T, (x_vector.shape[1], 1)).T

    X_matrix = matrix_3d[:, :, 0]
    Y_matrix = matrix_3d[:, :, 1]
    depth_matrix = matrix_3d[:, :, 2]

    # container for mass center
    mass_centers = np.zeros((box_matrix.shape[0], 3))
    for i in range(box_matrix.shape[0]):
        x_left = np.int(box_matrix[i][0])
        y_top = np.int(box_matrix[i][1])
        x_right = np.int(box_matrix[i][2])
        y_bottom = np.int(box_matrix[i][3])

        Z = depth_matrix[y_top:y_bottom, x_left:x_right]
        X = X_matrix[y_top:y_bottom, x_left:x_right]
        Y = Y_matrix[y_top:y_bottom, x_left:x_right]
        x = x_matrix[y_top:y_bottom, x_left:x_right]
        y = y_matrix[y_top:y_bottom, x_left:x_right]

        x_median = np.median(X)
        y_median = np.median(Y)
        z_median = np.median(Z)

        mass_centers[i,0] = x_median
        mass_centers[i,1] = y_median
        mass_centers[i,2] = z_median

        max_x, max_y, max_z = 0, 0, 0
        min_x, min_y, min_z = 9999, 9999, 9999
        for m in range(X.shape[0]):
            for n in range(X.shape[1]):
                distance = np.power((X[m, n] - x_median), 2) + \
                           np.power((Y[m, n] - y_median), 2) + \
                           np.power((Z[m, n] - z_median), 2)
                if np.power(distance, 1/2) < threshold:
                    if X[m, n] > max_x:
                        max_x = X[m, n]
                        seg_matrix[i, 0] = x[m, n]
                        seg_matrix[i, 1] = y[m, n]
                    if X[m, n] < min_x:
                        min_x = X[m, n]
                        seg_matrix[i, 2] = x[m, n]
                        seg_matrix[i, 3] = y[m, n]
                    if Y[m, n] > max_y:
                        max_y = Y[m, n]
                        seg_matrix[i, 4] = x[m, n]
                        seg_matrix[i, 5] = y[m, n]
                    if Y[m, n] < min_y:
                        min_y = Y[m, n]
                        seg_matrix[i, 6] = x[m, n]
                        seg_matrix[i, 7] = y[m, n]
                    if Z[m, n] > max_z:
                        max_z = Z[m, n]
                        seg_matrix[i, 8] = x[m, n]
                        seg_matrix[i, 9] = y[m, n]
                    if Z[m, n] < min_z:
                        min_z = Z[m, n]
                        seg_matrix[i, 10] = x[m, n]
                        seg_matrix[i, 11] = y[m, n]

    return mass_centers, seg_matrix


def draw_box(img, seg_matrix):
    for box_num in range(seg_matrix.shape[0]):
        seg_val = seg_matrix[box_num, :]

        # top layer plane:
        t_center = [seg_val[4], seg_val[5]]
        x_ver_len = seg_val[0] - seg_val[2]
        x_hor_len = seg_val[1] - seg_val[3]
        z_ver_len = seg_val[8] - seg_val[10]
        z_hor_len = seg_val[9] - seg_val[11]
        # four top layer edge centers
        tl_center = [t_center[0] - x_ver_len / 2, t_center[1] - x_hor_len / 2]
        tr_center = [t_center[0] + x_ver_len / 2, t_center[1] + x_hor_len / 2]
        # tb_center = [t_center[0] - z_ver_len / 2, t_center[1] - z_hor_len / 2]
        tt_center = [t_center[0] + z_ver_len / 2, t_center[1] + z_hor_len / 2]
        # four top layer corners
        ttl = [tl_center[0] + tt_center[0] - t_center[0], tl_center[1] + tt_center[1] - t_center[1]]
        ttr = [tr_center[0] + tt_center[0] - t_center[0], tr_center[1] + tt_center[1] - t_center[1]]
        tbl = [tl_center[0] - tt_center[0] + t_center[0], tl_center[1] - tt_center[1] + t_center[1]]
        tbr = [tr_center[0] - tt_center[0] + t_center[0], tr_center[1] - tt_center[1] + t_center[1]]

        # bottom layer plane:
        b_center = [seg_val[6], seg_val[7]]
        # four bottom layer edge centers
        bl_center = [b_center[0] - x_ver_len / 2, b_center[1] - x_hor_len / 2]
        br_center = [b_center[0] + x_ver_len / 2, b_center[1] + x_hor_len / 2]
        # bb_center = [b_center[0] - z_ver_len / 2, b_center[1] - z_hor_len / 2]
        bt_center = [b_center[0] + z_ver_len / 2, b_center[1] + z_hor_len / 2]
        # four bottom layer corners
        btl = [bl_center[0] + bt_center[0] - b_center[0], bl_center[1] + bt_center[1] - b_center[1]]
        btr = [br_center[0] + bt_center[0] - b_center[0], br_center[1] + bt_center[1] - b_center[1]]
        bbl = [bl_center[0] - bt_center[0] + b_center[0], bl_center[1] - bt_center[1] + b_center[1]]
        bbr = [br_center[0] - bt_center[0] + b_center[0], br_center[1] - bt_center[1] + b_center[1]]

        # draw 12 lines
        b1 = np.array([ttl, ttr, tbr, tbl], np.int32)
        b1 = b1.reshape((-1, 1, 2))
        img = cv2.polylines(img, b1, True, (255, 0, 0))
        b2 = np.array([btl, btr, bbr, bbl], np.int32)
        b2 = b2.reshape((-1, 1, 2))
        img = cv2.polylines(img, b2, True, (255, 0, 0))
        b3 = np.array([ttl, ttr, btr, btl], np.int32)
        b3 = b3.reshape((-1, 1, 2))
        img = cv2.polylines(img, b3, True, (255, 255, 0))
        b4 = np.array([tbl, tbr, bbr, bbl], np.int32)
        b4 = b4.reshape((-1, 1, 2))
        img = cv2.polylines(img, b4, True, (255, 0, 0))
        plt.imshow(img), plt.show()


if __name__ == '__main__':
    img = cv2.imread('../left/testing/image/um_000000.png')
    seg_matrix = np.array([[210, 1000, 100, 600, 50, 850, 250, 800, 200, 1100, 180, 650]])
    draw_box(img, seg_matrix)