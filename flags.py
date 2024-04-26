import cv2

GRAYING_STACKED_WIDGET = 0
FILTER_STACKED_WIDGET = 1
MORPH_STACKED_WIDGET = 2
GRAD_STACKED_WIDGET = 3
THRESH_STACKED_WIDGET = 4
EDGE_STACKED_WIDGET = 5

#BGR2GRAY_COLOR = 0
#GRAY2BGR_COLOR = 1
#COLOR = {
#    BGR2GRAY_COLOR: cv2.COLOR_BGR2GRAY,
#    GRAY2BGR_COLOR: cv2.COLOR_GRAY2BGR
#}

MEAN_FILTER = 0
GAUSSIAN_FILTER = 1
MEDIAN_FILTER = 2

ERODE_MORPH_OP = 0
DILATE_MORPH_OP = 1
OPEN_MORPH_OP = 2
CLOSE_MORPH_OP = 3
GRADIENT_MORPH_OP = 4
TOPHAT_MORPH_OP = 5
BLACKHAT_MORPH_OP = 6

MORPH_OP = {
    ERODE_MORPH_OP: cv2.MORPH_ERODE,
    DILATE_MORPH_OP: cv2.MORPH_DILATE,
    OPEN_MORPH_OP: cv2.MORPH_OPEN,
    CLOSE_MORPH_OP: cv2.MORPH_CLOSE,
    GRADIENT_MORPH_OP: cv2.MORPH_GRADIENT,
    TOPHAT_MORPH_OP: cv2.MORPH_TOPHAT,
    BLACKHAT_MORPH_OP: cv2.MORPH_BLACKHAT
}

RECT_MORPH_SHAPE = 0
CROSS_MORPH_SHAPE = 1
ELLIPSE_MORPH_SHAPE = 2

MORPH_SHAPE = {
    RECT_MORPH_SHAPE: cv2.MORPH_RECT,
    CROSS_MORPH_SHAPE: cv2.MORPH_CROSS,
    ELLIPSE_MORPH_SHAPE: cv2.MORPH_ELLIPSE
}

SOBEL_GRAD = 0
SCHARR_GRAD = 1
LAPLACIAN_GRAD = 2

BINARY_THRESH_METHOD = 0
BINARY_INV_THRESH_METHOD = 1
TRUNC_THRESH_METHOD = 2
TOZERO_THRESH_METHOD = 3
TOZERO_INV_THRESH_METHOD = 4
OTSU_THRESH_METHOD = 5
THRESH_METHOD = {
    BINARY_THRESH_METHOD: cv2.THRESH_BINARY,  # 0
    BINARY_INV_THRESH_METHOD: cv2.THRESH_BINARY_INV,  # 1
    TRUNC_THRESH_METHOD: cv2.THRESH_TRUNC,  # 2
    TOZERO_THRESH_METHOD: cv2.THRESH_TOZERO,  # 3
    TOZERO_INV_THRESH_METHOD: cv2.THRESH_TOZERO_INV,  # 4
    OTSU_THRESH_METHOD: cv2.THRESH_OTSU  # 5
}

EXTERNAL_CONTOUR_MODE = 0
LIST_CONTOUR_MODE = 1
CCOMP_CONTOUR_MODE = 2
TREE_CONTOUR_MODE = 3
CONTOUR_MODE = {
    EXTERNAL_CONTOUR_MODE: cv2.RETR_EXTERNAL,
    LIST_CONTOUR_MODE: cv2.RETR_LIST,
    CCOMP_CONTOUR_MODE: cv2.RETR_CCOMP,
    TREE_CONTOUR_MODE: cv2.RETR_TREE
}

NONE_CONTOUR_METHOD = 0
SIMPLE_CONTOUR_METHOD = 1
CONTOUR_METHOD = {
    NONE_CONTOUR_METHOD: cv2.CHAIN_APPROX_NONE,
    SIMPLE_CONTOUR_METHOD: cv2.CHAIN_APPROX_SIMPLE
}

NORMAL_CONTOUR = 0
RECT_CONTOUR = 1
MINRECT_CONTOUR = 2
MINCIRCLE_CONTOUR = 3


# 均衡化
BLUE_CHANNEL = 0
GREEN_CHANNEL = 1
RED_CHANNEL = 2
ALL_CHANNEL = 3
