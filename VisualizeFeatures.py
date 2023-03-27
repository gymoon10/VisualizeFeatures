'''source: https://github.com/lxtGH/SFSegNets/blob/19cd84a0c78a2fcbc36b59cd7df27f14fb5b75ad/utils/flow_lib/visualize.py'''

from sklearn.decomposition import PCA
import cv2
import numpy as np
import matplotlib.pyplot as plt


def visulizeFeatureMapPCA(feature):
    '''feature: (C, H, W) torch.tensor with CUDA
       img_out: (H, W, 3) in [0, 255] numpy.ndarray, dtype=np.uint8'''
    feature = feature.squeeze().data.cpu().numpy()
    # img_out = np.mean(feature, axis=0)
    c, h, w = feature.shape
    img_out = feature.reshape(c, -1).transpose(1, 0)
    pca = PCA(n_components=3)
    pca.fit(img_out)
    img_out_pca = pca.transform(img_out)
    img_out_pca = img_out_pca.transpose(1, 0).reshape(3, h, w).transpose(1, 2, 0)

    cv2.normalize(img_out_pca, img_out_pca, 0, 255, cv2.NORM_MINMAX)
    img_out_pca = cv2.resize(img_out_pca, (h, w),interpolation=cv2.INTER_LINEAR)
    img_out = np.array(img_out_pca, dtype=np.uint8)

    return img_out


def visulizeFeatureMap(feature):
    '''feature: (C, H, W) torch.tensor with CUDA
       img_out: (H, W, 3) in [0, 255] numpy.ndarray, dtype=np.uint8'''
    feature = feature.data.cpu().numpy()
    img_out = np.mean(feature, axis=0)
    cv2.normalize(img_out, img_out, 0, 255, cv2.NORM_MINMAX)
    img_out = np.array(img_out, dtype=np.uint8)
    img_out = cv2.applyColorMap(img_out, cv2.COLORMAP_JET)
    
    return img_out


def visulizeFeatureMapSingle(feature):
    '''feature: (C, H, W) torch.tensor with CUDA
       img_out: (H, W, 3) in [0, 255] numpy.ndarray, dtype=np.uint8'''
    feature = feature.squeeze().data.cpu().numpy()
    img_out = feature
    cv2.normalize(img_out, img_out, 0, 255, cv2.NORM_MINMAX)
    img_out = np.array(img_out, dtype=np.uint8)
    img_out = cv2.applyColorMap(img_out, cv2.COLORMAP_JET)
    
    return img_out


# visualize 2d (optical) flow
def visualize_flow(flow, mode='Y'):
    """
    this function visualize the input flow
    :param flow: input flow in numpy.ndarray (H, W, 2) w.o CUDA
    *** if original flow is torch.tensor(CUDA) with (N, 2, H, W) you should convert it by 
        flow[0].permute(1,2,0).cpu().detach().numpy()
       
    :param mode: choose which color mode to visualize the flow (Y: Ccbcr, RGB: RGB color)
    :return: None
    """
    if mode == 'Y':
        # Ccbcr color wheel
        img = flow_to_image(flow)
        plt.imshow(img)
        plt.show()
    elif mode == 'RGB':
        (h, w) = flow.shape[0:2]
        du = flow[:, :, 0]
        dv = flow[:, :, 1]
        valid = flow[:, :, 2]
        max_flow = max(np.max(du), np.max(dv))
        img = np.zeros((h, w, 3), dtype=np.float64)
        # angle layer
        img[:, :, 0] = np.arctan2(dv, du) / (2 * np.pi)
        # magnitude layer, normalized to 1
        img[:, :, 1] = np.sqrt(du * du + dv * dv) * 8 / max_flow
        # phase layer
        img[:, :, 2] = 8 - img[:, :, 1]
        # clip to [0,1]
        small_idx = img[:, :, 0:3] < 0
        large_idx = img[:, :, 0:3] > 1
        img[small_idx] = 0
        img[large_idx] = 1
        # convert to rgb
        img = cl.hsv_to_rgb(img)
        # remove invalid point
        img[:, :, 0] = img[:, :, 0] * valid
        img[:, :, 1] = img[:, :, 1] * valid
        img[:, :, 2] = img[:, :, 2] * valid
        # show
        plt.imshow(img)
        plt.show()
    return None


# util functions for 'visualize_flow'
def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8

def flow_to_image(flow, display=False):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    if display:
        print("max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu,maxu, minv, maxv))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)

