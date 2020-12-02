import math

import cv2
import numpy as np
import scipy as sp
import scipy.ndimage
from PIL import Image;
from skimage.morphology import disk

def local_mean(arr,r):
    (rows,cols) = arr.shape[:2];
    out = np.zeros((rows,cols))
    mask = np.zeros((2*r+1,2*r+1))
    # truyen cols vaf rows 0 vao
    zeros_1 = np.zeros((1,arr.shape[1]))
    for i in range(r):
        arr = np.insert(arr,arr.shape[0],zeros_1,axis=0)
        arr = np.insert(arr,0,zeros_1,axis=0)
    zeros_2 = np.zeros((1,arr.shape[0]))
    for i in range(r):
        arr = np.insert(arr,arr.shape[1],zeros_2,axis=1)
        arr = np.insert(arr,0,zeros_2,axis=1)
    for i in range(arr.shape[0]-(2*r)):
        for j in range(arr.shape[1]-(2*r)):
            # print(j)
            mask = arr[i:i+2*r+1,j:j+2*r+1]
            out[i,j] = np.mean(mask)
    return out
def box(img, r):
    """ O(1) box filter
        img - >= 2d image
        r   - radius of box filter
    """
    (rows, cols) = img.shape[:2]
    imDst = np.zeros_like(img)

    tile = [1] * img.ndim   #[1,1,..,1] hai chieu thi img.ndim la 2 = [1,1]

    tile[0] = r
    # print("img",img)
    imCum = np.cumsum(img, 0)   #1200,800
    imDst[0:r+1, :, ...] = imCum[r:2*r+1, :, ...]
    imDst[r+1:rows-r, :, ...] = imCum[2*r+1:rows, :, ...] - imCum[0:rows-2*r-1, :, ...]
    imDst[rows-r:rows, :, ...] = np.tile(imCum[rows-1:rows, :, ...], tile) - imCum[rows-2*r-1:rows-r-1, :, ...]

    tile = [1] * img.ndim
    tile[1] = r
    imCum = np.cumsum(imDst, 1)
    imDst[:, 0:r+1, ...] = imCum[:, r:2*r+1, ...]
    imDst[:, r+1:cols-r, ...] = imCum[:, 2*r+1 : cols, ...] - imCum[:, 0 : cols-2*r-1, ...]
    imDst[:, cols-r: cols, ...] = np.tile(imCum[:, cols-1:cols, ...], tile) - imCum[:, cols-2*r-1 : cols-r-1, ...]

    return imDst

def _gf_color(I, p, r, eps, s=None):
    """ Color guided filter
    I - guide image (rgb)
    p - filtering input (single channel)
    r - window radius
    eps - regularization (roughly, variance of non-edge noise)
    s - subsampling factor for fast guided filter
    """
    fullI = I
    fullP = p
    if s is not None:
        I = sp.ndimage.zoom(fullI, [1/s, 1/s, 1], order=1)
        p = sp.ndimage.zoom(fullP, [1/s, 1/s], order=1)
        r = round(r / s)
    # print(p)
    h, w = p.shape[:2]
    N = box(np.ones((h, w)), r)
    # print(N)


    mI_r = box(I[:,:,0], r) / N
    mI_g = box(I[:,:,1], r) / N
    mI_b = box(I[:,:,2], r) / N

    mP = box(p, r) / N

    # mean of I * p
    mIp_r = box(I[:,:,0]*p, r) / N
    mIp_g = box(I[:,:,1]*p, r) / N
    mIp_b = box(I[:,:,2]*p, r) / N

    # per-patch covariance of (I, p)
    covIp_r = mIp_r - mI_r * mP
    covIp_g = mIp_g - mI_g * mP
    covIp_b = mIp_b - mI_b * mP

    # symmetric covariance matrix of I in each patch:
    #       rr rg rb
    #       rg gg gb
    #       rb gb bb
    var_I_rr = box(I[:,:,0] * I[:,:,0], r) / N - mI_r * mI_r;
    var_I_rg = box(I[:,:,0] * I[:,:,1], r) / N - mI_r * mI_g;
    var_I_rb = box(I[:,:,0] * I[:,:,2], r) / N - mI_r * mI_b;

    var_I_gg = box(I[:,:,1] * I[:,:,1], r) / N - mI_g * mI_g;
    var_I_gb = box(I[:,:,1] * I[:,:,2], r) / N - mI_g * mI_b;

    var_I_bb = box(I[:,:,2] * I[:,:,2], r) / N - mI_b * mI_b;

    a = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            sig = np.array([
                [var_I_rr[i,j], var_I_rg[i,j], var_I_rb[i,j]],
                [var_I_rg[i,j], var_I_gg[i,j], var_I_gb[i,j]],
                [var_I_rb[i,j], var_I_gb[i,j], var_I_bb[i,j]]
            ])
            covIp = np.array([covIp_r[i,j], covIp_g[i,j], covIp_b[i,j]])
            a[i,j,:] = np.linalg.solve(sig + eps * np.eye(3), covIp)
    # print(a[:,:,1])
    b = mP - a[:,:,0] * mI_r - a[:,:,1] * mI_g - a[:,:,2] * mI_b

    meanA = box(a, r) / N[...,np.newaxis]
    meanB = box(b, r) / N

    # if s is not None:
    #     meanA = sp.ndimage.zoom(meanA, [s, s, 1], order=1)
    #     meanB = sp.ndimage.zoom(meanB, [s, s], order=1)

    q = np.sum(meanA * fullI, axis=2) + meanB

    return q


def _gf_gray(I, p, r, eps, s=None):
    """ grayscale (fast) guided filter
        I - guide image (1 channel)
        p - filter input (1 channel)
        r - window raidus
        eps - regularization (roughly, allowable variance of non-edge noise)
        s - subsampling factor for fast guided filter
    """

    Isub = I
    Psub = p


    (rows, cols) = Isub.shape

    N = box(np.ones([rows, cols]), r)


    # meanI = box(Isub, r) / N
    meanI = local_mean(Isub,r);
    meanP = local_mean(Psub,r);
    corrI = local_mean(Isub * Isub,r);
    corrIp = local_mean(Isub * Psub,r);
    varI = corrI - meanI * meanI
    covIp = corrIp - meanI * meanP


    a = covIp / (varI + eps)
    b = meanP - a * meanI
    # print("a.....",a)

    meanA = local_mean(a,r);
    meanB = local_mean(b,r);

    q = meanA * I + meanB
    return q
def mean_of_all_guidance_at_pixel_k(i,j,local_var_I,local_var_I1):
    return (local_var_I[i,j]+local_var_I1[i,j])/2
def edge_aware_constraint(i,j,local_var_I,local_var_I1,covI1p,covIp):
    global edge_aware_mu
    edge_aware_mu = np.array([])
    t = local_var_I[i,j]/np.mean(local_var_I)
    edge1 = (2/(1+math.exp(-t)) - 1)
    if(covIp[i,j] > 0 ) :
        edge_aware_mu = np.append(edge_aware_mu,edge1*1)
    else :
        edge_aware_mu = np.append(edge_aware_mu,edge1*(-1))

    t = local_var_I1[i,j]/np.mean(local_var_I1)
    edge1 = (2/(1+math.exp(-t)) - 1)
    if (covI1p[i,j] > 0):
        edge_aware_mu = np.append(edge_aware_mu, edge1 * 1)
    else:
        edge_aware_mu = np.append(edge_aware_mu, edge1 * (-1))

    return edge_aware_mu;
def edge_aware(i,j,eps,local_var_I,local_var_I1):
    e = (0.001*256)**2
    wk1 = (eps * np.mean(local_var_I) + eps * mean_of_all_guidance_at_pixel_k(i,j,local_var_I,local_var_I1)) / (local_var_I[i,j]+e)
    wk2 = (eps * np.mean(local_var_I1) + eps * mean_of_all_guidance_at_pixel_k(i,j,local_var_I,local_var_I1)) / (local_var_I1[i,j]+e)
    Wk = np.array([wk1,wk2]);
    return Wk;


def _gf_gray_multi(I,I1,p,r,eps):
    (rows, cols) = I.shape
    N = box(np.ones([rows, cols]), r)

    meanI = local_mean(I,r)
    meanI1 = local_mean(I1,r)
    meanII = local_mean(I * I,r)
    meanI1I1 = local_mean(I1*I1,r)
    meanII1 = local_mean(I*I1,r)
    meanIp = local_mean(I*p,r)
    meanI1p = local_mean(I1*p,r)
    meanp = local_mean(p,r)
    # phuong sai kenh 1
    local_varianceI = meanII - meanI*meanI
    # phuong sai kenh 2
    local_varianceI1 = meanI1I1 - meanI1*meanI1
    # hiep phuong sai giua hai channel
    covII = meanII - meanI*meanI
    covII1 = meanII1 - meanI*meanI1
    covI1I = meanII1 - meanI*meanI1
    covI1I1 = meanI1I1 - meanI1*meanI1
    # hiep phuong sai kenh va inputimage
    covI1p = meanI1p - meanI1 * meanp
    covIp = meanIp - meanI * meanp
    #tao 2 mang chua akj ung voi 2 kenh
    Akk = np.zeros((rows,cols,2))
    for i in range(rows):
        for j in range(cols):
            W = np.diag(edge_aware(i,j,eps,local_varianceI,local_varianceI1))
            Cj1j2 = np.array([ [covII[i,j],covI1I[i,j]] , [covII1[i,j],covI1I1[i,j]] ]);
            edge_aware_const = edge_aware(i,j,eps,local_varianceI,local_varianceI1)*edge_aware_constraint(i,j,local_varianceI,local_varianceI1,covI1p,covIp)
            Cj0 = np.array([covIp[i,j],covI1p[i,j]])

            Ak =np.dot(np.linalg.inv(Cj1j2 + W),(Cj0 + edge_aware_const).T)
            Akk[i,j,0] = Ak[0]
            Akk[i,j,1] = Ak[1]
    print(Akk[:, :, 0]);
    return 0;

    bk = meanp - Akk[:,:,0]* meanI - Akk[:,:,1]* meanI1;

    q = (Akk[:,:,0]/N * I) + (Akk[:,:,1]/N * I1) + bk/N
    return q

    # print(meanI)
def _gf_colorgray(I, p, r, eps, s=None):
    """ automatically choose color or gray guided filter based on I's shape """
    if I.ndim == 2 or I.shape[2] == 1:
        print('hello')
        # return _gf_gray_multi(I, I1, p, r, eps, s)
    elif I.ndim == 3 and I.shape[2] == 3:
        return _gf_color(I, p, r, eps, s)
    else:
        print("Invalid guide dimensions:", I.shape)


def guided_filter(I, p, r, eps, s=None):
    """ run a guided filter per-channel on filtering input p
        I - guide image (1 or 3 channel)
        p - filter input (n channel)
        r - window raidus
        eps - regularization (roughly, allowable variance of non-edge noise)
        s - subsampling factor for fast guided filter
    """
    # print("p...",p)
    if p.ndim == 2:
        p3 = p[:,:,np.newaxis]   #(h,w,1)
    out = np.zeros_like(p3)
    # print(p3[:,:,0])
    for ch in range(p3.shape[2]):
        # print(ch)
        print(p3[:,:,ch])
        out[:,:,ch] = _gf_colorgray(I, p3[:,:,ch], r, eps, s)
    return np.squeeze(out) if p.ndim == 2 else out  #//squeeze de xoa 1 (h,w,1) => (h,w)


def test_gf():
    import imageio
    # cat = imageio.imread('cat.bmp').astype(np.float32) / 255
    img1 = cv2.imread('img1_1.png').astype(np.float32) / 255
    img2 = cv2.imread('img2_2.png').astype(np.float32) / 255
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    # print(cat.shape)
    print(img1);
    r = 8
    eps = 0.05
    result = _gf_gray_multi(img1,img2,img1,r,eps)
    print(result)
    # image_result = Image.fromarray((result * 255).astype(np.uint8))
    # image_result.save('hello.png')
    # print(result)

test_gf();



