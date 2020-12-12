import cv2
import numpy as np
import scipy as sp
import scipy.ndimage
from PIL import Image
from skimage.morphology import disk

def local_mean(arr,r):
    (rows,cols) = arr.shape[:2];
    # print(rows,cols)
    out = np.zeros((rows,cols))
    mask = np.zeros((2*r+1,2*r+1))
    # print(mask)
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
    #xoa cols va rows 0
    # for i in range(r):
    #     arr = np.delete(arr,0,axis=0)
    #     arr = np.delete(arr,arr.shape[0]-1,axis=0)
    # for i in range(r):
    #     arr = np.delete(arr,0,axis=1)
    #     arr = np.delete(arr,arr.shape[1]-1,axis=1)
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
    print('a...',a[:,:,1])
    b = mP - a[:,:,0] * mI_r - a[:,:,1] * mI_g - a[:,:,2] * mI_b
    print('mP...',mP)

    meanA = box(a, r) / N[...,np.newaxis]
    print('hello',meanA[:,:,0])
    meanB = box(b, r) / N
    print('b...',meanB)

    # if s is not None:
    #     meanA = sp.ndimage.zoom(meanA, [s, s, 1], order=1)
    #     meanB = sp.ndimage.zoom(meanB, [s, s], order=1)

    q = np.sum(meanA * fullI, axis=2) + meanB
    print('npsss',np.sum(meanA * fullI, axis=2))
    print ('q..',q)

    return q


def _gf_gray(I, p, r, eps, s=None):
    """ grayscale (fast) guided filter
        I - guide image (1 channel)
        p - filter input (1 channel)
        r - window raidus
        eps - regularization (roughly, allowable variance of non-edge noise)
        s - subsampling factor for fast guided filter
    """
    if s is not None:
        Isub = sp.ndimage.zoom(I, 1/s, order=1)
        Psub = sp.ndimage.zoom(p, 1/s, order=1)
        r = round(r / s)
    else:
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

    if s is not None:
        meanA = sp.ndimage.zoom(meanA, s, order=1)
        meanB = sp.ndimage.zoom(meanB, s, order=1)

    q = meanA * I + meanB
    return q


def _gf_colorgray(I, p, r, eps, s=None):
    """ automatically choose color or gray guided filter based on I's shape """
    if I.ndim == 2 or I.shape[2] == 1:
        print('hello')
        return _gf_gray(I, p, r, eps, s)
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
        out[:,:,ch] = _gf_colorgray(I, p3[:,:,ch], r, eps, s)
    return np.squeeze(out) if p.ndim == 2 else out  #//squeeze de xoa 1 (h,w,10 => (h,w)


def test_gf():
    import imageio
    import cv2
    # cat = imageio.imread('cat.bmp').astype(np.float32) / 255
    # print(tulips[:,:,0])

    r = 8
    eps = 0.05

    # cat_smoothed = guided_filter(cat, cat, r, eps)
    # cat_smoothed_s4 = guided_filter(cat, cat, r, eps, s=4)
    #
    # imageio.imwrite('cat_showsmoothed4.png', cat_smoothed)
    # imageio.imwrite('cat_smoothed_s4.png', cat_smoothed_s4)

    # tulips_smoothed4s = np.zeros_like(tulips)
    # for i in range(3):
    #     tulips_smoothed4s[:,:,i] = guided_filter(tulips, tulips[:,:,i], r, eps, s=4)
    # imageio.imwrite('tulips_smoothed4s.png', tulips_smoothed4s)

    tulips = cv2.imread('tulips.bmp').astype(np.float32) / 255
    tulips_smoothed = np.zeros_like(tulips)
    for i in range(3):
        tulips_smoothed[:, :, i] = guided_filter(tulips, tulips[:, :, i], r, eps)
    print(tulips_smoothed)
    print((tulips_smoothed * 255).astype(np.uint8))
    tulips_smoothed = cv2.cvtColor(tulips_smoothed, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray((tulips_smoothed * 255).astype(np.uint8))
    pil_img.save('tulips_smoothed2.png')
    # imageio.imwrite('tulips_smoothed1.png', tulips_smoothed)
test_gf();

# a = np.array([[5,3,7,2],[11,3,22,8],[9,2,8,22],[6,5,4,4],[7,3,1,4]],dtype=np.float64)
# print(a.ndim)
# N = box(np.ones((5, 4)), 1)
# print(a_mean)
# print(a.shape)
# print(N)
# print(box(a,1)/N)
# print(local_mean(a,1))

