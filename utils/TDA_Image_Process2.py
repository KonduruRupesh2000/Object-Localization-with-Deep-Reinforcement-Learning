__author__ = 'Robin Vandaele'

import numpy as np
from scipy import sparse
from ripser import lower_star_img
from scipy.sparse.csgraph import connected_components
import cv2
import random
from scipy import ndimage
import PIL
from scipy.ndimage.morphology import distance_transform_edt
from skimage.measure import find_contours

def img_to_sparseDM(img):
    """
    Compute a sparse distance matrix from the pixel entries of a single channel image for persistent homology
    
    Parameters
    ----------
    img: ndarray (M, N)
        An array of single channel image data
        Infinite entries correspond to empty pixels
        
    Returns
    -------
    sparseDM: scipy.sparse (M * N, M * N)
        A sparse distance matrix representation of img
    """
    m, n = img.shape

    idxs = np.arange(m * n).reshape((m, n))

    I = idxs.flatten()
    J = idxs.flatten()
    
    # Make sure non-finite pixel entries get added at the end of the filtration
    img[img==-np.inf] = np.inf
    V = img.flatten()

    # Connect 8 spatial neighbors
    tidxs = np.ones((m + 2, n + 2), dtype=np.int64) * np.nan
    tidxs[1:-1, 1:-1] = idxs

    tD = np.ones_like(tidxs) * np.nan
    tD[1:-1, 1:-1] = img

    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:

            if di == 0 and dj == 0:
                continue

            thisJ = np.roll(np.roll(tidxs, di, axis=0), dj, axis=1)
            thisD = np.roll(np.roll(tD, di, axis=0), dj, axis=1)
            thisD = np.maximum(thisD, tD)

            # Deal with boundaries
            boundary = ~np.isnan(thisD)
            thisI = tidxs[boundary]
            thisJ = thisJ[boundary]
            thisD = thisD[boundary]

            I = np.concatenate((I, thisI.flatten()))
            J = np.concatenate((J, thisJ.flatten()))
            V = np.concatenate((V, thisD.flatten()))
            
    return sparse.coo_matrix((V, (I, J)), shape=(idxs.size, idxs.size))

def connected_components_img(img):
    """
    Identify the connected components of an image
    
    Parameters
    ----------
    img: ndarray (M, N)
        An array of single channel image data
        Infinite entries correspond to empty pixels
        
    Returns
    -------
    img: ndarray (M, N)
        An array of single channel image data where each pixel value equals its modified lifespan
    """
    
    m, n = img.shape
    
    component = connected_components(img_to_sparseDM(img), directed=False)[1].reshape((m, n))
            
    return component

def smoothen(img, window_size):
    return(ndimage.uniform_filter(img.astype("float"), size=window_size))

def add_border(img, border_width):
    border_value = np.min(img) - 1 # make sure the pixels near the border reach te minimal value
    
    img[0:border_width,:] = border_value
    img[(img.shape[0] - border_width):img.shape[0],:] = border_value
    img[:,0:border_width] = border_value
    img[:,(img.shape[1] - border_width):img.shape[1]] = border_value

    return(img)

def lifetimes_from_dgm(dgm, tau=False):
    """
    Rotate a persistence diagram by 45 degrees, to indicate lifetimes by the y-coordinate
    
    Parameters
    ----------
    dgm: ndarray (K, 2)
        The persistence diagram to rotate
    tau: boolean
        Whether to return a threshold for indentifying connected components
        
    Returns
    -------
    dgm_lifetimes: ndarray (K, 2)
        The rotated diagram
        
    tau: float
        A threshold for identifying connected components 
        as those with finite oordinate above tau in the rotated diagram
    """ 
    
    dgm_lifetimes = np.vstack([dgm[:,0], dgm[:,1] - dgm[:,0]]).T
    #row: birth, column: death-birth
        
    if(tau):
        #remove death-birth=infinite
        dgm_for_tau = np.delete(dgm_lifetimes.copy(), np.where(dgm_lifetimes[:,1] == np.inf)[0], axis=0)
        #sort according to lifetime
        sorted_points = dgm_for_tau[:,1]
        
        sorted_points[::-1].sort()
        dist_to_next = np.delete(sorted_points, len(sorted_points) - 1) - np.delete(sorted_points, 0)
        most_distant_to_next = np.argmax(dist_to_next)
        tau = (sorted_points[most_distant_to_next] + sorted_points[most_distant_to_next + 1]) / 2
        
        return dgm_lifetimes, tau
    
    return dgm_lifetimes

def contour_segmentation(img, isovalue=None, return_contours=False):
    if isovalue is None:
        isovalue = np.mean(img)
    
    contours = find_contours(img, isovalue)
    img_segmented = np.zeros_like(img)
    for contour in contours:
        contour = np.int32(contour[:,range(1, -1, -1)]).reshape([1, contour.shape[0], contour.shape[1]])
        cv2.fillPoly(img_segmented, contour, 1)
        
    if return_contours:
        return img_segmented, contours
    
    return img_segmented

def topological_process_img(imgs,img, dgm=None, tau=None, window_size=None, border_width=None):
    return_modified = False
    if dgm is None:
        if window_size is not None:
           img = smoothen(img, window_size=window_size)
           return_modified = True
            
        if border_width is not None:
            img = add_border(img, border_width=border_width)
            return_modified = True
            
        dgm = lower_star_img(img)
    
    if tau is None:
        dgm_lifetimes, tau = lifetimes_from_dgm(dgm, tau=True)
        
    else:
        dgm_lifetimes = lifetimes_from_dgm(dgm)
        
    idxs = np.where(np.logical_and(tau < dgm_lifetimes[:,1], dgm_lifetimes[:,1] < np.inf))[0]
    idxs = np.flip(idxs[np.argsort(dgm[idxs, 0])])
    didxs = np.zeros(0).astype("int")
    
    img_components = np.zeros_like(img,dtype='bool')

    for i, idx in enumerate(idxs):
        bidx = np.argmin(np.abs(img - dgm[idx, 0]))
        didxs = np.append(didxs, np.argmin(np.abs(img - dgm[idx, 1])))

        img_temp = np.ones_like(img,dtype='float')
        img_temp[np.logical_or(img < dgm[idx, 0] - 0.01, dgm[idx, 1] - 0.01 < img)] = np.nan
        component_at_idx = connected_components_img(img_temp)
        del(img_temp)

        component_at_idx = component_at_idx == component_at_idx[bidx // img.shape[1], bidx % img.shape[1]]
        if i > 0:
            didxs_in_component = idxs[np.where([component_at_idx[didx // img.shape[1], didx % img.shape[1]] 
                                                for didx in didxs])[0]]
            if len(didxs_in_component) > 0:
                component_at_idx[img > np.min(dgm[didxs_in_component, 1]) - 0.1] = False
        img_components[component_at_idx] = True

    col=np.amax(img_components, axis=0)
    row=np.amax(img_components, axis=1)
    t=0
    while not row[t]:
        t+=1
    b=img_components.shape[0]-1
    while not row[b]:
        b-=1
    l=0
    while not col[l]:
        l+=1
    r=img_components.shape[1]-1
    while not col[r]:
        r-=1
    
    return [t,b,l,r]

def get_metrics(img_predicted, img_true):
    """
    Evaluate the performance 
    
    Parameters
    ----------
    img_predicted: ndarray (M, N)
        A binary segmented image
    img_true: ndarray (M, N)
        The true binary segmentation of the image
        
    Returns
    -------
    dictionary:
        A dictionary containing the accuracy, mcc, dice, and inclusion score for the performed segmentation
    """

    tp = np.sum(np.logical_and(img_true, img_predicted))
    fp = np.sum(np.logical_and(1 - img_true, img_predicted))
    tn = np.sum(np.logical_and(1 - img_true, 1 - img_predicted))
    fn = np.sum(np.logical_and(img_true,  1 - img_predicted))

    accuracy = (tp + tn) / (tp + fp + tn + fn)
    mcc_denom = np.sqrt(tp + fp) * np.sqrt(tp + fn) * np.sqrt(tn + fp) * np.sqrt(tn + fn)
    if mcc_denom == 0:
        mcc = -1
    else:
        mcc = ((tp * tn) - (fp * fn)) / mcc_denom
    dice = 2 * tp / (2 * tp + fp + fn)
    inclusion = tp / (tp + fn)

    return {"accuracy": accuracy, "mcc": mcc, "dice": dice, "inclusion": inclusion}