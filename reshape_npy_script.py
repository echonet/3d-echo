# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pydicom as dicom
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm

def extract_points(x1, v1, v2, w, h, d):
    '''
    :param x1: the center point of the slice
    :param v1: one vector in the slice
    :param v2: another vector in the slice
    :param w: width of the slice
    :param h: height of the slice
    :param d: depth of the slice
    :return:
    p1, p2, p3, p4：4 points that can form 2 perpendicular vectors and a depth vector
    '''
    # convert the points and vectors to be np.array
    x1, v1, v2 = np.array(x1), np.array(v1), np.array(v2)
    
    # normalize the 2 given vectors
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # calculate the normal vector and normalize the normal vector
    nv = np.cross(v1, v2)
    nv = nv / np.linalg.norm(nv)
    [a,b,c] = nv
    
    # use the unit normal vector to get the 90 degree rotation matrix
    rotate_matrix = np.array([[a**2, a*b-c, a*c+b],[b*a+c, b**2, b*c-a],[c*a-b, c*b+a, c**2]])
    
    # calculate v1's 90 degree rotated vector and normalize it
    v1_p = rotate_matrix@v1.T
    v1_p = v1_p / np.linalg.norm(v1_p)
    
    # get the 3 perpendicular points we want for the slice area
    p1 = x1 + h/2*v1 - w/2*v1_p
    p2 = x1 - h/2*v1 - w/2*v1_p
    p3 = x1 - h/2*v1 + w/2*v1_p
    p4 = p2 + d/2*nv
    p1 = p1 - d/2*nv
    p2 = p2 - d/2*nv
    p3 = p3 - d/2*nv
    
    return p1, p2, p3, p4

def slice_area(img, p1, p2, p3, p4, rho_range, theta_range, phi_range, rx, ry, rz):
    '''
    :param img: 3d pixel array
    :param p1,p2,p3,p4: 4 points that can form 2 perpendicular vectors and a depth vector
    :param  rho_range, theta_range, phi_range: polar coordinate ranges
    :rx: the resolution on x-axis
    :ry: the resolution on y-axis
    :rz: the resolution on z-axis
    :return:
    slicer：3d pixel array
    '''
    # convert all the input points to be np.array
    p1, p2, p3, p4 = np.array(p1), np.array(p2), np.array(p3), np.array(p4)
    
    
    # ratio for linear interpolation and apply them on the line between two points
    t1 = np.linspace(0, 1, rx)[:, None]
    u = t1 * p3[None, :] + (1 - t1) * p2[None, :]
    
    t2 = np.linspace(0, 1, ry)[:, None]
    v = t2 * p1[None, :] + (1 - t2) * p2[None, :]
    
    t3 = np.linspace(0, 1, rz)[:, None]
    w = t3 * p4[None, :] + (1 - t3) * p2[None, :]
    
    
    # create the position matrix
    p = u[:,None,None,:] + (v - v[0])[None,:,None,:] + (w - w[0])[None,None,:,:]

    # convert to polar coordinate system
    rho = (p ** 2).sum(axis=-1) ** 0.5
    theta = np.arctan2(p[...,2], (p[...,0] ** 2 + p[...,1] ** 2) ** 0.5)
    phi = np.arctan2(p[...,1], p[...,0])
    
    loc = np.zeros(p.shape)

    # Convert rho,theta,phi to i,j,k by mapping points 
    loc[..., 0] = (rho - rho_range[0])/(rho_range[1] - rho_range[0]) * (img.shape[0] - 1)
    loc[..., 1] = (theta - theta_range[0])/(theta_range[1] - theta_range[0]) * (img.shape[1] - 1)
    loc[..., 2] = (phi - phi_range[0])/(phi_range[1] - phi_range[0]) * (img.shape[2] - 1)
    
    # initialize the return slicer
    slicer = np.zeros((rx, ry, rz))
    
    # create a mask that checks if the position is in the bounds
    mask = (loc[..., 0] >= 0) & (loc[..., 0] < img.shape[0]-1) & (
        loc[..., 1] >= 0) & (loc[..., 1] < img.shape[1]-1) & (
        loc[..., 2] >= 0) & (loc[..., 2] < img.shape[2]-1)
    
    # if no intersection points, return the whole black slicer
    if mask.sum() == 0:
        return slicer
    
    #### trilinear interpolation ####
    
    # find the eight points around the position points
    c1 = np.floor(loc[mask]).astype(int)
    c2 = c1 + np.array([0,0,1])[None,:]
    c3 = c1 + np.array([0,1,0])[None,:]
    c4 = c1 + np.array([0,1,1])[None,:]
    c5 = c1 + np.array([1,0,0])[None,:]
    c6 = c1 + np.array([1,0,1])[None,:]
    c7 = c1 + np.array([1,1,0])[None,:]
    c8 = c1 + np.array([1,1,1])[None,:]
    
    # the differences
    d = loc[mask] - c1
    
    # get the differences on x-axis
    x_d = d[:,0]
    
    # limit eight points to four points on differences on x-axis
    c00 = (1 - x_d) * img[c1[:, 0], c1[:, 1], c1[:, 2]] + x_d * img[c5[:, 0], c5[:, 1], c5[:, 2]]
    c01 = (1 - x_d) * img[c2[:, 0], c2[:, 1], c2[:, 2]] + x_d * img[c6[:, 0], c6[:, 1], c6[:, 2]]
    c10 = (1 - x_d) * img[c3[:, 0], c3[:, 1], c3[:, 2]] + x_d * img[c7[:, 0], c7[:, 1], c7[:, 2]]
    c11 = (1 - x_d) * img[c4[:, 0], c4[:, 1], c4[:, 2]] + x_d * img[c8[:, 0], c8[:, 1], c8[:, 2]]
    
    # get the differences on y-axis
    y_d = d[:,1]
    
    # limit four points to two points on differences on y-axis
    c0 = (1 - y_d) * c00 + y_d * c10
    c1 = (1 - y_d) * c01 + y_d * c11
    
    # get the differences on z-axis
    z_d = d[:,2]
    
    # limit two points to the final one points on differences on z-axis
    c = (1 - z_d) * c0 + z_d * c1

    # set the final pixel values to its positions in the initialized slicer
    slicer[mask] = c

    return slicer

def reshape_3d(img, frame, rho_range, theta_range, phi_range):
    r = np.mean(rho_range)
    t = np.mean(theta_range)
    p = np.mean(phi_range)
    center = [r*np.cos(t)*np.cos(p), r*np.cos(t)*np.sin(p), r*np.sin(t)]
    length = (rho_range[1] - rho_range[0])*1.5
    p1, p2, p3, p4 = extract_points(center,[1,0,0],[0,0,1],length,length,length)
    ret = []
    for f in tqdm(range(frame)):
        slicer = slice_area(img[...,f], p1, p2, p3, p4, rho_range, theta_range, phi_range, 128, 128, 128)
        ret.append(slicer)
    return np.stack(ret)

if __name__ == "__main__":
    
    df_1879 = pd.read_csv('cluster2_same_theta_phi_metadata.csv')

    list_1897 = list(df_1879['file_name'])

    df = pd.read_csv('data_3d_new.csv')

    reduced_df = df[df['file_uid'].isin(list_1897)]

    reduced_df = reduced_df[['file_uid', 'rho_min', 'rho_max', 'theta_min', 'theta_max', 'phi_min', 'phi_max']]

    for i in tqdm(range(len(reduced_df))):
        file = reduced_df['file_uid'].iloc[i]
        rho = [reduced_df['rho_min'].iloc[i], reduced_df['rho_max'].iloc[i]]
        theta = [reduced_df['theta_min'].iloc[i], reduced_df['theta_max'].iloc[i]]
        phi = [reduced_df['phi_min'].iloc[i], reduced_df['phi_max'].iloc[i]]
        img = np.load('/workspace/data/drives/Local_SSD/sda/x5_1897_npy/'+file+'.npy')
        if len(img.shape) == 4:
            data = reshape_3d(img, img.shape[-1], rho, theta, phi)
            np.save('/workspace/data/drives/Local_SSD/sda/x5_reshaped_1897_npy/'+file+'.npy', data)
