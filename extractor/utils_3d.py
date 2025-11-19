"""
Utils for extracting 3D echocardiogram data from DICOM files.

This module provides utility functions for reading and processing 3D echocardiogram data stored in DICOM format.
The main functionality includes:

- Decompressing raw DICOM data fragments using ZLib compression
- Reading ECG data from DICOM fragments 
- Reshaping raw data arrays into proper 3D volumes
- Extracting array shapes and frustum bounds from DICOM metadata
- Reading full 3D volumes from DICOM files

The main entry point is the read_3d() function which takes a DICOM FileDataset object and returns:
1. The 3D volume data as a numpy array
2. The bounds/dimensions of the frustum containing the data

Example usage:
    import pydicom
    from utils_3d import read_3d
    
    dcm = pydicom.dcmread("echo.dcm")
    data, bounds = read_3d(dcm)

Dependencies:
    - numpy
    - pandas  
    - pydicom
    - zlib
    - struct
    - tqdm

Tag definitions according to https://www.documents.philips.com/doclib/enc/fetch/2000/4504/577242/577256/588723/5144873/5144488/5144888/DICOM_Conformance_Statement_iE33_R1.0.0.x.pdf
"""

import numpy as np
import pandas as pd
import pydicom as dicom
import zlib
import struct
from tqdm import tqdm


def decompress(dcm: dicom.FileDataset):
    """
    Given a dicom file containin 3D volume, extract and decompress bytes
    Args:
        dcm: dicom.FileDataset
    Return:
        out: list of byte objects, where each byte object corresponds to a single frame.
    """
    # read a field where most info about 3D data is stored
    fragment = dcm[(0x200d, 0x3cf5)][1]
    name = fragment[(0x200d, 0x300d)]
    data_frag = fragment[(0x200d, 0x3cf1)][0]
    
    compression = data_frag[(0x200d, 0x3cfa)].value
    raw_bts = data_frag[(0x200d, 0x3cf3)].value
    crc_array = data_frag[(0x200d, 0x3cfb)].value

    # reads first 8 bytes and unpacks two little endian (<) type integers each 4 bytes long
    data_size, n_frames = struct.unpack('<II', raw_bts[:8])

    #reads n_frames integers, each element in start_i array represent byte offset where byte data for frame i starts.
    start_i = struct.unpack('<' + 'I' * n_frames, raw_bts[8:8 + 4 * n_frames])

    out = []
    if compression == 'ZLib':
        for i, si in enumerate(tqdm(start_i)):
            # check that crc matches and bytes are ok.
            # first 32 bytes of raw_bts are dedicated for CRC checksum
            if crc_array[i * 32: (i + 1) * 32] != raw_bts[si:si + 32]:
                raise Exception('CRC did not match')
            
            # no need for ending position (32:) because zlib stops when it detects end of frame.
            out.append(zlib.decompress(raw_bts[si + 32:]))
    else:
        raise Exception('Non ZLib not implemented')

    return out


def read_ecg(dcm: dicom.FileDataset):
    # read a field where most info about 3D data is stored
    fragment = dcm[(0x200d, 0x3cf5)][1]

    data_frag = fragment[(0x200d, 0x3cf1)][0]
    raw_bts = data_frag[(0x200d, 0x3cf3)].value

    data_size, n_frames = struct.unpack('<II', raw_bts[:8])
    start_i = struct.unpack('<' + 'I' * n_frames, raw_bts[8:8 + 4 * n_frames])

    out = []
    for i, (si, ei) in enumerate(tqdm(zip(start_i[:-1], start_i[1:]))):
        bts = raw_bts[si:ei]
        idx = struct.unpack('H', bts[:2])[0]
        d, s = bts[14], bts[15]
        v = struct.unpack('h', bts[32:34])[0]
        out.append((idx, d, s, v))

    i = np.array([x[0] for x in out])
    d, s = np.array([[x[1], x[2]] for x in out]).T
    v = np.array([x[3] for x in out])

    return i, d, s, v


def reshape_arrays(data, stride, shape):
    """
    Given raw bytes gets an array representing pixel intensities of a 3D object at each timestep
    Args:
        data: list of byte objects, where each byte object corresponds to a single frame.
        stride: (θ, φ, ρ)
        shape: (θ, φ, ρ)
    Return:
        4D array (ρ, φ, θ, t) that specifies pixel intensities
    """
    # just take a product to get the number of total entries
    n = np.prod(stride)

    # read-friendly version
    # for each timestep
    # out = []
    # for x in data:
    #     # read a numpy array uint8 from bytes and limit it to total stride shape
    #     x = np.frombuffer(x,dtype=np.uint8)[:n]
    #     x = x.reshape(stride)
    #     x = x[:, :shape[0], :shape[1], :shape[2]].T
    # return np.array(out)


    # fast version
    return np.array([
        np.frombuffer(x, dtype=np.uint8)[:n].reshape(stride) for x in data
    ])[:, :shape[0], :shape[1], :shape[2]].T


def get_array_shape(dcm: dicom.FileDataset):
    """
    Given a dicom file that contains a 3D volume get the shape of the matrix of pixel intesities
    and a stride matrix telling you how to read bytes
    Args:
        dcm: dicom.FileDataset
    Returns:
        stride (θ, φ, ρ) e.g. (48, 64, 416)
        shape  (θ, φ, ρ) e.g. (48, 64, 404)
    Note: we return it as (θ, φ, ρ) instead of (ρ, θ, φ) for the decoding purposes later
    """
    #private field containing 4 integers
    arr_15 = np.array([int(x) for x in dcm[(0x200d, 0x3315)].value])
    #private field containing 4 integers
    arr_16 = np.array([int(x) for x in dcm[(0x200d, 0x3316)].value])
    # arr_15 can for example be [404,64,48]
    arr_15 = arr_15 ^ arr_15[-1]
    arr_16 = arr_16 ^ arr_16[-1]
    #just reverse the order
    stride = tuple(arr_16[2::-1])
    shape = tuple(arr_15[2::-1])
    #TODO: Figure out if we need a stride at all, and just need shape
    return stride,shape 


def get_frustum(dcm: dicom.FileDataset):
    """
    s
    """
    # (rho_min, rho_max), (phi_min, phi_max), (theta_min, theta_max)
    tags = [0x3102, 0x3103, 0x3104, 0x3105, 0x3203, 0x3204]
    arr = np.array([float(dcm[(0x200d, k)].value) for k in tags]).reshape((3, 2))
    return arr


def read_3d(dcm: dicom.FileDataset):
    data = decompress(dcm)

    stride, shape = get_array_shape(dcm)
    data = reshape_arrays(data, stride, shape)

    bounds = get_frustum(dcm)

    return data, bounds