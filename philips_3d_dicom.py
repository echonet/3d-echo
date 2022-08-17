import numpy as np
import pandas as pd
import pydicom as dicom
import zlib
import struct
from tqdm import tqdm
from database_utils import DbConnection


def decompress(fragment: dicom.dataset):
    name = fragment[(0x200d, 0x300d)]
    data_frag = fragment[(0x200d, 0x3cf1)][0]
    compression = data_frag[(0x200d, 0x3cfa)].value
    raw_bts = data_frag[(0x200d, 0x3cf3)].value
    crc_array = data_frag[(0x200d, 0x3cfb)].value

    data_size, n_frames = struct.unpack('<II', raw_bts[:8])
    start_i = struct.unpack('<' + 'I' * n_frames, raw_bts[8:8 + 4 * n_frames])

    out = []
    if compression == 'ZLib':
        for i, si in enumerate(tqdm(start_i)):
            if crc_array[i * 32: (i + 1) * 32] != raw_bts[si:si + 32]:
                raise Exception('CRC did not match')
            out.append(zlib.decompress(raw_bts[si + 32:]))
    else:
        raise Exception('Non ZLib not implemented')

    return out


def read_ecg(fragment: dicom.dataset):
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
    n = np.prod(stride)
    return np.array([
        np.frombuffer(x, dtype=np.uint8)[:n].reshape(stride) for x in data
    ])[:, :shape[0], :shape[1], :shape[2]].T


def get_array_shape(dcm):
    arr_15 = np.array([int(x) for x in dcm[(0x200d, 0x3315)].value])
    arr_16 = np.array([int(x) for x in dcm[(0x200d, 0x3316)].value])
    arr_15 = arr_15 ^ arr_15[-1]
    arr_16 = arr_16 ^ arr_16[-1]
    return tuple(arr_16[2::-1]), tuple(arr_15[2::-1])


def get_frustum(dcm: dicom.FileDataset):
    tags = [0x3102, 0x3103, 0x3104, 0x3105, 0x310d, 0x310e]
    arr = np.array([dcm[(0x200d, k)].value for k in tags]).reshape((3, 2))
    return arr


def read_3d(dcm: dicom.FileDataset):
    stride, shape = get_array_shape(dcm)
    fragment = dcm[(0x200d, 0x3cf5)][1]
    data = decompress(fragment)
    data = reshape_arrays(data, stride, shape)
    bounds = get_frustum(dcm)
    return data, bounds


sql = """SELECT * FROM patients
    LEFT JOIN studies ON patients.patient_id=studies.patient_id
    LEFT JOIN files ON studies.study_id=files.study_id
    """


if __name__ == '__main__':
    from pathlib import Path
    from tqdm import tqdm

    import time
    t0 = time.time()
    db = DbConnection()
    df = db.fetch_sql_to_df(sql)
    db.disconnect()
    t0 = time.time() - t0
    print(t0)
    print(df)

    file_ids = []
    rho_min = []
    rho_max = []
    theta_min = []
    theta_max = []
    phi_min = []
    phi_max = []
    data_stride = []
    data_shape = []
    data_shape_w_frame = []
    pixel_shape = []

    n_3d = 0
    for i, row in tqdm(df.iloc[2900:].iterrows(), total=len(df.iloc[2900:])):
        try:
            study, series, file = row[['study_uid', 'series_uid', 'file_uid']]
            p = Path(f'/workspace/data/NAS2/DVTk/Data/{study}/{series}/{file}/{file}.dcm')
            if not p.exists():
                continue
            dcm = dicom.dcmread(p, stop_before_pixels=True)

            try:
                i, d, s, v = read_ecg(dcm[(0x200d, 0x3cf5)][0])
                print(f'ECG Found: v={v.shape}')
            except KeyError as e:
                pass

            try:
                bounds = get_frustum(dcm)
                stride, shape = get_array_shape(dcm)
                data = reshape_arrays(decompress(dcm[(0x200d, 0x3cf5)][1]), stride, shape)
            except KeyError as e:
                # print('Missing tag:', e)
                continue
            except Exception as e:
                print('Fail to read 3d data')
                print(e)
                continue
        except Exception as e:
            print('failed for unknown reason')
            print(e)
            continue

        print('Found 3D:', n_3d, file)
        print(data.shape)
        print(bounds)
        print('---------')
        n_3d += 1

        np.save(f'/workspace/data/NAS2/DVTk/Data/{study}/{series}/{file}/{file}.npy', data)

        file_ids.append(file)
        rho_min.append(float(bounds[0][0]))
        rho_max.append(float(bounds[0][1]))
        theta_min.append(float(bounds[1][0]))
        theta_max.append(float(bounds[1][1]))
        phi_min.append(float(bounds[2][0]))
        phi_max.append(float(bounds[2][1]))
        data_stride.append(str(stride[::-1]))
        data_shape.append(str(shape[::-1]))
        data_shape_w_frame.append(str(data.shape))
        dcm = dicom.dcmread(p)
        pixel_shape.append(str(dcm.pixel_array.shape))
    
    data_3d = pd.DataFrame({'file_uid': file_ids,
                            'rho_min': rho_min,
                            'rho_max': rho_max,
                            'theta_min': theta_min,
                            'theta_max': theta_max,
                            'phi_min': phi_min,
                            'phi_max': phi_max,
                            'data_stride': data_stride,
                            'data_shape': data_shape,
                            'data_shape_with_frame': data_shape_w_frame,
                            'pixel_shape': pixel_shape
                            })

    data_3d = data_3d.merge(df, left_on='file_uid', right_on='file_uid', how='inner')
    data_3d.to_csv('data_3d_new.csv', index=False, na_rep='Unknown')
