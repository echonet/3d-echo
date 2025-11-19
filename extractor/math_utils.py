import numpy as np
import scipy
import PIL
def convert_to_cartesian(data,rho,phi,theta):
    """
    Given voxel values as a grid and linspaces of its spherical space
    Return the cartesian coordinates of the data and the values
    Args:
            data: 3d volume at a single point in time
            rho: linspace of radial distances
            phi: linspace of azimuthal angles
            theta: linspace of elevation angles 
    Returns:
        position_cartesian: 2d numpy array of shape (N,3) containing the cartesian coordinates of the data
        values: 1d numpy array of shape (N,) containing the values of the data
    """


    # Create meshgrid
    R, P, T = np.meshgrid(rho, phi, theta, indexing='ij')
    position_spherical = np.stack([R, P, T], axis=-1).reshape(-1, 3)

    # Convert spherical to Cartesian coordinates (vectorized)
    X = R * np.cos(T) * np.cos(P)
    Y = R * np.sin(T)
    Z = R * np.cos(T) * np.sin(P)

    # Flatten the Cartesian coordinates and data
    position_cartesian = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    x_min,x_max=np.min(X),np.max(X)
    y_min,y_max=np.min(Y),np.max(Y)
    z_min,z_max=np.min(Z),np.max(Z)
    values = data.ravel()
    return position_cartesian,values

def interpolate(positions,values,spherical_grid,height_px,width_px,first_frame=False):
    """
    Interpolate the values on the spherical grid to a 2D grid
    """
    if first_frame:
        values = values[...,0:1]
    interpolator = scipy.interpolate.RegularGridInterpolator(
    positions, values, bounds_error=False, fill_value=0
    )
    interp_values = interpolator(spherical_grid)
    interp_values_2D = interp_values.reshape((height_px,width_px,values.shape[3])).transpose(2,0,1)
    frames_min = interp_values_2D.min(axis=(1,2), keepdims=True)
    frames_max = interp_values_2D.max(axis=(1,2), keepdims=True)
    frames_uint8 = ((interp_values_2D - frames_min) * (255.0/(frames_max - frames_min))).astype(np.uint8)
    video = np.stack([PIL.Image.fromarray(frame).convert('RGB') for frame in frames_uint8])

    # for the purposes of writing to mp4 height and width cannot be divisible by 2
    # if that's the case, remove the last row and column
    if video.shape[1]%2==1:
        video = video[:,:-1,:,:]
    if video.shape[2]%2==1:
        video = video[:,:,:-1,:]
    # sqeueeze if it's an image
    if first_frame:
        video = video.squeeze(0)
    return video
def normal_from_phitheta(phi, theta):
    """
    Given phi and theta, return the normal to the plane
    """
    nx = np.cos(theta) * np.cos(phi)
    ny = np.sin(theta)
    nz = np.cos(theta) * np.sin(phi)
    return np.array([nx, ny, nz])

def phitheta_from_normal(n):
    """
    Invert n = (cos(theta)*cos(phi), sin(theta), cos(theta)*sin(phi))
    to get (theta, phi).
    Assumes n is normalized and that theta in [-pi/2, pi/2].
    """
    nx, ny, nz = n
    phi = np.arctan2(nz, nx)
    theta = np.arcsin(ny)
    return phi, theta

def scalar_from_point(P0,n):
    """
    Given a point P0 (px,py,pz)
    return a projection on the normal vector n = (nx,ny,nz)
    """
    return np.dot(P0,n)/np.linalg.norm(n)

def point_from_scalar(d,n):
    """
    Given a scalar d and a normal vector n, return a point on the normal vector
    """
    return d*n

def radian_to_degree(radian):
    return radian * 180 / np.pi

def degree_to_radian(degree):
    return degree * np.pi / 180
