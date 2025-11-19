# Python Standard Libs
import copy
import os
from dataclasses import dataclass

# Third Party Libs
import numpy as np
import pandas as pd
import PIL
import scipy
import torch
import torchvision
import tqdm
import pydicom
import matplotlib.pyplot as plt
# Local Imports
import lv_segmenter
import view_classifier
from . import math_utils 
from .utils_3d import read_3d

@dataclass
class Plane:
    """phi and theta are in degrees"""
    name: str
    d: float
    phi: float
    theta: float
    
@dataclass
class ViewParams:
    """phi and theta are in degrees"""
    name: str
    d: float
    phi: float
    theta: float
    flipu: bool
    flipv: bool
    rotation_angle: float
    cm_pix: float
    h_cm: float
    w_cm: float

class ViewExtractor:
    def __init__(self,original_weights=True):
        self.device = torch.device("cpu")
        self.view_classifier = view_classifier.EchoViewClassifier(original_weights=original_weights).to(self.device)
        self.lv_segmentation_model = lv_segmenter.LVseg(device=self.device, original_weights=original_weights)
        self.view_classifier.eval()


    def extract_views(self, dicom_path, output_path):
        """
        Extract views from a dicom file
        """
        dcm = pydicom.dcmread(dicom_path)
        
        try:
            recommended_display_frame_rate = int(dcm[('0008','2144')].value)
        except:
            recommended_display_frame_rate = 21
        data, bounds = read_3d(dcm)
        views,videos,probs = self.get_videos(data,bounds)
        for idx, view in enumerate(views):
            view_name = view.name
            
            # write videos
            os.makedirs(output_path, exist_ok=True)
            torchvision.io.write_video(os.path.join(output_path, view_name+".mp4"),
                                    videos[idx],
                                    recommended_display_frame_rate)

            view.p = probs[idx]
            # save view to csv
            view_df = pd.DataFrame([view.__dict__])
            view_df.to_csv(os.path.join(output_path, view_name+".csv"), index=False)

        return "Files saved to "+output_path

    def get_videos(self, data, bounds):
        """
        Get these views from a 3D echo volume ["A2C", "A3C", "A4C", "A5C", "PLAX",  "PSAX_(level_of_apex)", "PSAX_(level_of_MV)", "PSAX_(level_of_papillary_muscles)"]
        Args:
            data: 4d volume (3D echo + time)
            bounds: 3x2 numpy array of shape (3,2) containing:
            [[rho_min, rho_max],
            [phi_min, phi_max], 
            [theta_min, theta_max]]
        Returns:
            views: list of T strings 
            videos: list of videos for each view
        """
        views=["A2C", "A3C", "A4C", "A5C", "PLAX", "PSAX_(level_of_apex)", "PSAX_(level_of_MV)", "PSAX_(level_of_papillary_muscles)"]
        
        rho_min, rho_max = bounds[0]
        phi_min, phi_max = bounds[1]
        theta_min, theta_max = bounds[2] 

        # Create linspace for rho, phi, and theta
        rho_coords = np.linspace(rho_min, rho_max, data.shape[0])
        phi_coords = np.linspace(phi_min, phi_max, data.shape[1])
        theta_coords = np.linspace(theta_min, theta_max, data.shape[2])
        positions = (rho_coords, phi_coords, theta_coords)
        
        data_at_first_time_point = data[:,:,:,0]
        position_cartesian, values_cartesian = math_utils.convert_to_cartesian(data_at_first_time_point,rho_coords,phi_coords,theta_coords)

        #make a dictionary of view params start from approximat positions
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), "view_planes.csv"))
        views = {row['Name']:ViewParams(row['Name'],
                                    row['d'],
                                    row['phi'],
                                    row['theta'],
                                    row['flipu'],
                                    row['flipv'],
                                    row['rotation_angle'],
                                    row['cm_pix'],
                                    row['h_cm'],
                                    row['w_cm']) 
                                    for idx, row in df.iterrows()} 
        #first find an a4c view
        views['A4C'], best_prob = self._find_a4c(views['A4C'],data,positions,position_cartesian)
        #get A4C image
        spherical_grid, (width,height), (width_px,height_px) = self.slice_volume(views['A4C'], position_cartesian)
        a4c_image = math_utils.interpolate(positions,data,spherical_grid,height_px,width_px, first_frame=True)

        L_LV, P_A4C, P_SAX, P_LA = self._find_landmarks(views,a4c_image)
        videos=[]
        probs=[]
        for v in views.keys():
            phi_search, theta_search, d_search = self._define_search_range(views[v],L_LV,P_A4C,P_SAX,P_LA,views)
            views[v] = self._guided_search(views[v],
                                                phi_search = phi_search,
                                                theta_search = theta_search,
                                                d_search = d_search,
                                                data = data,
                                                positions = positions,
                                                position_cartesian = position_cartesian)
            spherical_grid, (width,height), (width_px,height_px) = self.slice_volume(views[v], position_cartesian, scaling = v.startswith("PSAX"))
            video = math_utils.interpolate(positions,data,spherical_grid,height_px,width_px)
            videos.append(video)
            probs.append(self._view_probs(video[0],v))

        return list(views.values()), videos, probs



    def _find_a4c(self,view:ViewParams,
                                 data,
                                 positions,
                                 position_cartesian,
                                 open_interval_search=True):
        # for reproducibility
        np.random.seed(42) 

        # get current view/slice
        spherical_grid, (width,height), (width_px,height_px) = self.slice_volume(view, position_cartesian)

        #interpolate the first frame only
        image = math_utils.interpolate(positions,data,spherical_grid,height_px,width_px, first_frame=True)
        best_prob = self._view_probs(image,view.name)
        best_view=view

        num_it=100
        offset=10

        for i in tqdm.tqdm(range(num_it)):
            if best_prob>0.99:
                break
            perturbed_view = copy.deepcopy(view)
            perturbed_view.d = view.d + np.random.uniform(-offset,offset)
            perturbed_view.phi = view.phi + np.random.uniform(-offset,offset)
            perturbed_view.theta = view.theta + np.random.uniform(-offset,offset)
            spherical_grid, (width, height), (width_px,height_px) = self.slice_volume(perturbed_view, position_cartesian)
            image = math_utils.interpolate(positions,data,spherical_grid,height_px,width_px, first_frame=True)
            prob = self._view_probs(image,view.name)
            if prob>best_prob:
                best_prob=prob
                best_view = perturbed_view
                if open_interval_search:
                    view = best_view
        return best_view, best_prob

    def _find_landmarks(self,views,a4c_image):
        """
        Find the landmarks of the LV in the A4C image
        """
        #find the apex and base of the LV
        apex,base = self.lv_segmentation_model.get_base_apex_line(torch.tensor(a4c_image).unsqueeze(0).float().permute(0,3,1,2))
        H = a4c_image.shape[0]
        W = a4c_image.shape[1]
        probe_location = (W//2,0) 
        mm_per_pixel=views['A4C'].cm_pix * 10
        # world coordinates go left to right, but image coordinates go right to left that's why we add - on z
        apex_point = ( (apex[1] - probe_location[1]) * mm_per_pixel,0,-(apex[0]-probe_location[0])*mm_per_pixel)
        base_point = ( (base[1] - probe_location[1]) * mm_per_pixel,0,-(base[0]-probe_location[0])*mm_per_pixel)

        n_SAX = np.array(base_point)-np.array(apex_point)
        n_SAX = n_SAX/np.linalg.norm(n_SAX)

        n_A4C = math_utils.normal_from_phitheta(math_utils.degree_to_radian(views['A4C'].phi),math_utils.degree_to_radian(views['A4C'].theta))
        n_A4C = n_A4C/np.linalg.norm(n_A4C)
        n_LA = np.cross(n_SAX,n_A4C)
        n_LA /= np.linalg.norm(n_LA)

        # left ventricle length corresponding to apex to base distance
        L_LV = np.sqrt((apex_point[0]-base_point[0])**2 + (apex_point[1]-base_point[1])**2 + (apex_point[2]-base_point[2])**2)
        # offset along nSAX before LV begins
        d_SAX = math_utils.scalar_from_point(apex_point,n_SAX)
        d_LA = math_utils.scalar_from_point(apex_point,n_LA)

        # convert normal planes to angles
        phi_SAX,theta_SAX = math_utils.phitheta_from_normal(n_SAX)
        phi_SAX = math_utils.radian_to_degree(phi_SAX)
        theta_SAX = math_utils.radian_to_degree(theta_SAX)
        
        phi_LA,theta_LA = math_utils.phitheta_from_normal(n_LA)
        phi_LA = math_utils.radian_to_degree(phi_LA)
        theta_LA = math_utils.radian_to_degree(theta_LA)
        
        P_A4C = Plane(name="A4C",d=views['A4C'].d,phi=views['A4C'].phi,theta=views['A4C'].theta)
        P_SAX = Plane(name="PSAX",d=d_SAX,phi=phi_SAX,theta=theta_SAX)
        P_LA = Plane(name="LA",d=d_LA,phi=phi_LA,theta=theta_LA)
    
        return L_LV, P_A4C, P_SAX, P_LA

    def _define_search_range(self,view:ViewParams,L_LV,P_A4C,P_SAX,P_LA,views):
        """
        Define the search range for the view
        """
        if view.name=='A2C':
            d_search = (P_LA.d,P_LA.d)
            phi_search = (P_LA.phi,P_LA.phi)
            theta_search = (P_LA.theta,P_LA.theta+30)
        elif view.name=='A3C':
            d_search = (P_LA.d,P_LA.d)
            phi_search = (P_LA.phi,P_LA.phi)
            theta_search = (views['A2C'].theta-60, views['A2C'].theta-15)
        elif view.name=='A4C':
            d_search = (views['A4C'].d,views['A4C'].d)
            phi_search = (P_A4C.phi,P_A4C.phi)
            theta_search = (P_A4C.theta,P_A4C.theta)
        elif view.name=='A5C':
            d_search = (views['A4C'].d,views['A4C'].d)
            phi_search = (P_A4C.phi,P_A4C.phi)
            theta_search = (P_A4C.theta+10,P_A4C.theta+35)
        elif view.name=='PSAX_(level_of_apex)':
            d_search = (P_SAX.d+0.10*L_LV,P_SAX.d+0.20*L_LV)
            phi_search = (P_SAX.phi,P_SAX.phi)
            theta_search = (P_SAX.theta,P_SAX.theta)
        elif view.name=='PSAX_(level_of_papillary_muscles)':
            d_search = (P_SAX.d+0.40*L_LV,P_SAX.d+0.50*L_LV)
            phi_search = (P_SAX.phi,P_SAX.phi)
            theta_search = (P_SAX.theta,P_SAX.theta)
        elif view.name=='PSAX_(level_of_MV)':
            d_search = (P_SAX.d+0.75*L_LV,P_SAX.d+0.80*L_LV)
            phi_search = (P_SAX.phi,P_SAX.phi)
            theta_search = (P_SAX.theta,P_SAX.theta)
        elif view.name=='PLAX':
            d_search = (P_LA.d,P_LA.d)
            phi_search = (views['A3C'].phi,views['A3C'].phi)
            theta_search = (views['A3C'].theta,views['A3C'].theta)
        else:
            raise ValueError(f"View {view} not supported")

        return phi_search, theta_search, d_search

    def _guided_search(self,
                            view:ViewParams,
                            phi_search,
                            theta_search,
                            d_search,
                            data,
                            positions,
                            position_cartesian,
                            open_interval_search=True):
        """
        Finds the ideal plane to cut the 3D echo to get the best view
        It searches from the given search ranges.
        """
        # scale it so that the view classifier is happy
        if view.name.startswith("PSAX"):
            scaling=True
        else: 
            scaling=False
        # get current view/slice
        spherical_grid, (width,height), (width_px,height_px) = self.slice_volume(view, position_cartesian, scaling=scaling)
        #interpolate the first frame only
        image = math_utils.interpolate(positions,data,spherical_grid,height_px,width_px, first_frame=True)
        best_prob = -1
        best_view = view
        best_image = image
        for phi in np.arange(phi_search[0], phi_search[1]+1e-6, 1):
            for theta in np.arange(theta_search[0], theta_search[1]+1e-6,1):
                for d in np.arange(d_search[0], d_search[1]+1e-6, 1):
                    perturbed_view = copy.deepcopy(view)
                    perturbed_view.d = d
                    perturbed_view.phi =  phi
                    perturbed_view.theta = theta

                    spherical_grid, (width, height), (width_px,height_px) = self.slice_volume(perturbed_view, position_cartesian, scaling=scaling)
                    image = math_utils.interpolate(positions,data,spherical_grid,height_px,width_px, first_frame=True)
                    prob = self._view_probs(image,view.name)

                    if prob>best_prob:
                        best_prob=prob
                        best_view = perturbed_view
                        best_image = image
                        if view.name.startswith("PSAX") and best_view.phi<0:
                            best_view.flipv = not best_view.flipv

        return best_view


    def _view_probs(self, image, view="A4C"):
        """
            View classifier checks what is the probability of the view 
            Args:
                image: 2d numpy array
                view: string
            Returns:
                probability of the view
        """
        # Replace nan with 0 and make 3 channels
        no_nan_image_slice = np.nan_to_num(image, nan=0)
        # Convert float array to uint8 range 0-255
        image_slice_uint8 = ((no_nan_image_slice - no_nan_image_slice.min()) * (255.0/(no_nan_image_slice.max() - no_nan_image_slice.min()))).astype(np.uint8)
        proc_img = np.array(PIL.Image.fromarray(image_slice_uint8).convert('RGB'))
        activations = self.view_classifier.activations(self.view_classifier.preprocess_single_image(proc_img).unsqueeze(0))
        if view not in activations and view.startswith("PSAX"):
            # for open-weights view classifier we don't have fine-grained psax level information
            return activations['PSAX']
        return activations[view] 


    def slice_volume(self,
                    view_params: ViewParams,
                    position_cartesian,
                    scaling=False):
        """
        Given a plane determined by P0 and n, cut the 3D volume and return the spherical coordinates that need to be interpolated
            Args:
                view_params: ViewParams object
                position_cartesian: all positions of the points in the volume
                next three determine the orientation of the plane
            Returns:
                r_grid: 2d numpy array of radial distances
                phi_grid: 2d numpy array of azimuthal angles
                theta_grid: 2d numpy array of polar angles
                width: width of the plane in mm 
                height: height of the plane in mm
        """
        # express them as a normal and a point.
        n = math_utils.normal_from_phitheta(math_utils.degree_to_radian(view_params.phi),math_utils.degree_to_radian(view_params.theta))
        P0 = math_utils.point_from_scalar(view_params.d,n)
        n/=np.linalg.norm(n)

        # get in plane rotation angle in radians
        rotation_angle = np.radians(view_params.rotation_angle)
        # get in plane flip
        flipu = view_params.flipu
        flipv = view_params.flipv

        # Find a vector not parallel to n
        a = np.array([1,0,0], dtype=float)
        if np.allclose(n, a) or np.allclose(n, -a):
            a = np.array([0,1,0], dtype=float)

        # Create orthonormal basis
        u = np.cross(n, a)
        u = u / np.linalg.norm(u)
        if flipu:
            u=-u
        v = np.cross(n, u)
        if flipv:
            v=-v

        # Apply in-plane rotation to u and v vectors
        cos_theta = np.cos(rotation_angle)
        sin_theta = np.sin(rotation_angle)
        u_rot = cos_theta * u + sin_theta * v
        v_rot = -sin_theta * u + cos_theta * v

        # position_cartesian: Nx3 array of data points
        r = position_cartesian - P0  # Shift points so plane point is origin of plane coords
        s_coords = np.dot(r, u_rot)  # Projection on rotated u-axis
        t_coords = np.dot(r, v_rot)  # Projection on rotated v-axis

        s_min, s_max = np.min(s_coords), np.max(s_coords)
        t_min, t_max = np.min(t_coords), np.max(t_coords)

        if scaling:
            # perform scale relative to ideal scale
            s_length = (view_params.w_cm * 10) / np.linalg.norm(u_rot)
            t_length = (view_params.h_cm * 10) / np.linalg.norm(v_rot)
            #scale 
            s_center = (s_min+s_max)/2
            t_center = (t_min+t_max)/2
            s_min = s_center - s_length/2
            s_max = s_center + s_length/2
            t_min = t_center - t_length/2
            t_max = t_center + t_length/2

        # height and width of the plane in cm
        #/10 is for converting to cm
        width = np.linalg.norm((s_max-s_min)*u_rot) / 10
        height = np.linalg.norm((t_max-t_min)*v_rot) / 10

        # Create a grid of s and t values 
        if view_params.cm_pix==0:
            num_points_x = 224
            num_points_y = 224
        else:
            num_points_x = int(width/view_params.cm_pix)
            num_points_y = int(height/view_params.cm_pix)
        s_values = np.linspace(s_min, s_max, num_points_x)
        t_values = np.linspace(t_min, t_max, num_points_y)
        S, T = np.meshgrid(s_values, t_values)
        
        # Convert to 3D 
        X = P0[0] + S*u_rot[0] + T*v_rot[0]
        Y = P0[1] + S*u_rot[1] + T*v_rot[1]
        Z = P0[2] + S*u_rot[2] + T*v_rot[2]
        grid_points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

        # Convert to spherical coordinates
        r_grid = np.sqrt(grid_points[:,0]**2 + grid_points[:,1]**2 + grid_points[:,2]**2)
        phi_grid = np.arctan2(grid_points[:,2],grid_points[:,0])
        theta_grid = np.arctan2(grid_points[:,1], (np.sqrt(grid_points[:,0]**2 + grid_points[:,2]**2)))
        
        return (r_grid,phi_grid,theta_grid), (width,height), (num_points_x,num_points_y)



