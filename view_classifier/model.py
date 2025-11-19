import torch 
import torch.nn.functional as F
import torchvision
import os 
class EchoViewClassifier(torch.nn.Module):
    def __init__(self, original_weights=False):
        super(EchoViewClassifier, self).__init__()
        self.model = torchvision.models.convnext_base()
        self.original_weights = original_weights
        if original_weights:
            self.model.classifier[-1] = torch.nn.Linear(self.model.classifier[-1].in_features, 11)
            self.model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "view_classifier.pt"), map_location=torch.device('cpu')))
            self.views=REDUCED_VIEWS
        else:
            self.model.classifier[-1] = torch.nn.Linear(self.model.classifier[-1].in_features, 60)
            self.model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "full_view_classifier.pth"), map_location=torch.device('cpu')))
            self.views = ALL_VIEWS
        self.model.eval()


    def forward(self, x):
        x = self.model(x)
        return x
    
    def preprocess_single_image(self, image):
        """
        Args: image (np.ndarray): image to preprocess 0-255 uint8 of shape (height, width, channels)
        Returns:
            x (torch.Tensor): preprocessed image with shape (3, height, width)
        """
        # make it channels, height, width
        image = image.transpose(2, 0, 1)
        video_size = 224
        mean=torch.tensor([29.110628, 28.076836, 29.096405], dtype=torch.float32)
        std=torch.tensor([47.989223, 46.456997, 47.20083], dtype=torch.float32)
        mean = mean.reshape(3, 1, 1)
        std =  std.reshape(3, 1, 1)
        x = torch.as_tensor(image, dtype=torch.float)
        x.sub_(mean).div_(std)
        if x.shape[2] != video_size:
            x = torch.nn.functional.interpolate(
                x.unsqueeze(0),
                size=(
                    video_size,
                    video_size,
                ),  # specify target height and width
                mode="bilinear",  # use bilinear interpolation for 2D images
                align_corners=False,
            ).squeeze(0)
        return x
    
    def activations(self, x):
        """
        returns a dictionary of softmaxed activations
        """
        # Get predictions
        yhat = self.model(x)
        # Apply softmax
        softmax = torch.nn.Softmax(dim=1)
        activations = softmax(yhat).squeeze()
        activations_dict={view: activation.item() for view, activation in zip(self.views, activations)}

        if not self.original_weights:
            # Merge all PLAX views into one
            plax_views = [
                "PLAX",
                "PLAX_AV_MV", 
                "PLAX_Zoom_out",
                "PLAX_Proximal_Ascending_Aorta",
                "PLAX_zoomed_AV",
                "PLAX_zoomed_MV"
            ]
            plax_prob = sum(activations_dict[view] for view in plax_views if view in activations_dict)
            # Remove individual PLAX views and add merged probability
            for view in plax_views:
                activations_dict.pop(view, None)
            activations_dict["PLAX"] = plax_prob
            # Merge all PSAX level great vessels views into one
            psax_views = [
                "PSAX_(level_great_vessels)",
                "PSAX_(level_great_vessels)_focus_on_PV_and_PA",
                "PSAX_(level_great_vessels)_focus_on_TV", 
                "PSAX_(level_great_vessels)_zoomed_AV",
            ]
            psax_prob = sum(activations_dict[view] for view in psax_views if view in activations_dict)
            # Remove individual PSAX views and add merged probability
            for view in psax_views:
                activations_dict.pop(view, None)
            activations_dict["PSAX_(level_great_vessels)"] = psax_prob
    
 
        return activations_dict

    def test(self):
        """ Test that everything is working"""
        working=True
        for v in ['A2C','A3C','A4C','A5C','PLAX','PSAX']:
            vid,_,__ = torchvision.io.read_video(f"view_classifier/sample_views/{v}.mp4",pts_unit='sec')
            pre_img = self.preprocess_single_image(vid[0].numpy()).unsqueeze(0)
            pred = max(self.activations(pre_img).items(), key=lambda x: x[1])[0]
            if pred!=v:
                working=False
                break
        return working

ALL_VIEWS=["A2C","A2C_LV","A3C","A3C_LV","A4C","A4C_LA","A4C_LV","A4C_MV","A4C_RV","A5C","PLAX","PLAX_AV_MV","PLAX_Zoom_out","PLAX_Proximal_Ascending_Aorta","PLAX_RV_inflow", "PLAX_RV_outflow", "PLAX_zoomed_AV","PLAX_zoomed_MV","PSAX_(level_great_vessels)", "PSAX_(level_great_vessels)_focus_on_PV_and_PA","PSAX_(level_great_vessels)_focus_on_TV","PSAX_(level_great_vessels)_zoomed_AV","PSAX_(level_of_MV)","PSAX_(level_of_apex)","PSAX_(level_of_papillary_muscles)","SSN_aortic_arch","Subcostal_4C","Subcostal_Abdominal_Aorta","Subcostal_IVC","DOPPLER_PSAX_level_great_vessels_TV","DOPPLER_PSAX_level_great_vessels_PA","DOPPLER_PSAX_level_great_vessels_AV","DOPPLER_PLAX_AV_zoomed","DOPPLER_PLAX_MV_zoomed","DOPPLER_PLAX_AV_MV","DOPPLER_PLAX_Ascending_Aorta","DOPPLER_PLAX_IVS","DOPPLER_PLAX_RVOT","DOPPLER_PLAX_RVIT","DOPPLER_A4C_MV_TV","DOPPLER_PSAX_MV","DOPPLER_A4C_MV","DOPPLER_A4C_TV","DOPPLER_A4C_Apex","DOPPLER_A4C_IVS","DOPPLER_A4C_IAS","DOPPLER_A4C_IVS_IAS","DOPPLER_A2C","DOPPLER_PSAX_IAS","DOPPLER_PSAX_IVS","DOPPLER_A5C","DOPPLER_A3C","DOPPLER_A3C_MV","DOPPLER_A3C_AV","DOPPLER_SSN_Aortic_Arch","DOPPLER_A4C_Pulvns","DOPPLER_SC_4C_IAS","DOPPLER_SC_4C_IVS","DOPPLER_SC_IVC","DOPPLER_SC_aorta"]
REDUCED_VIEWS=['A2C','A3C','A4C','A5C','Apical_Doppler','Doppler_PLAX','Doppler_PSAX','PLAX','PSAX','SSN','Subcostal']