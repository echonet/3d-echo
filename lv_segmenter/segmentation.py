import numpy as np
import torch
import torchvision
import os
import sklearn
import sklearn.decomposition
class LVseg():

    """
    User must override self.load_model(), self.load_data(), self.post_process(), name
    """

    def __init__(self,
                 device=torch.device("cpu"),
                 original_weights=False):

        self.device = device
        self.original_weights = original_weights

        if original_weights:
            checkpoint = torch.load(os.path.join(os.path.dirname(__file__), "deeplabv3_resnet50_random.pt"), map_location=device)
            self.mean = np.array([26.845592, 27.333233, 27.760092], dtype=np.float32)
            self.std = np.array([42.89915 , 43.217285, 43.87006 ], dtype=np.float32)
            self.model = torchvision.models.segmentation.deeplabv3_resnet50(weights=None, num_classes=1)
        else:
            checkpoint = torch.load(os.path.join(os.path.dirname(__file__), "deeplabv3_resnet50_pretrained_lr_1e-6_wd_0.pt"), map_location=device)
            self.mean = checkpoint["mean"]
            self.std = checkpoint["std"]
            self.model = torchvision.models.segmentation.deeplabv3_resnet50(weights=None, num_classes=3)
        
        if device.type == "cuda":
            self.model = torch.nn.DataParallel(self.model)
            self.model = self.model.to(device)
        state_dict = checkpoint["state_dict"]
        #state_dict = {key.split: state_dict[key] for key in state_dict if key[7:22] != "aux_classifier."}
        state_dict = {".".join(key.split(".")[1:]): state_dict[key] for key in state_dict if key[7:22] != "aux_classifier."}

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def __call__(self, x):
        return self.model(x)["out"]
    
    def get_LV_masks(self, frames):
        output_res =  tuple(list(frames.shape[2:]))

        # (F, 3, 112, 112)
        frames = torchvision.transforms.functional.resize(frames, (112, 112), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)

        # (F, 112, 112, 3)
        frames = frames.permute((0, 2, 3, 1))
        frames = (frames - self.mean) / self.std

        # (F, 3, 112, 112)
        frames = frames.permute((0, 3, 1, 2))

        # (F, 112, 112)
        lv_masks = self.__call__(frames)[:,0,:,:] > 0

        # (F, H, W)
        lv_masks = torchvision.transforms.functional.resize(lv_masks, output_res, interpolation=torchvision.transforms.InterpolationMode.NEAREST_EXACT) 
        lv_masks = lv_masks.cpu().numpy()
        lv_masks = np.where(lv_masks > 0, 1, 0).astype(np.uint8)

        return lv_masks

    def get_base_apex_line(self,
                           frames,
                           visualize=False):
        """
        Returns apex coordinates and base coordinates
        """
        org_frames=frames.clone()
        if self.original_weights:
            lv_masks = self.get_LV_masks(frames)
            mask1 = lv_masks[0]
            i, j = mask1.nonzero()
            i_mean = i.mean()
            j_mean = j.mean()
            i = i - i_mean
            j = j - j_mean
            pca = sklearn.decomposition.PCA(1)
            pca.fit(np.stack((i, j), axis=1))
            [(di, dj)] = pca.components_
            apex_x = dj * i.min() / di + j_mean
            apex_y = i.min() + i_mean
            base_x = dj * i.max() / di + j_mean
            base_y = i.max() + i_mean
        else:
            output_res =  tuple(list(frames.shape[2:]))
            mean = self.mean
            std = self.std 

            # (F, 3, 112, 112)
            frames = torchvision.transforms.functional.resize(frames, (112, 112), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)

            # (F, 112, 112, 3)
            frames = frames.permute((0, 2, 3, 1))
            frames = (frames - mean) / std

            # (F, 3, 112, 112)
            frames = frames.permute((0, 3, 1, 2))

            # (F, 112, 112)
            out = self.__call__(frames)
            mask = out[:,0]
            apex_mask = out[:,1]
            base_mask = out[:,2]

            apex_mask = torchvision.transforms.functional.resize(apex_mask, output_res, interpolation=torchvision.transforms.InterpolationMode.NEAREST_EXACT).detach().cpu()
            base_mask = torchvision.transforms.functional.resize(base_mask, output_res, interpolation=torchvision.transforms.InterpolationMode.NEAREST_EXACT).detach().cpu()
            
            softmax = torch.nn.Softmax(dim=0)
            #first frame softmaxed
            apex_mask_ed = softmax(apex_mask[0].reshape(-1)).reshape(output_res)
            base_mask_ed = softmax(base_mask[0].reshape(-1)).reshape(output_res)
            #get the mean of the softmaxed masks
            apex_x = (apex_mask_ed.sum(0).detach().cpu().numpy() * np.arange(output_res[1])).sum()
            apex_y = (apex_mask_ed.sum(1).detach().cpu().numpy() * np.arange(output_res[0])).sum()

            base_x = (base_mask_ed.sum(0).detach().cpu().numpy() * np.arange(output_res[1])).sum()
            base_y = (base_mask_ed.sum(1).detach().cpu().numpy() * np.arange(output_res[0])).sum()

        if visualize:
            import matplotlib.pyplot as plt
            plt.imshow( org_frames[0].permute(1,2,0).detach().cpu().numpy())
            #draw apex_x apex_y
            plt.scatter(apex_x,apex_y,color="red")
            #draw base_x base_y
            plt.scatter(base_x,base_y,color="blue")
            plt.savefig("test.png")

        return (apex_x,apex_y),(base_x,base_y)

    def test(self):
        vid= torchvision.io.read_video("/workspace/milos/EchoSlicer/view_classifier/sample_views/A4C.mp4",pts_unit='sec',output_format="TCHW")[0]
        out = self.get_base_apex_line(vid,visualize=True)
        if abs(out[0][0] - 113.2) < 1 and abs(out[0][1] - 37.0) < 1 and abs(out[1][0] - 126.3) < 1 and abs(out[1][1] - 148.1) < 1:
            return True
        else:
            return False
