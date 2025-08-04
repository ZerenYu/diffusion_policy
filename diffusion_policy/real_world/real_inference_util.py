from typing import Dict, Callable, Tuple
import numpy as np
from diffusion_policy.common.cv2_util import get_image_transform

def get_real_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        ) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    obs_shape_meta = shape_meta['obs']
    print(f"[zyu] obs_shape_meta is {obs_shape_meta}")
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            try:
                this_imgs_in = env_obs[key]
            except:
                this_imgs_in = env_obs[f'camera_0']
                print(f"[zyu] key is {key} use camera_0")
            t,hi,wi,ci = this_imgs_in.shape
            co,ho,wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi,hi), 
                    output_res=(wo,ho), 
                    bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            print(f"[zyu] out_imgs is {out_imgs.shape}")
            obs_dict_np[key] = np.moveaxis(out_imgs,-1,1)
        elif type == 'low_dim':
            try:
                this_data_in = env_obs[key]
            except:
                this_data_in = env_obs[f'robot_eef_pose'][:,:2]
                print(f"[zyu] shape of this data in is {this_data_in.shape} {this_data_in}")
            if 'pose' in key and shape == (2,):
                # take X,Y coordinates
                this_data_in = this_data_in[...,[0,1]]
                print(f"[zyu] this_data_in is {this_data_in}")
            obs_dict_np[key] = this_data_in
    return obs_dict_np


def get_real_obs_resolution(
        shape_meta: dict
        ) -> Tuple[int, int]:
    out_res = None
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            co,ho,wo = shape
            if out_res is None:
                out_res = (wo, ho)
            assert out_res == (wo, ho)
    return out_res
