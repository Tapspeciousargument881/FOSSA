
from util.camera import ThinLenCamera
from util.render import PowerExpPSF
import torch

def get_depth_dependent_fd_list(depth, N):
   assert (depth > 0).all(), "Depth map contains zero or negative values, cannot compute disparity."
   disparity = 1/depth
   B, C, H, W = disparity.shape
   disparity_flat = disparity.view(B, -1) 

   min_disparity = torch.quantile(disparity_flat, 0.05, dim=1)
   max_disparity = torch.quantile(disparity_flat, 0.95, dim=1)

   # Calculate min and max disparity stats

   steps = torch.linspace(0, 1, steps=N, device=min_disparity.device)  # shape (N,)
   # min_disparity[:, None]: shape (B, 1)
   # (1 - steps)[None, :]: shape (1, N) -- multiplying these two gives shape (B, N)
   disparity_list = min_disparity[:, None] * (1 - steps)[None, :] + max_disparity[:, None] * steps[None, :]
   fd_list = 1.0 / disparity_list  # Convert disparity to focal distance

   return fd_list  # shape (B, N)


def get_fd_list(fd_list_params, depth=None, N=None, depth_valid_mask=None):
   """Returns a focal distance list based on the provided parameters.
   
   depth: [B, 1, H, W] depth map in meters
   N: number of focal distances to generate
   
   returns: fd_list: [B, N] focal distances in meters
   """

   if fd_list_params is None:
      raise ValueError("fd_list_params must be provided to get_focal_stack.")
   elif 'fd_list' in fd_list_params and fd_list_params['fd_list'] is not None:
      fd_list = fd_list_params['fd_list']
      fd_list = torch.tensor(fd_list).unsqueeze(0).expand(depth.shape[0], -1).to(depth.device)  # shape (B, N)
   elif 'depth_dependent' in fd_list_params and fd_list_params['depth_dependent']:
      # Depth-based focal distance sampling
      if depth is None or N is None:
         raise ValueError("Depth and N must be provided for depth-based focal distance sampling.")
      
      fd_list = get_depth_dependent_fd_list(depth, N)  # shape (B, N)

   else:
      raise ValueError("fd_list_params must contain either 'fd_list' or 'depth_dependent' during evaluation.")
   
   return fd_list


def camera_setup(fnumber=0.5, focal_length=2.9*1e-3, sensor_size=3.1*1e-3, img_size=256):
   return ThinLenCamera(fnumber, focal_length, sensor_size, img_size)

def render_setup(kernel_size, psf_type, p):
   """Setup the PowerExpPSF render"""
   if psf_type == 'gauss':
      renderer = PowerExpPSF(kernel_size=kernel_size, p=2) # GaussPSF is a special case of PowerExpPSF with p=2
   else:
      raise ValueError(f"Unknown psf_type: {psf_type}")
   renderer.cuda()
   return renderer

def get_coc_and_blurred_image(dpt, aif, fd, camera, render, N):
   """This dpt should be a numpy array"""
   depth_map = dpt

   # B x H x W x 3 --> B x 3 x H x W for pytorch
   im = aif


   defocus = camera.getCoC(depth_map, fd, N).type(torch.float32)


   
   fs = render(im.cuda(), defocus.cuda()) # render the defocused image


   # fs: [B, C(3), H, W] --> [B, H, W, C(3)] for numpy format
   blurred_img = fs

   coc_map = defocus


   return coc_map, blurred_img


def gen_focal_stack(depth, rgb, K, fnumber=2.8, N=5, fd_list_params=None, depth_valid_mask=None, psf_type=None, p=None):

   rgb_vis = rgb.clone()

   fd_list = get_fd_list(fd_list_params, depth=depth, N=N, depth_valid_mask=depth_valid_mask)  # [B, N]
   if fd_list.shape[1] != N:
      N = fd_list.shape[1] # If we are using a fixed fd list of different size than N, update N
   renderer = render_setup(kernel_size=15, psf_type=psf_type, p=p)

   img_size = max(depth.shape[2], depth.shape[3]) # Map the longer edge to the sensor_size
   sensor_size = 36*1e-3
   pixel_size = sensor_size / img_size # Sensor size in meters
   # K: [B, 3, 3]
   fx = K[..., 0, 0]  # shape: (B,): gets the batch dimension and the (0,0) element
   fy = K[..., 1, 1]  # shape: (B,)


   # If you want a single focal length per matrix (e.g., average fx and fy):
   focal_length = pixel_size*((fx + fy) / 2)  # shape: (B,)


   camera = camera_setup(fnumber=fnumber, focal_length=focal_length, sensor_size=sensor_size, img_size=img_size)


   B, C, H, W = rgb_vis.shape  # rgb_vis is [B, C, H, W]



   depth_repeated = depth.repeat_interleave(N, dim=0)         # [B*N, ...]: repeat each depth pixel 
                                                         # N times, then go to the next pixel
   rgb_repeated = rgb_vis.repeat_interleave(N, dim=0)         # [B*N, C, H, W]

   fd_repeated = fd_list.reshape(-1)                         # [B*N]  


   coc_maps, focal_stack = get_coc_and_blurred_image(
      depth_repeated, rgb_repeated, fd_repeated, camera, renderer, N
   )

   coc_maps = coc_maps.view(B, N, H, W)          # [B, N, H, W]
   focal_stack = focal_stack.view(B, N, C, H, W)  # [B, N, C, H, W]
   
   return focal_stack.detach(), fd_list, coc_maps
