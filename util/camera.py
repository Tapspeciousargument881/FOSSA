import torch

class ThinLenCamera():
    def __init__(self, fnumber=0.5, focal_length=2.9*1e-3, sensor_size=36*1e-3, img_size=256):
        self.focal_length = focal_length
        self.D = self.focal_length / fnumber
        self.pixel_size = sensor_size / img_size
        
    def getCoC(self, dpt, focus_dist, N):
        # B' = B x N
        # dpt : B' x 1 x H x W
        # focus_dist : (B',)
        # CoC formula from Wang et al
        
        focus_dist = focus_dist.view(-1, 1, 1, 1)
        self.focal_length = self.focal_length.view(-1, 1, 1, 1).repeat_interleave(N, dim=0)  # shape (BxN,1,1,1)
        self.D = self.D.view(-1, 1, 1, 1).repeat_interleave(N, dim=0)  # shape (BxN,1,1,1)

        CoC = (torch.abs(dpt - focus_dist)*(self.focal_length*self.D))/((dpt+(1e-8)) * (focus_dist - self.focal_length))


        # CoC radius in pixels
        sigma = CoC / 2 / self.pixel_size
        return sigma.type(torch.float32)