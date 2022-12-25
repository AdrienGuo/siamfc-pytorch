class SiamFCTemplateMatch():
    def __init__(self) -> None:
        # model.eval()
        pass

    def init(self, z_img):
        # 用 template 做初始化

        # z_img = torch.from_numpy(z_img).to(
        #     self.device).permute(2, 0, 1).unsqueeze(0).float()

        # exemplar feature
        # self.z_f: (1, 256, 6, 6)
        # self.z_f = self.net.backbone(z_img)
        pass

    def match(self, x_img):
        # 在 search 上面做 matching

        # x_img = torch.from_numpy(x_img).to(
        #     self.device).permute(0, 3, 1, 2).float()

        # search feature
        # x_f: (1, 256, 22, 22)
        # x_f = self.net.backbone(x_img)

        # response: (1, 1, 17, 17)
        # response = self.net.head(self.z_f, x)
        # response = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        # responses: (3, 272, 272)

        # peak scale
        # scale_id: 在 scale_num 個裡面，最大的那個數 (選擇維度)
        # scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        # response = responses[scale_id]
        # response -= response.min()
        # response /= response.sum() + 1e-16
        # # response: (272, 272)
        # response = (1 - self.cfg.window_influence) * response + \
        #     self.cfg.window_influence * self.hann_window
        # loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        # disp_in_response: 在 self.upscale_sz，以 (135.5, 135.5) 當作中心點後，位置的比例
        # disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        # # disp_in_instance: 改換到 self.scale_sz 上的比例後 (除以 response_up)，再乘上 total_stride 變成實際的位移
        # disp_in_instance = disp_in_response * \
        #     self.cfg.total_stride / self.cfg.response_up
        # disp_in_image = disp_in_instance * self.x_sz * \
        #     self.scale_factors[scale_id] / self.cfg.instance_sz
        # self.center += disp_in_image

        # box: [x1, y1, w, h]
        # box = np.array([
        #     self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
        #     self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
        #     self.target_sz[1], self.target_sz[0]])
        return 
