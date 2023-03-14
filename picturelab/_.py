
    def Gaus(self, pred):
        # pred = self.Norm(pred)
        W, H = pred.size(2), pred.size(3)
        X, Y = torch.meshgrid(torch.arange(W).cuda(), torch.arange(H).cuda())
        weight = pred.clone().detach()
        pred_gaus = torch.zeros_like(pred)
        for bs in range(weight.size(0)):
            for c in range(weight.size(1)):
                weight[bs, c, :, :] = weight[bs, c, :, :] / (weight[bs, c, :, :].sum() + 1e-6)
                count = torch.count_nonzero(weight[bs, c, :, :])
                if not count > 0.1 * H * W:
                    pred_gaus[bs, c, :, :] = pred[bs, c, :, :]
                else:
                    gaus = torch.zeros((H, W)).cuda()
                    ux, uy = (weight[bs, c, :, :] * X).sum(), (weight[bs, c, :, :] * Y).sum()
                    sx, sy = (weight[bs, c, :, :] * (X - ux) ** 2).sum(), (weight[bs, c, :, :] * (Y - uy) ** 2).sum()
                    sxy = (weight[bs, c, :, :] * (X - ux) * (Y - uy)).sum()
                    if sx * sy - sxy ** 2 <= 0:
                        pred_gaus[bs, c, :, :] = pred[bs, c, :, :]
                    else:
                        a = 1 / (2 * math.pi * math.sqrt(sx * sy - sxy ** 2))
                        b = - 1 / (2 * (1 - sxy ** 2 / sx / sy))
                        dx = (X - ux) ** 2 / sx
                        dy = (Y - uy) ** 2 / sy
                        dxy = (X - ux) * (Y - uy) * sxy / sx / sy
                        pred_gaus[bs, c, :, :] = pred[bs, c, :, :] * torch.maximum(gaus, a * torch.exp(
                            b * (dx - 2 * dxy + dy)))
                        pred_gaus[bs, c, :, :] = self.Norm(pred_gaus[bs, c, :, :])
        return pred_gaus