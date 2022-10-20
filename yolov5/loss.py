import torch


def build_targets(self, p, targets):
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    na, nt = self.na, targets.shape[0]  # number of anchors, targets
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(7,
                        device=self.device)  # normalized to gridspace gain
    ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(
        1, nt)  # same as .repeat_interleave(nt)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]),
                        2)  # append anchor indices

    g = 0.5  # bias
    off = torch.tensor(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],  # j,k,l,m
            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
        ],
        device=self.device).float() * g  # offsets

    for i in range(self.nl):
        anchors, shape = self.anchors[i], p[i].shape
        gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

        # Match targets to anchors
        t = targets * gain  # shape(3,n,7)
        if nt:
            # Matches
            r = t[..., 4:6] / anchors[:, None]  # wh ratio
            j = torch.max(
                r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            t = t[j]  # filter

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            j, k = ((gxy % 1 < g) & (gxy > 1)).T
            l, m = ((gxi % 1 < g) & (gxi > 1)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            t = t.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
        else:
            t = targets[0]
            offsets = 0

        # Define
        bc, gxy, gwh, a = t.chunk(
            4, 1)  # (image, class), grid xy, grid wh, anchors
        a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid indices

        # Append
        indices.append((b, a, gj.clamp_(0, shape[2] - 1),
                        gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class

    return tcls, tbox, indices, anch