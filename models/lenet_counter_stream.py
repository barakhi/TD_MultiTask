
import torch.nn as nn
import torch.nn.functional as F


class BUTD_aligned_Lenet9(nn.Module):
    def __init__(self):
        super(BUTD_aligned_Lenet9, self).__init__()

        # BU1, BU2 - shared weights
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc = nn.Linear(320, 50)
        self.fc1 = nn.Linear(50, 50)
        self.fc2 = nn.Linear(50, 10)

        # TD stream
        self.emb = nn.Linear(9, 320)
        self.TDconv1 = nn.Conv2d(10, 1, kernel_size=5, padding=4)
        self.TDconv2 = nn.Conv2d(20, 10, kernel_size=5, padding=4)

        # lateral connections
        self.lat1_1 = nn.Conv2d(10, 10, kernel_size=1)
        self.lat1_2 = nn.Conv2d(20, 20, kernel_size=1)
        self.lat1_3 = nn.Conv2d(20, 20, kernel_size=1)

        self.lat2_1 = nn.Conv2d(1, 1, kernel_size=1)
        self.lat2_2 = nn.Conv2d(10, 10, kernel_size=1)
        self.lat2_3 = nn.Conv2d(20, 20, kernel_size=1)

    def forward(self, image, task):

        bs = image.size(0)

        # BU1
        bu1_e1 = self.conv1(image)
        bu1_c1 = F.max_pool2d(F.relu(bu1_e1), 2)
        bu1_e2 = self.conv2(bu1_c1)
        bu1_c2 = bu1_e2
        bu1_e3 = F.max_pool2d(F.relu(bu1_c2), 2)
        bu1_c3 = bu1_e3.view(-1, 320)
        rep_bu1 = F.relu(self.fc(bu1_c3))
        bu1_c3 = F.relu(self.fc1(rep_bu1))
        bu1_c3 = self.fc2(bu1_c3)

        # TD + lat1
        task_emb = self.emb(task)
        td_e3 = task_emb.view(bs,20,4,4) + self.lat1_3(bu1_e3)

        td_c2 = F.interpolate(td_e3, scale_factor=2, mode='nearest') + self.lat1_2(bu1_e2)
        td_e2 = self.TDconv2(td_c2)

        td_c1 = F.interpolate(td_e2, scale_factor=2, mode='nearest') + self.lat1_1(bu1_e1)
        td_e1 = self.TDconv1(td_c1)

        # BU2 * lat2
        image_plus = image * (self.lat2_1(td_e1))
        bu2_e1 = self.conv1(image_plus) * (self.lat2_2(td_c1))
        bu2_c1 = F.max_pool2d(F.relu(bu2_e1), 2)
        bu2_e2 = self.conv2(bu2_c1)
        bu2_c2 = bu2_e2 * (self.lat2_3(td_c2))
        bu2_e3 = F.max_pool2d(F.relu(bu2_c2), 2)
        bu2_c3 = bu2_e3.view(-1, 320)
        rep = F.relu(self.fc(bu2_c3))
        bu2_c3 = F.relu(self.fc1(rep))
        bu2_c3 = self.fc2(bu2_c3)  # ====================> loss

        # output + optional auxiliary losses
        bu1_logits = bu1_c3
        bu2_logits = bu2_c3
        seg_sigm = td_e1

        return bu1_logits, bu2_logits, seg_sigm



