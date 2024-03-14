import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.Dist import Dist

class ARPLoss(nn.CrossEntropyLoss):
    def __init__(self, **options):
        super(ARPLoss, self).__init__()
        self.use_gpu = options['use_gpu']
        self.weight_pl = float(options['weight_pl'])
        self.temp = options['temp']
        self.Dist = Dist(num_classes=options['num_classes'], feat_dim=options['feat_dim'])
        self.points = self.Dist.centers
        self.radius = nn.Parameter(torch.Tensor(1))
        self.radius.data.fill_(0)
        self.margin_loss = nn.MarginRankingLoss(margin=1.0)


    def forward(self, x, y, labels=None):
        dist_dot_p = self.Dist(x, center=self.points, metric='dot')
        dist_l2_p = self.Dist(x, center=self.points)
        logits = dist_l2_p - dist_dot_p

        if labels is None: return logits, 0
        loss = F.cross_entropy(logits / self.temp, labels)

        center_batch = self.points[labels, :]
        _dis_known = (x - center_batch).pow(2).mean(1)
        target = torch.ones(_dis_known.size()).cuda()
        loss_r = self.margin_loss(self.radius, _dis_known, target)

        loss = loss + self.weight_pl * loss_r

        return logits, loss

    def fake_loss(self, x):
        logits = self.Dist(x, center=self.points)
        """calculates the distance between the fake data x and the pred-defind class centers
        ---> What distance is used (eucledian, cosine-similarity? How does it produce logits? """
        
        prob = F.softmax(logits, dim=1)
        """The softmax function is used to convert raw logits into 
        probabilities that sum up to 1 across all classes."""
        
        loss = (prob * torch.log(prob)).sum(1).mean().exp()
        """computes the entropy for each sample in the batch. 
        Entropy is a measure of uncertainty or disorder; in this context, 
        it quantifies how spread out the probability distribution is
        across different classes for each fake sample.
        A higher entropy indicates that the model is less certain about the class of a given fake sample,
        distributing its probability more evenly across classes.
        --> .mean().exp() takes the mean of the entropies calculated 
        for each sample in the batch and then applies the exponential function. 
        The mean operation aggregates the entropy across the batch, 
        providing a single scalar value representing the average uncertainty
        of the model about THAT CLASS of the fake data. 
        Applying the exponential function to this mean entropy likely serves to
        amplify the effect of differences in entropy values, 
        making the loss more sensitive to changes in uncertainty."""

        return loss
