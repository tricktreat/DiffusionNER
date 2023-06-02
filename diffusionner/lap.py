import torch

def auction_lap(X, eps=None, compute_score=True):
    device = X.device
    if X.size(0) == 0 or X.size(1) == 0:
        return torch.tensor([], device=device),torch.tensor([], device=device),0

    X = -X
    flag = 0
    if X.size(0) > X.size(1):
        flag = 1
        X = X.transpose(0, 1)
    eps = 1 / X.shape[0] if eps is None else eps
    
    cost = torch.zeros((1, X.shape[1]), device = device)
    curr_ass = torch.zeros(X.shape[0], device = device).long() - 1
    bids = torch.zeros(X.shape, device = device)

    counter = 0
    while (curr_ass == -1).any():
        counter += 1

        # bidding
        
        unassigned = (curr_ass == -1).nonzero().squeeze(-1)
        value = X[unassigned] - cost
        top_value, top_idx = value.topk(2, dim=1)
        
        first_idx = top_idx[:,0]
        first_value, second_value = top_value[:,0], top_value[:,1]
        
        bid_increments = first_value - second_value + eps
        
        bids_ = bids[unassigned]
        bids_.zero_()
        bids_.scatter_(
            dim=1,
            index=first_idx.contiguous().view(-1, 1),
            src=bid_increments.view(-1, 1)
        )

        # assignment
        
        have_bidder = (bids_ > 0).int().sum(dim=0).nonzero()
        
        high_bids, high_bidders = bids_[:,have_bidder].max(dim=0)
        high_bidders = unassigned[high_bidders.squeeze()]
        
        cost[:,have_bidder] += high_bids
        
        curr_ass[(curr_ass.view(-1, 1) == have_bidder.view(1, -1)).sum(dim=1)] = -1
        curr_ass[high_bidders] = have_bidder.squeeze()
    
    score = None
    if compute_score:
        score = int(X.gather(dim=1, index=curr_ass.view(-1, 1)).sum())
    
    if flag == 1:
        return  curr_ass, torch.arange(X.size(0), device=device), -score
    return torch.arange(X.size(0), device=device), curr_ass, -score
    # return curr_ass, score, counter

# import numpy as np
# from scipy.optimize import linear_sum_assignment
# import torch
# cost = torch.randint(-10, 10, (43, 334))
# # cost = torch.randn(34, 54)
# # cost = torch.tensor(np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]]))
# print(linear_sum_assignment(cost))
# print(cost[linear_sum_assignment(cost)].sum())
# print(auction_lap(cost, 0.01))

class SinkhornDistance(torch.nn.Module):
    r"""
        Given two empirical measures each with :math:`P_1` locations
        :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
        outputs an approximation of the regularized OT cost for point clouds.
        Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
        'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
        'mean': the sum of the output will be divided by the number of
        elements in the output, 'sum': the output will be summed. Default: 'none'
        Shape:
            - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
            - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps=1e-3, max_iter=100, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, mu, nu, C):
        u = torch.ones_like(mu)
        v = torch.ones_like(nu)

        # Sinkhorn iterations
        for i in range(self.max_iter):
            v = self.eps * \
                (torch.log(
                    nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            u = self.eps * \
                (torch.log(
                    mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(
            self.M(C, U, V)).detach()
        # Sinkhorn distance
        cost = torch.sum(
            pi * C, dim=(-2, -1))
        return cost, pi

    def M(self, C, u, v):
        '''
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / epsilon$"
        '''
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps