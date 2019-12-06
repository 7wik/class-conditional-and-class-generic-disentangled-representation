import torch
from torch.autograd import Variable


class Normal(object):
    def __init__(self, mu, logvar):
        assert mu.size() == logvar.size()
        self.mu = mu
        self.logvar = logvar

    def size(self):
        return self.mu.size()

    def __getitem__(self, slice_val):
        return Normal(self.mu[:, slice_val], self.logvar[:, slice_val])

    def sample(self):
        eps = Variable(
            self.mu.data.new(self.mu.size()).normal_().type_as(self.mu.data),
            requires_grad=False,
        )
        return self.mu + eps * torch.exp(0.5 * self.logvar)

    def kl_div(self):
        # KL(self || N(0,1))
        sq_mean = torch.pow(self.mu, 2)
        var = torch.exp(self.logvar)
        return -0.5 * (1 + self.logvar - sq_mean - var).sum(1)

    def kl_div_from(self, other):
        # computes kld(d1 || d2) where d1, d2 are diagonal gaussians
        var1 = torch.exp(self.logvar)
        var2 = torch.exp(other.logvar)

        kld = 0.5 * torch.sum(
            -1
            + other.logvar
            - self.logvar
            + ((other.mu - self.mu) ** 2) / var2
            + var1 / var2,
            -1,
        )
        return kld
