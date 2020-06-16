import torch 
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import math
import pdb

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lin = nn.Linear(10, 5)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, X):
        all_params = self.lin(X)

        # all_params[:,:2] = self.Sigmoid(all_params[:,:2])
        
        # all_params[:,2:4] = self.Sigmoid(all_params[:,2:4]) + 1

        # all_params[:,4] = torch.clamp(self.Sigmoid(all_params[:,4]), max=0.7, min=0.1)


        all_params[:,:2] = all_params[:,:2]
        
        all_params[:,2:4] = torch.abs(all_params[:,2:4])

        all_params[:,4] = torch.clamp(self.Sigmoid(all_params[:,4]), max=0.7, min=0.1)

        return all_params

'''
def Gaussian2D(params, y):
    (mu_x,mu_y), (sig_x,sig_y), rho_xy = (params[0],params[1]), (torch.exp(params[2]),torch.exp(params[3])), params[4]
    # print((mu_x,mu_y), (sig_x,sig_y), rho_xy)
    covariance = rho_xy*sig_x*sig_y
    rv = MultivariateNormal(torch.Tensor([mu_x, mu_y]), torch.Tensor([[sig_x**2, covariance], [covariance, sig_y**2]]))
    logP = rv.log_prob(y)
    logP.requires_grad = True
    return 
'''

def Gaussian2D1(params, y):
    (mu_x,mu_y), (sig_x,sig_y), rho_xy = (params[0],params[1]), (torch.exp(params[2]),torch.exp(params[3])), params[4]
    covariance = params[4]*torch.exp(params[2])*torch.exp(params[3])
    try:
    	rv = MultivariateNormal(torch.Tensor([params[0], params[1]]), torch.Tensor([[torch.exp(params[2])**2, covariance], [covariance, torch.exp(params[3])**2]]))
    except RuntimeError:
    	print("(mu_x,mu_y), (sig_x,sig_y), rho_xy", (mu_x,mu_y), (sig_x,sig_y), rho_xy)
    return rv.log_prob(y)
    
    
# def Gaussian2Dmaual(params,y):
#     x1=y[0]-params[0]; x2=y[1]-params[1]
#     x1 = torch.Tensor()
#     s00=torch.exp(params[2]); s11=torch.exp(params[3]); s01=params[4]*s00*s11
#     #print(s00,s11,s01)
#     if (s00**2*s11**2-s01**2)<=0:
#         print('fuckfuckfuckfuck')
#         fuck()
#     return (-0.5* (   s00**2*x1**2+2*s01*x1*x2+s11**2*x2**2   ))- 0.5*torch.log((s00*s11)**2-s01**2)


def Gaussian2Dmaual(params,y):
    x0=y[0]-params[0]; x2=y[1]-params[1]
    x1 = torch.Tensor([x0.item(), x0.item()])
    x3 = x1[0]
    s00=torch.exp(params[2]); s11=torch.exp(params[3]); s01=params[4]*s00*s11
    #print(s00,s11,s01)
    if (s00**2*s11**2-s01**2)<=0:
        print('fuckfuckfuckfuck')
        fuck()

    return (-0.5* (   s00**2*x3**2+2*s01*x3*x2+s11**2*x2**2   ))- 0.5*torch.log((s00*s11)**2-s01**2)


def Gaussian2Dmy(params, event):
    y = event.view(2,-1)

    mu_x = params[0].view(1,1)
    mu_y = params[1].view(1,1)
    sig_x = params[2].view(1,1)
    sig_y = params[3].view(1,1)
    rho_xy = params[4].view(1,1)

    mu = torch.cat((mu_x,mu_y))

    covariance = rho_xy*sig_x*sig_y
    covariance = covariance.view(1,1)

    cov_mat_r1 = torch.cat((sig_x**2, covariance), 1)
    cov_mat_r2 = torch.cat((covariance, sig_y**2), 1)
    cov_mat = torch.cat((cov_mat_r1, cov_mat_r2))

    covmat_det = torch.sum((sig_x**2*sig_y**2)) - covariance**2

    things_inexp = -1/2*torch.mm( torch.t(y-mu),torch.mm(torch.inverse(cov_mat),(y-mu)) )

    coeff_before_exp = 1/(2*np.pi*torch.sqrt(covmat_det)) 

    P = coeff_before_exp * torch.exp(things_inexp)
    logP = torch.sum(torch.log(P))

    # pdb.set_trace()

    return logP


def Gaussian2DNll(all_params, targets):
    loss = torch.zeros(targets.shape[0])
    for idx, (params, target) in enumerate(zip(all_params, targets)):
        #print('testing gaussain:',(Gaussian2D1(params, target)-Gaussian2Dmaual(params, target)).item())
        #print("param", params)                
        
        #loss[idx] += Gaussian2D1(params, target)
        #print("shape", loss[idx].shape)
        # loss[idx] += Gaussian2Dmaual(params, target)
        #loss[idx] += nonsense(params, target)
        # pdb.set_trace()
        loss[idx] += Gaussian2Dmy(params, target)
        if loss[idx] >= 100 or loss[idx] <= -100:
            print("idx", idx, loss[idx])
    cost = torch.sum(-loss)
    return cost


def Gaussian2DNllmy(all_params, targets):
    loss = torch.zeros(targets.shape[0])
    for idx, (params, target) in enumerate(zip(all_params, targets)):
        #print('testing gaussain:',(Gaussian2D1(params, target)-Gaussian2Dmaual(params, target)).item())
        #print("param", params)                
        
        #loss[idx] += Gaussian2D1(params, target)
        #print("shape", loss[idx].shape)
        # loss[idx] += Gaussian2Dmaual(params, target)
        #loss[idx] += nonsense(params, target)
        # pdb.set_trace()
        loss[idx] += Gaussian2Dmy(params, target)
        # if loss[idx] >= 100 or loss[idx] <= -100:
        #     print("idx", idx, loss[idx])
    cost = torch.sum(-loss)
    return cost    


torch.manual_seed(0)
X = torch.randn(50, 10)*1
Y = torch.randn(50, 2)*1

model = Model()

criterion = Gaussian2DNll
optimizer = torch.optim.Adagrad(model.parameters())

for i in range(10000):
    #forward prop
    all_params = model(X)

    #calc cost
    cost = Gaussian2DNll(all_params, Y)

    #print
    if i % 10 == 9:
        print(i, cost.item())
        if math.isnan(cost):
            print("i", i)

    #backward prop
    optimizer.zero_grad()
    cost.backward()

    #update param
    optimizer.step()

    # if i % 10 == 9:
    #     print("in the same time", Gaussian2DNllmy(model(X), Y),'\n\n')            


print("final", Gaussian2DNllmy(model(X), Y))