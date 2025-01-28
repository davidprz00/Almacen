import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

# Equivariant Wavelet Neural Network
def B_spline(x):
    return (-462 * (-1 / 2 + x) ** 10 * t.sign(1 / 2 - x) + 330 * (-3 / 2 + x) ** 10 * t.sign(
            3 / 2 - x) - 165 * (-5 / 2 + x) ** 10 * t.sign(5 / 2 - x) + 55 * (-7 / 2 + x) ** 10 * t.sign(
            7 / 2 - x) - 11 * (-9 / 2 + x) ** 10 * t.sign(9 / 2 - x) + (-11 / 2 + x) ** 10 * t.sign(
            11 / 2 - x) - 462 * (1 / 2 + x) ** 10 * t.sign(1 / 2 + x) + (
                            165 * (3 + 2 * x) ** 10 * t.sign(3 / 2 + x)) / 512 - 165 * (
                            5 / 2 + x) ** 10 * t.sign(5 / 2 + x) + 55 * (7 / 2 + x) ** 10 * t.sign(
            7 / 2 + x) - 11 * (9 / 2 + x) ** 10 * t.sign(9 / 2 + x) + (11 / 2 + x) ** 10 * t.sign(
            11 / 2 + x)) / 7257600

def kernel_maker(s, weight, device):
    c_out, c_in, kernel_size = weight.shape

    eff_kernel_size = 2*int((kernel_size//2 + 1)*s) + (kernel_size)%2

    if (eff_kernel_size == 1): psi = weight.mean(dim = -1, keepdim = False).view(c_out, c_in, 1)
    else:
        x = t.arange(0, eff_kernel_size, device = device).view(eff_kernel_size, 1) - eff_kernel_size//2
        tau = (s*(t.arange(0, kernel_size, device = device).view(kernel_size) - kernel_size//2))

        psi = (weight.view(c_out, c_in, 1, kernel_size)*B_spline(x - tau)).sum(dim = -1, keepdim = False)

    return psi

class Lifting_Convolution(nn.Module):
    def __init__(self,
        c_in : int = None,
        c_out : int = None,
        kernel_size : int = None,
        k_s_max : int = None,
        device : t.device = t.device('cpu'),
        *args,
        **kwargs

    ) -> None:

        super().__init__(*args, **kwargs)

        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = t.tensor(kernel_size, device = device)
        self.k_s_min = (-t.log2((self.kernel_size + 2)//2) - 1).int()
        self.k_s_max = k_s_max

        self.s_list = 2**t.arange(self.k_s_min, k_s_max + 1, dtype = t.float32, device = device)

        self.weight = nn.Parameter(nn.init.xavier_uniform_(t.empty(c_out, c_in, kernel_size)), requires_grad = True).float()
        # self.bias = nn.Parameter(nn.init.xavier_uniform_(t.empty(c_out, 1)).view(c_out), requires_grad = True).float()

        self.device = device


    def forward(self, x : t.tensor):
        Batch, c_in, N = x.shape

        s = self.s_list[0]

        psi = kernel_maker(s, self.weight, self.device)/s

        out = F.conv1d(
                input = x,
                weight = psi,
                # bias = self.bias,
                padding = (psi.shape[-1])//2,
                dilation = 1
            ).view(Batch, self.c_out, 1, N)

        for s in self.s_list[1:]:
            psi = kernel_maker(s, self.weight, self.device)/s


            out_s = F.conv1d(
                input = x,
                weight = psi,
                # bias = self.bias,
                padding = (psi.shape[-1])//2,
                dilation = 1
            ).view(Batch, self.c_out, 1, N)

            out = t.cat((out, out_s), dim = -2)

        return out

class Group_Convolution(nn.Module):
    def __init__(self,
        c_in : int = None,
        c_out : int = None,
        kernel_size : int = None,
        k_s_max : int = None,
        device : t.device = t.device('cpu'),
        *args,
        **kwargs

    ) -> None:

        super().__init__(*args, **kwargs)

        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = t.tensor(kernel_size, device = device)
        self.k_s_min = (-t.log2((self.kernel_size + 2)//2) - 1).int()
        self.k_s_max = k_s_max

        self.s_list = 2**t.arange(self.k_s_min, k_s_max + 1, dtype = t.float32, device = device)

        self.weight = nn.Parameter(nn.init.xavier_uniform_(t.empty(c_out, c_in, kernel_size)), requires_grad = True).float()
        self.bias = nn.Parameter(nn.init.xavier_uniform_(t.empty(c_out, 1)).view(c_out), requires_grad = True).float()

        self.device = device

    def forward(self, x : t.tensor):
        Batch, c_in, rho, N = x.shape

        _2rho = 2**(2*t.arange(self.k_s_min, self.k_s_min + rho, dtype = t.float32, device = self.device))

        s = self.s_list[0]

        x = t.einsum('birn, r -> bin', x, _2rho)

        psi = kernel_maker(s, self.weight, self.device)/s**2

        out = F.conv1d(
                input = x,
                weight = psi,
                bias = self.bias,
                padding = (psi.shape[-1])//2,
                dilation = 1
            ).view(Batch, self.c_out, 1, N)

        for s in self.s_list[1:]:
            psi = kernel_maker(s, self.weight, self.device)/s**2

            out_s = F.conv1d(
                input = x,
                weight = psi,
                bias = self.bias,
                padding = (psi.shape[-1])//2,
                dilation = 1
            ).view(Batch, self.c_out, 1, N)

            out = t.cat((out, out_s), dim = -2)

        return out
    
class EstraNet(nn.Module):
    def __init__(self,
        n : int = None,
        d : int = None,
        n_c : int = None,
        beta : float = None,
        d_k : int = None,
        dropout : float = 0,
        device : t.device = t.device('cpu'),

        ) -> None:
        nn.Module.__init__(self)

        self.device = device

        self.n = n
        self.d = d
        self.n_c = n_c
        self.d_c = self.d//self.n_c
        self.beta = beta
        self.d_k = d_k
        self.dropout = dropout

        k = t.arange((d - 4)/4, -1, -1, dtype = t.float32, device = self.device)

        p_aux = (d - 4*k)/(d*(n - 1))
        ind = t.arange(0, n, 1, dtype = t.float32, device = device).view(self.n, 1)

        self.beta_p = beta*t.cat((p_aux, -p_aux, -p_aux, p_aux), dim = 0)[:self.d]
        self.beta_i_p = self.beta*t.cat((ind*p_aux, (n - 1 - ind)*p_aux, -ind*p_aux, -(n - 1 - ind)*p_aux), dim = 1)[:,:self.d]

        self.W_v = nn.Parameter(nn.init.xavier_uniform_(t.zeros(self.d, self.n_c, self.d_c, dtype = t.float32)), requires_grad = True)

        self.s_p = nn.Parameter(nn.init.ones_(t.zeros(n_c, dtype = t.float32)), requires_grad = True)
        self.c_p = nn.Parameter((1 + 2*t.arange(0, n_c, 1, dtype = t.float32, device = device))/n_c)

        self.W_p = nn.Parameter(nn.init.xavier_uniform_(t.zeros(d, n_c, self.d_c, dtype = t.float32)), requires_grad = True)

        self.W_A = nn.Parameter(self.W_A_creador(), requires_grad = True)

        self.W_o = nn.Parameter(nn.init.xavier_uniform_(t.zeros(n_c, self.d_c, d, dtype = t.float32)), requires_grad = True)

        self.drop = nn.Dropout(
            p = self.dropout
        )

    def W_A_creador(self) -> t.tensor:
        n_bloques = self.d_k//self.d_c

        W_A = t.tensor([], dtype = t.float32)

        for _ in range(n_bloques):
            bloque = t.randn(self.d_c, self.d_c, dtype = t.float32)
            q, _ = t.linalg.qr(bloque, mode = 'complete')
            q = q.T
            W_A = t.cat((W_A, q), dim = 0)

        extra = int(self.d_k - n_bloques*self.d_c)
        if (extra > 0):
            bloque = t.randn(self.d_c, self.d_c, dtype = t.float32)
            q, _ = t.linalg.qr(bloque, mode = 'complete')
            q = q.T
            W_A = t.cat((W_A, q[:extra,:]), dim = 0)

        alpha = t.norm(t.randn(self.d_k, self.d_c, dtype = t.float32), dim = 1).view(self.d_k, 1)

        return alpha*W_A

    def forward(self, x : t.tensor) -> t.tensor:
        phi_q = t.einsum('h, dhc, nd -> nhc', self.s_p, self.W_p, self.beta_i_p)
        phi_k = phi_q + t.einsum('h, h, dhc, d -> hc', self.s_p, self.c_p, self.W_p, self.beta_p).view(1, self.n_c, self.d_c)

        aux_phi_k = t.einsum('kc, nhc -> nhk', self.W_A, phi_k)/self.d_c**0.25
        aux_phi_q = t.einsum('kc, nhc -> nhk', self.W_A, phi_q)/self.d_c**0.25

        phi_fr_phi_k = t.cat((t.sin(aux_phi_k), t.cos(aux_phi_k)), dim = 2)/self.d_k**0.5
        phi_fr_phi_q = t.cat((t.sin(aux_phi_q), t.cos(aux_phi_q)), dim = 2)/self.d_k**0.5

        aux_A = t.einsum('nhk, nhk -> nh', phi_fr_phi_q, phi_fr_phi_k)

        norma = t.norm(t.einsum('h, dhc, d -> hc', self.s_p, self.W_p, self.beta_p), dim = 1, dtype = t.float32)

        x = t.einsum('dhc, bnd -> bnhc', self.W_v, x)
        A = t.einsum('nh, bnhc -> bnhc', aux_A, x)
        x = t.einsum('hcd, h, bnhc -> bnd', self.W_o, norma, x)
        x = self.drop(x)

        return x
    
class SoftMax_Attention(nn.Module):
    def __init__(self,
            d : int = None,
            n_c : int = None,
            softmax_att_smoothing : float = None,
            device : t.device = t.device('cpu'),

        ) -> None:
        nn.Module.__init__(self)

        self.device = device

        self.d = d
        self.n_c = n_c
        self.d_c = self.d//self.n_c
        self.softmax_att_smoothing = softmax_att_smoothing

        self.Densa_K = nn.Linear(
            in_features = self.d,
            out_features = int(self.n_c*self.d_c),
            dtype = t.float32
        )

        self.Densa_V = nn.Linear(
            in_features = self.d,
            out_features = int(self.n_c*self.d_c),
            dtype = t.float32
        )

        self.Q = nn.Parameter(nn.init.xavier_uniform_(t.zeros(n_c, self.d_c, dtype = t.float32)), requires_grad = True)

        self.Densa_Score = nn.Linear(
            in_features = int(self.n_c*self.d_c),
            out_features = 256,
            dtype = t.float32
        )

    def forward(self, x : t.tensor) -> t.tensor:
        b, n, d = x.shape

        K = self.Densa_K(x).view(b, n, self.n_c, self.d_c)
        V = self.Densa_V(x).view(b, n, self.n_c, self.d_c)

        A = self.softmax_att_smoothing*t.einsum('bnhc, hc -> bnh', K, self.Q).div(self.d_c**0.5)

        y = t.einsum('bnh, bnhc -> bnhc', A.softmax(dim = 1, dtype = t.float32), V).mean(dim = 1, dtype = t.float32).view(b, int(self.n_c*self.d_c))
        score = self.Densa_Score(y)

        return score
    
class Layer_Centering(nn.Module):
    def __init__(self,
            d : int = None,
            device : t.device = t.device('cpu'),

        ) -> None:
        nn.Module.__init__(self)

        self.device = device

        self.d = d

        self.epsilon = nn.Parameter(nn.init.constant_(t.zeros(self.d, dtype = t.float32), 0), requires_grad = True)

    def forward(self, x : t.tensor) -> t.tensor:
        return x - x.mean(dim = 2, keepdim = True) + self.epsilon.view(1, 1, self.d)
       
class Capa_Harmonica_1(nn.Module):
    def __init__(self, m, c_out, c_in, N, device):
        nn.Module.__init__(self)

        self.m = m
        self.c_out = c_out
        self.c_in = c_in
        self.N = N
        self.device = device

        if (self.m == 0): self.m = 1

        self.beta = nn.Parameter(t.nn.init.uniform_(t.zeros(self.c_out, self.c_in, 1, dtype = t.float32), a = -t.pi, b = t.pi), requires_grad = True)
        self.A = nn.Parameter(t.nn.init.uniform_(t.zeros(c_out, self.c_in, 1, dtype = t.float32), a = 0.75, b = 1.25), requires_grad = True)

        # self.beta = nn.Parameter(t.zeros(self.c_out, self.c_in, 1, dtype = t.float32), requires_grad = True)
        # self.A = nn.Parameter(t.ones(self.c_out, self.c_in, 1, dtype = t.float32), requires_grad = True)

        self.bias = nn.Parameter(t.nn.init.uniform_(t.zeros(c_out, 1, dtype = t.float32), a = -0.1, b = 0.1), requires_grad = True)

        self.ker_size = self.N//(2*abs(self.m))
        self.lim_inf = t.tensor([0], dtype = t.int32, device = self.device)
        self.lim_upp = t.tensor([self.N + self.ker_size], dtype = t.int32, device = self.device)

        self.phi = t.arange(0, self.ker_size, dtype = t.float32, device = device)

        self.c_outs = t.arange(0, self.c_out, dtype = t.int32, device = self.device).view(self.c_out, 1, 1)
        self.mu = t.arange(0, self.N, dtype = t.int32, device = self.device).view(self.N, 1)
        self.n = t.arange(-2*abs(self.m) + 1, 2*abs(self.m) + 1, dtype = t.int32, device = self.device)

        self.ind = self.mu + self.ker_size*self.n
        self.cond = self.ind.greater(self.lim_inf).logical_and(self.ind.less(self.lim_upp)).float()
        self.ind = self.ind.clamp(self.lim_inf, self.lim_upp)

        self.menos_1 = (-1)**(1 + self.n)

        self.BatchNorm = nn.BatchNorm1d(
            num_features = self.c_out,
            dtype = t.float32
        )

        self.m = m

    def activation(self, z):

        return F.sigmoid(z.abs() + self.bias)*z/(z.abs() + 1e-5)

    def C_BatchNorm(self, z):
        return self.BatchNorm(z.abs())*z/(z.abs() + 1e-5)

    def forward(self, z : t.tensor):
        k, c_in, N = z.shape

        ks = t.arange(0, k, dtype = t.int32, device = self.device).view(k, 1, 1, 1)

        if (self.m != 0):
            ker_W_m = t.abs(self.A)*t.exp(1j*(self.beta + 2*t.pi*self.m*self.phi/self.N))

            z_raw = F.conv1d(
                z,
                ker_W_m,
                bias = None,
                stride = 1,
                padding = self.ker_size,
                dilation = 1,
                groups = 1
            )

            salida = t.zeros(k, self.c_out, self.N, dtype = t.cfloat, device = self.device)
            salida = (self.cond*self.menos_1*z_raw[ks, self.c_outs, self.ind]).sum(dim = 3, keepdim = False, dtype = t.cfloat)

        else:
            salida = z.sum(dim = 2, keepdim = False, dtype = t.cfloat)
            salida = salida.view(k, self.c_out, c_in)*(t.abs(self.A)*t.exp(1j*self.beta)).view(self.c_out, self.c_in)
            salida = salida.sum(dim = 2, keepdim = False, dtype = t.cfloat).view(k, self.c_out, 1).tile(1, 1, N)

        salida = self.activation(salida)

        # salida = self.C_BatchNorm(salida)


        return salida
    
class Convolucion_Transformable(nn.Module):
    def __init__(self,
        device,
        N : int,
        c_in : int,
        c_out : int,
        kernel_size : int = 3,
        dilatacion : int = 1,

        eps : float = 0.1,

        inferencia : int = 1,
        modulado : bool = False,
        estatico : bool = False,
        dinamico : bool = False,

        olvido : float = 0
    ):
        nn.Module.__init__(self)

        self.w = nn.Parameter(nn.init.normal_(t.zeros(c_out, c_in, kernel_size, dtype = t.float32), 0, 1/np.sqrt(c_in*kernel_size)), requires_grad = True)
        self.b = nn.Parameter(nn.init.normal_(t.zeros(1, c_out, 1, dtype = t.float32), 0, 1/np.sqrt(c_in*kernel_size)), requires_grad = True)

        if (estatico):
            self.dw_e = nn.Parameter(nn.init.normal_(t.zeros(c_out, c_in, kernel_size, dtype = t.float32), 0, eps), requires_grad = True)

        if (dinamico):
            self.w_dw_d = nn.Parameter(nn.init.normal_(t.zeros(kernel_size, c_in, kernel_size, dtype = t.float32), 0, eps), requires_grad = True)

        if (modulado):
            self.w_m = nn.Parameter(nn.init.zeros_(t.zeros(kernel_size, c_in, kernel_size, dtype = t.float32)), requires_grad = True)
            # self.w_m = nn.Parameter(nn.init.zeros_(t.zeros(1, c_out*c_in, 1, dtype = t.float32)), requires_grad = True)

        self.device = device

        self.N = N
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.dilatacion = dilatacion

        self.inferencia = inferencia

        self.modulado = modulado
        self.estatico = estatico
        self.dinamico = dinamico

        self.olvido = olvido
        self.L1_desfase_dinamico = t.zeros(1, dtype = t.float32, device = device)

    def forward(self, x):
        Batch = x.shape[0]
        N, c_in, c_out, mu = self.N, self.c_in, self.c_out, self.kernel_size

        X_amp = t.zeros(Batch, c_out*c_in, N*mu, dtype = t.float32, device = self.device)

        k = t.arange(Batch, dtype = t.int32, device = self.device).view(Batch, 1, 1, 1, 1)
        cos = t.arange(c_out, dtype = t.int32, device = self.device).view(c_out, 1, 1, 1)
        cis = t.arange(c_in, dtype = t.int32, device = self.device).view(c_in, 1, 1)
        n = t.arange(N, dtype = t.int32, device = self.device).view(N, 1)
        mus = t.arange(mu, dtype = t.int32, device = self.device)

        if (self.inferencia == 1):
            if (self.estatico) and (self.olvido != 0):
                grid_e = t.zeros(Batch, c_out*c_in, 2, N*mu, dtype = t.int32, device = self.device)

                Gr_s_z = n + self.dilatacion*(mus.sub(mu//2)) + self.dw_e[cos, cis, mus].int()

                grid_e[k, cos*c_in + cis, 0, n*mu + mus] = Gr_s_z
                grid_e[k, cos*c_in + cis, 1, n*mu + mus] = Gr_s_z + 2*self.dw_e[cos, cis, mus].sub(self.dw_e[cos, cis, mus].int()).heaviside(t.ones(mu, dtype = t.float32, device = self.device)).int() - 1

                grid_e_clip = grid_e.clamp(0, N - 1)

                C1_s = grid_e[k, cos*c_in + cis, 0, n*mu + mus].greater_equal(0).logical_and(grid_e[k, cos*c_in + cis, 0, n*mu + mus].less(N))
                C2_s = grid_e[k, cos*c_in + cis, 1, n*mu + mus].greater_equal(0).logical_and(grid_e[k, cos*c_in + cis, 1, n*mu + mus].less(N))

                x1_s_i = self.olvido*(1 - self.dw_e[cos, cis, mus].sub(self.dw_e[cos, cis, mus].int()).abs())*x[k, cis, grid_e_clip[k, cos*c_in + cis, 0, n*mu + mus]]
                x2_s_i = self.olvido*self.dw_e[cos, cis, mus].sub(self.dw_e[cos, cis, mus].int()).abs()*x[k, cis, grid_e_clip[k, cos*c_in + cis, 1, n*mu + mus]]

                X_amp[k, cos*c_in + cis, n*mu + mus] += C1_s.float()*x1_s_i
                X_amp[k, cos*c_in + cis, n*mu + mus] += C2_s.float()*x2_s_i

            if (self.dinamico) and (self.olvido != 1):
                dw_d = F.conv1d(
                    input = x,
                    weight = self.w_dw_d,
                    padding = (mu + (mu - 1)*(self.dilatacion - 1))//2,
                    dilation = self.dilatacion
                )

                self.L1_desfase_dinamico = dw_d.abs().max()

                grid_d = t.zeros(Batch, c_out*c_in, 2, N*mu, dtype = t.int32, device = self.device)

                Gr_d_z = n + self.dilatacion*(mus.sub(mu//2)) + dw_d[k, mus, n].int()

                grid_d[k, cos*c_in + cis, 0, n*mu + mus] = Gr_d_z
                grid_d[k, cos*c_in + cis, 1, n*mu + mus] = Gr_d_z + 2*dw_d[k, mus, n].sub(dw_d[k, mus, n].int()).heaviside(t.ones(mu, dtype = t.float32, device = self.device)).int() - 1

                grid_d_clip = grid_d.clamp(0, N - 1)

                C1_d = grid_d[k, cos*c_in + cis, 0, n*mu + mus].greater_equal(0).logical_and(grid_d[k, cos*c_in + cis, 0, n*mu + mus].less(N))
                C2_d = grid_d[k, cos*c_in + cis, 1, n*mu + mus].greater_equal(0).logical_and(grid_d[k, cos*c_in + cis, 1, n*mu + mus].less(N))

                x1_d_i = (1 - self.olvido)*(1 - dw_d[k, mus, n].sub(dw_d[k, mus, n].int()).abs())*x[k, cis, grid_d_clip[k, cos*c_in + cis, 0, n*mu + mus]]
                x2_d_i = (1 - self.olvido)*dw_d[k, mus, n].sub(dw_d[k, mus, n].int()).abs()*x[k, cis, grid_d_clip[k, cos*c_in + cis, 1, n*mu + mus]]

                X_amp[k, cos*c_in + cis, n*mu + mus] += C1_d.float()*x1_d_i
                X_amp[k, cos*c_in + cis, n*mu + mus] += C2_d.float()*x2_d_i


        if (self.modulado):
            X_amp *= F.conv1d(
                input = x,
                weight = self.w_m,
                padding = (mu + (mu - 1)*(self.dilatacion - 1))//2,
                dilation = self.dilatacion
            ).permute(0, 2, 1).reshape(Batch, N*mu).sigmoid().view(Batch, 1, N*mu)

        y = F.conv1d(
            input = X_amp,
            weight = self.w,
            stride = mu,
            groups = c_out
        ) + self.b

        return y