import taichi as ti

k = 0.001

@ti.data_oriented
class Particles:
    '''
    A collection of particles for 2D simulation
    '''

    @ti.func
    def grad_V(self, x):
        '''
        Gradient of potential
        x - position vector
        '''
        return k * x

    def __init__(self, N, h, dt, lmbda, gui):
        self.N = N
        self.h = h
        self.m = 1/N
        self.dt = dt
        self.lmbda = lmbda

        self.pos = ti.Vector.field(2, dtype=ti.f32, shape=N)    # position (N,2)
        self.vel = ti.Vector.field(2, dtype=ti.f32, shape=N)    # velocity (N,2)
        self.acc = ti.Vector.field(2, dtype=ti.f32, shape=N)    # acceleration (N,2)
        self.rho = ti.field(dtype=ti.f32, shape=N)              # density (N)

        self.drho_1 = ti.Vector.field(2, dtype=ti.f32, shape=N)     # first derivative w.r.t. space of density (N,2)
        self.drho_2 = ti.Matrix.field(2, 2, dtype=ti.f32, shape=N)  # second derivative w.r.t. space of density (N,[2,2])

        self.p_tensor = ti.Matrix.field(2, 2, dtype=ti.f32, shape=N)    # pressure tensor (N,[2,2])

        # gui
        self.gui = gui

        self.initialize()

    @ti.kernel
    def test(self):# -> ti.f32:
        r = ti.Vector([0.0, 0.0]) - ti.Vector([0.5, 0.0])
        Wij = self.gaussian_kernel(r, ti.static(self.h))
        gWij = self.gaussian_kernel_grad(r, ti.static(self.h))
        dWij = self.gaussian_kernel_hess(r, ti.static(self.h))
        print(Wij, gWij, dWij)

    @ti.kernel
    def initialize(self):
        '''
        Initialize all particles
        Assumes steady state convergence, and initializes all positions randomly
        Velocities are set to 0
        '''
        for i in range(self.N):
            self.pos[i] = 2.0 * ti.Vector([ti.random(), ti.random()]) - 1.0
            self.vel[i] = ti.Vector([0.0, 0.0])
            self.acc[i] = ti.Vector([0.0, 0.0])
    
    @ti.func
    def gaussian_kernel(self, r, h):
        '''
        Gaussian kernel
        r - some 2D vector
        h - smoothing length
        '''
        return 1.0/(h*h * ti.math.pi) * ti.exp(-r.dot(r)/(h*h))

    @ti.func
    def gaussian_kernel_grad(self, r, h):
        '''
        Gradient of Gaussian kernel
        r - some 2D vector
        h - smoothing length
        '''
        return -2*r/(h*h) * self.gaussian_kernel(r, h)

    @ti.func
    def gaussian_kernel_hess(self, r, h):
        '''
        Hessian of Gaussian kernel
        r - some 2D vector
        h - smoothing length
        '''
        return 2*self.gaussian_kernel(r, h)/(h*h) * (2*r.outer_product(r)/(h*h) - ti.Matrix.identity(ti.f32, 2))

    @ti.kernel
    def update_density(self):
        '''
        Update density and spacial derivatives of density
        '''
        # zero all values
        for i in range(self.N):
            self.rho[i] = 0.0
            self.drho_1[i] = ti.Vector([0.0, 0.0])
            self.drho_2[i] = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

        #=======================================================#
        # rho_i = m_j * W_ij                                    #
        # dx drho_i = m_j * d_x dW_ij                           #
        # dxx drho_i = m_j (rho_j - rho_i) / rho_j * d_xx dW_ij #
        #=======================================================#
        for i,j in ti.ndrange(self.N, self.N):
            r = self.pos[i] - self.pos[j]
            self.rho[i] += ti.static(self.m) * self.gaussian_kernel(r, ti.static(self.h))
            self.drho_1[i] += ti.static(self.m) * self.gaussian_kernel_grad(r, ti.static(self.h))
        
        for i,j in ti.ndrange(self.N, self.N):
            r = self.pos[i] - self.pos[j]
            self.drho_2[i] += ti.static(self.m) * (self.rho[j] - self.rho[i])/self.rho[j] * self.gaussian_kernel_hess(r, ti.static(self.h))
    
    @ti.kernel
    def update_pressure(self):
        '''
        Update pressure tensor
        '''
        # zero all values
        for i in range(self.N):
            self.p_tensor[i] = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        
        #============================================================================#
        # P_xy = m_j / (4 rho_j) * ((dx drho_j) (dy drho_j) / rho_j - dxy rho_j) Wij #
        #============================================================================#
        for i,j in ti.ndrange(self.N, self.N):
            r = self.pos[i] - self.pos[j]
            self.p_tensor[i] += ti.static(self.m) / (4 * self.rho[j]) * ((self.drho_1[j].outer_product(self.drho_1[j]) / self.rho[j] - self.drho_2[j]) * self.gaussian_kernel(r, ti.static(self.h)))

    @ti.kernel
    def update(self):
        '''
        Update particle acceleration, velocity, and position
        '''
        for i in range(self.N):
            self.acc[i] = ti.static(-1.0 / self.m) * self.grad_V(self.pos[i])

        for i,j in ti.ndrange(self.N, self.N):
            p_ix = ti.Vector([self.p_tensor[i][0,0], self.p_tensor[i][0,1]])
            p_iy = ti.Vector([self.p_tensor[i][1,0], self.p_tensor[i][1,1]])
            p_jx = ti.Vector([self.p_tensor[j][0,0], self.p_tensor[j][0,1]])
            p_jy = ti.Vector([self.p_tensor[j][1,0], self.p_tensor[j][1,1]])

            rho_i2 = self.rho[i] * self.rho[i]
            rho_j2 = self.rho[j] * self.rho[j]
            g_Wij = self.gaussian_kernel_grad(self.pos[i] - self.pos[j], self.h)
            
            self.acc[i] -= self.m * ti.Vector([ \
                (p_ix / rho_i2 + p_jx / rho_j2).dot(g_Wij), \
                (p_iy / rho_i2 + p_jy / rho_j2).dot(g_Wij)])

        for i in range(self.N):
            self.vel[i] += (self.acc[i] + self.lmbda * self.vel[i]) * self.dt
            self.pos[i] += self.vel[i] * self.dt
    
    def plot(self):
        '''
        Plot particles
        '''
        for i in range(self.N):
            pos = [self.pos[i][0]*0.5 + 0.5, self.pos[i][1]*0.5 + 0.5]
            vel = [self.vel[i][0]*0.5, self.vel[i][1]*0.5]
            
            self.gui.circle(pos=pos, radius=5, color=0xffffff)
            self.gui.arrow(orig=pos, direction=vel, color=0xff0000)
        
    def step(self):
        '''
        Step forward in time
        '''
        self.update_density()
        self.update_pressure()
        self.update()
