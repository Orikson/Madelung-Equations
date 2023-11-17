import taichi as ti

@ti.data_oriented
class Particles2D:
    '''
    A collection of particles for 2D simulations
    '''
    
    @ti.func
    def V(self, x):
        '''
        Gradient of potential
        '''
        return x
    
    def __init__(self, N, h, dt, gui, width=400, height=400):
        self.N = N
        self.h = h
        self.m = 1/N
        self.dt = dt
        
        self.pos = ti.Vector.field(2, dtype=ti.f32, shape=N)
        self.vel = ti.Vector.field(2, dtype=ti.f32, shape=N)
        self.acc = ti.Vector.field(2, dtype=ti.f32, shape=N)
        self.rho = ti.field(dtype=ti.f32, shape=N)
        
        self.drho_1 = ti.Vector.field(2, dtype=ti.f32, shape=N)
        self.drho_2 = ti.Matrix.field(2, 2, dtype=ti.f32, shape=N)
        
        self.p_tensor = ti.Matrix.field(2, 2, dtype=ti.f32, shape=N)
        
        # plot
        self.graph = ti.field(dtype=ti.f32, shape=(width, height))
        self.width = width
        self.height = height
        self.gui = gui
        
        self.init()

    @ti.func
    def gaussian_kernel(self, r, h):
        '''
        Gaussian kernel
        r - some vector
        h - smoothing length
        '''
        return 1.0/(h**2 * ti.math.pi) * ti.exp(-r.dot(r)/h**2)

    @ti.func
    def gaussian_kernel_grad(self, r, h):
        '''
        Gradient of Gaussian kernel
        r - some vector
        h - smoothing length
        '''
        return -2*r * self.gaussian_kernel(r, h)/h
    
    @ti.func
    def gaussian_kernel_hess(self, r, h):
        '''
        Hessian of Gaussian kernel
        r - some vector
        h - smoothing length
        '''
        return 2 * self.gaussian_kernel(r, h)/h * (2*r.outer_product(r)/h - ti.Matrix.identity(ti.f32, 2))
    
    @ti.kernel
    def init(self):
        '''
        Initialize particles
        '''
        for i in range(self.N):
            self.pos[i] = 2.0 * ti.Vector([ti.random(), ti.random()]) - 1.0
            self.vel[i] = ti.Vector.zero(ti.f32, 2)
            self.acc[i] = ti.Vector.zero(ti.f32, 2)
            self.rho[i] = 0.0
            self.drho_1[i] = ti.Vector.zero(ti.f32, 2)
            self.drho_2[i] = ti.Matrix.zero(ti.f32, 2, 2)
            self.p_tensor[i] = ti.Matrix.zero(ti.f32, 2, 2)
    
    @ti.kernel
    def update_density(self):
        '''
        Update density of particles, including relevant gradients and second derivatives
        '''
        # zero out density and its derivatives
        for i in range(self.N):
            self.rho[i] = 0.0
            self.drho_1[i] = ti.Vector.zero(ti.f32, 2)
            self.drho_2[i] = ti.Matrix.zero(ti.f32, 2, 2)
        
        for i,j in ti.ndrange(self.N, self.N):
            r = self.pos[i] - self.pos[j]
            self.rho[i] += ti.static(self.m) * self.gaussian_kernel(r, ti.static(self.h))
            self.drho_1[i] += ti.static(self.m) * self.gaussian_kernel_grad(r, ti.static(self.h))
            
            # a faster first-order approximation of the Hessian
            #self.drho_2[i] += self.m * self.gaussian_kernel_hess(r, ti.static(self.h))
        
        # a more accurate second-order approximation of the Hessian
        for i,j in ti.ndrange(self.N, self.N):
            r = self.pos[i] - self.pos[j]
            self.drho_2[i] += ti.static(self.m) * (self.rho[j] - self.rho[i]) / self.rho[j] * self.gaussian_kernel_hess(r, ti.static(self.h))
    
    @ti.kernel
    def update_pressure(self):
        '''
        Update pressure tensor per particle
        '''
        # zero out pressure tensor
        for i in range(self.N):
            self.p_tensor[i] = ti.Matrix.zero(ti.f32, 2, 2)
        
        for i,j in ti.ndrange(self.N, self.N):
            r = self.pos[i] - self.pos[j]
            
            # TODO: speed up computation by not calculating this conglomerate in the double for loop
            self.p_tensor[i] += self.m / self.rho[j] * ti.static(1.0 / 4.0) * \
                (self.drho_1[j].outer_product(self.drho_1[j]) / self.rho[j] - \
                    self.drho_2[j] * self.rho[j]) * self.gaussian_kernel(r, ti.static(self.h))

    @ti.kernel
    def update(self):
        '''
        Update acceleration, velocity, and position of particles
        '''
        for i in range(self.N):
            self.acc[i] = -self.V(self.pos[i])
        
        for i,j in ti.ndrange(self.N, self.N):
            r = self.pos[i] - self.pos[j]
            Wij = self.gaussian_kernel_grad(r, ti.static(self.h))
            self.acc[i] += -10* self.m * ti.Vector([ \
                (ti.Vector([self.p_tensor[i][0,0], self.p_tensor[i][0,1]]) / self.rho[i]**2 + ti.Vector([self.p_tensor[j][0,0], self.p_tensor[j][0,1]]) / self.rho[j]**2).dot(Wij), \
                (ti.Vector([self.p_tensor[i][1,0], self.p_tensor[i][1,1]]) / self.rho[i]**2 + ti.Vector([self.p_tensor[j][1,0], self.p_tensor[j][1,1]]) / self.rho[j]**2).dot(Wij) \
            ])
        
        for i in range(self.N):
            self.vel[i] += ti.static(self.dt) * self.acc[i]
            self.vel[i] *= 0.99
            self.pos[i] += ti.static(self.dt) * self.vel[i]
    
    def plot(self):
        '''
        Generate image of particle plot
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
