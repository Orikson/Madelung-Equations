import taichi as ti
from particles2D import Particles2D
from particles import Particles

ti.init(arch=ti.gpu)

#============================#
# Particle System Parameters #
#============================#
N = 200
h = 1 #200 / N
dt = 0.1

#=====================#
# Simulation Settings #
#=====================#
T = 10.0

#=====================#
# Plotting Parameters #
#=====================#
width = 500
height = 500

steps = int(T/dt)
if __name__ == '__main__':
    gui = ti.GUI('SPH', (width, height))
    
    #particles = Particles2D(N, h, dt, gui, width, height)
    particles = Particles(N, h, dt, -0.1, gui)
    
    #for i in range(steps):
    i=0
    while True:
        if not gui.running:
            break
        particles.step()
        particles.plot()
        gui.show()
        
        #print(particles.p_tensor.to_numpy())
        
        print(f'Done with step {i}/{steps}', end='\r')
        i += 1
    
    #print(particles.test())