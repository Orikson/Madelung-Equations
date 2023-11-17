import taichi as ti
from particles2D import Particles2D
from potentials import HarmonicPotential

ti.init(arch=ti.gpu)

#============================#
# Particle System Parameters #
#============================#
N = 100
h = 200 / N
dt = 0.01

#=====================#
# Simulation Settings #
#=====================#
T = 10.0

#=====================#
# Plotting Parameters #
#=====================#
width = 1000
height = 1000

steps = int(T/dt)
if __name__ == '__main__':
    gui = ti.GUI('SPH', (width, height))
    
    particles = Particles2D(N, h, dt, gui, width, height)
    
    for i in range(steps):
        if not gui.running:
            break
        particles.step()
        particles.plot()
        gui.show()
        
        #print(particles.p_tensor.to_numpy())
        
        print(f'Done with step {i}/{steps}')
    