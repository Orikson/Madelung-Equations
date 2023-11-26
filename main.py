import taichi as ti
from particles2D import Particles2D
from particles import Particles
from matplotlib import cm

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
image = ti.field(dtype=ti.f32, shape=(width, height))

if __name__ == '__main__':
    gui = ti.GUI('SPH', (width, height))
    video_manager = ti.tools.VideoManager(output_dir=f'./output/', framerate=24, automatic_build=False)
    
    #particles = Particles2D(N, h, dt, gui, width, height)
    particles = Particles(N, h, dt, -0.75, gui)
    particles.setImage(image)
    
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
        if (i % 2 == 0):
            particles.densityImage(width, height, 4, -2)
            #print(particles.image.to_numpy())
            img = particles.image.to_numpy()
            img = cm.plasma(img / (img.max()))
            video_manager.write_frame(img)
    
    video_manager.make_video(gif=False, mp4=True)
    
    #print(particles.test())