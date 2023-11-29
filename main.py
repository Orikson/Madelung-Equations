import taichi as ti
from particles import Particles
from matplotlib import cm

ti.init(arch=ti.gpu)

#============================#
# Particle System Parameters #
#============================#
N = 200
h = 200 / N
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
    
    particles = Particles(N, h, dt, -0.75, gui)
    particles.setImage(image)
    
    i=0
    while True:
        if not gui.running:
            break
        # Compute one timestep
        particles.step()

        # Update density image
        particles.densityImage(width, height, 4, -2)
        img = particles.image.to_numpy()
        img = cm.plasma(img / (img.max()))
        video_manager.write_frame(img)
        gui.set_image(img)

        # Plot particle positions and velocities
        particles.plot()
        
        # Update GUI
        gui.show()
        
        print(f'Done with step {i}/{steps}', end='\r')
        i += 1
    
    video_manager.make_video(gif=False, mp4=True)
    