import taichi as ti
from particles import Particles
from matplotlib import cm
import argparse

# Parse arguments
parser = argparse.ArgumentParser(add_help=False)

# 0 - time independent 2D harmonic oscillator
# 1 - time dependent 2D harmonic oscillator
parser.add_argument('scene', type=int)

parser.add_argument('--N', type=int, default=200)
parser.add_argument('--dt', type=float, default=0.1)
parser.add_argument('-w', '--width', type=int, default=500)
parser.add_argument('-h', '--height', type=int, default=500)
args = parser.parse_args()

scene = args.scene

#============================#
# Particle System Parameters #
#============================#
N = args.N
h = 200 / N
dt = args.dt
lmbda = -0.75 if scene == 0 else 0.0

#=====================#
# Plotting Parameters #
#=====================#
width = args.width
height = args.height

ti.init(arch=ti.gpu)
image = ti.field(dtype=ti.f32, shape=(width, height))

if __name__ == '__main__':
    gui = ti.GUI('SPH', (width, height))
    video_manager = ti.tools.VideoManager(output_dir=f'./output/', framerate=24, automatic_build=False)
    
    particles = Particles(N, h, dt, lmbda, gui, scene)
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
        
        print(f'Done with step {i}', end='\r')
        i += 1
    
    video_manager.make_video(gif=False, mp4=True)
    