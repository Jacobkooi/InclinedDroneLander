
import matplotlib.pyplot as plt
from matplotlib import animation

# Nice piece of code adopted from: https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553

def save_frames_as_gif(frames, path='.', filename='gym_animation.gif'):

    # Mess with this to change frame size
    fig = plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    fig.tight_layout()

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save('%s/Gifs/%s' % (path, filename), writer='imagemagick', fps=50)
    print('Gif saved to %s/Gifs/%s' % (path, filename))