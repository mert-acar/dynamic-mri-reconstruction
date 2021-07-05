import matplotlib.pyplot as plt
import matplotlib.animation as animation

def visualize(data, cmap='gray', save_path=None, fps=60):
    '''
        Accepts data in shape [nt, w, h, 3] as float and creates a gif with given parameters.
    '''
    fig = plt.figure()
    im = []
    for i in range(data.shape[0]):
        temp = plt.imshow(data[i], cmap=cmap,  animated=True)
        plt.axis(False)
        im.append([temp])
    ani = animation.ArtistAnimation(fig, im, interval=1000 // fps, blit=True, repeat_delay=25)
    if save_path:
        ani.save(save_path, fps=fps)
        fig.clear()
        plt.close()
    else: 
        plt.show()


if __name__ == '__main__':
    import numpy as np
    from scipy.io import loadmat

    data_path = '../data/images/fs_0074_1_5T_combined_0.mat'
    data = loadmat(data_path)['xn']
    visualize(np.abs(data), 'gray', 'sample.gif')
