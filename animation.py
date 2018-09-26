import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def rand():
    return np.random.random_sample()

class Animation():

    def __init__(self, list_of_both, ball, goal_length, width, length):

        fig = plt.figure()
        t1 = plt.plot([0,width],[0,0],'g-',animated = True)
        t2 = plt.plot([0,width],[length,length],'g-',animated = True)
        t3 = plt.plot([0,0],[length,0],'g-',animated = True)
        t4 = plt.plot([width,width],[length,0],'g-',animated = True)
        g1 = plt.plot([width/2 - goal_length/2, width/2 + goal_length/2], [0,0], 'go', animated = True)
        g2 = plt.plot([width/2 - goal_length/2, width/2 + goal_length/2], [length, length], 'go', animated = True)
        x = np.linspace(0, 2 * np.pi, 120)
        y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
        # ims is a list of lists, each row is a list of artists to draw in the
        # current frame; here we are just animating one artist, the image, in
        # each frame
        ims = []
        for i in range(len(list_of_both[0])):
            
            l1 = plt.plot([list_of_both[0][i][j][0] for j in range(len(list_of_both[0][i]))], 
                          [list_of_both[0][i][j][1] for j in range(len(list_of_both[0][i]))], 'ro',animated = True)
            l2 = plt.plot([list_of_both[1][i][j][0] for j in range(len(list_of_both[1][i]))], 
                          [list_of_both[1][i][j][1] for j in range(len(list_of_both[1][i]))], 'bo',animated = True)
            im  = plt.plot(ball[i][0], ball[i][1], 'co', animated = True)
            #im = plt.plot(x, np.sin(x), 'ro', animated = True)
            im.extend(t1)
            im.extend(t2)
            im.extend(t3)
            im.extend(t4)

            im.extend(g1)
            im.extend(g2)

            im.extend(l1)
            im.extend(l2)
            #im.extend(b)

            ims.append(im)

        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                        repeat_delay=1000)

        # To save the animation, use e.g.
        #
        # ani.save("movie.mp4")
        #
        # or
        #
        # from matplotlib.animation import FFMpegWriter
        # writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        # ani.save("movie.mp4", writer=writer)

        plt.show()

def test():

    #blue or red; n of frames; n of agents;x or y

    l = [ [[[rand() * 30,rand() * 30] for k in range(5)] for j in range(50)], [[[rand() * 30,rand() * 30] for k in range(5)] for j in range(50)]   ]
    ball = [[i,i] for i in range(50)]
    Animation(l,ball, 7,50,60)

#test()