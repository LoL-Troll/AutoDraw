"""
AutoDraw - inital version

original (incomplete) code:

    Austin Nguyen, Jun 1, 2020

    How I Used Machine Learning to Automatically Hand-Draw Any Picture
    Supervised and unsupervised learning made easy!

    https://towardsdatascience.com/how-i-used-machine-learning-to-automatically-hand-draw-any-picture-7d024d0de997

code completion:

    Bartlomiej "furas" Burek (https://blog.furas.pl)
    https://gist.github.com/austinnguyen517
    https://github.com/furas/AutoDraw
    date: 2021.05.04

# pip install opencv-python
# pip install numpy
# pip install PyAutoGUI
# pip install sklearn
# pip install kdtree
"""


from tkinter import *
from tkinter.colorchooser import askcolor
from tkinter.filedialog import askopenfilename

import cv2  # for image processing
import numpy as np
import pyautogui as pg  # for getting info about screen size
from sklearn.cluster import KMeans  # KMeans algorithm for counting colors in the image
from kdtree import create  # K-NN algorithm for searching paths
from collections import defaultdict
import operator
import time
import threading



class Paint(object):
    DEFAULT_PEN_SIZE = 50.0
    DEFAULT_COLOR = 'black'


    def __init__(self):
        self.root = Tk()
        self.root.title("Imai")


        self.Imai_button = Button(self.root, text='Imai', command=self.use_Imai)
        self.Imai_button.grid(row=0, column=2)

        self.c = Canvas(self.root, bg='white', width=pg.size()[0], height=pg.size()[1])
        self.c.grid(row=1, columnspan=5)

        self.setup()
        self.root.mainloop()  # Launch the GUI

    def setup(self):
        self.old_x = None  # initialize x position
        self.old_y = None  # initialize y position
        self.color = self.DEFAULT_COLOR

    def use_Imai(self):
        image = askopenfilename()  # displaying a window to select an image from computer
        self.line_width = 7
        ad = Imai(image, self)  # Start auto drawing
        x = threading.Thread(target=ad.run)
        x.start()

    def paint_Imai(self, x, y):
        self.line_width = 8
        paint_color = self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, x, y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = x
        self.old_y = y

    def reset(self):
        self.old_x, self.old_y = None, None

    def setOldxy(self, x, y):
        self.old_x, self.old_y = x, y

class Imai(object):
    def __init__(self, name, paint):

        # Tunable parameters
        self.scale = 4 / 12  #
        self.num_colors = 2
        self.paint = paint

        # Load Image. Switch axes to match computer screen
        self.img = self.load_img(name)  # the image to be sketched
        self.blur = 0  # blur level
        self.img = np.swapaxes(self.img, 0, 1)
        self.img_shape = self.img.shape  # Image size
        self.dim = pg.size()  # size of the canvas which it will draw on

        # Scale to draw inside part of the screen
        self.startX = ((1 - self.scale) / 2) * self.dim[0]
        self.startY = ((1 - self.scale) / 2) * self.dim[1]

        #
        self.dim = (self.dim[0] * self.scale, self.dim[1] * self.scale)

        # fit the picture into this section of the screen
        if self.img_shape[1] > self.img_shape[0]:
            # if it's taller that it is wide, truncate the wide section
            self.dim = (int(self.dim[1] * (self.img_shape[0] / self.img_shape[1])), self.dim[1])
        else:
            # if it's wider than it is tall, truncate the tall section
            self.dim = (self.dim[0], int(self.dim[0] * (self.img_shape[1] / self.img_shape[0])))

        # Get dimension to translate picture. Dimension 1 and 0 are switched due to computer dimensions
        self.pseudoDim = (int(self.img_shape[1]), int(self.img_shape[0]))

        # Initialize directions for momentum when creating path
        self.maps = {0: (1, 1), 1: (1, 0), 2: (1, -1), 3: (0, -1), 4: (0, 1), 5: (-1, -1), 6: (-1, 0), 7: (-1, 1)}
        self.momentum = 1
        self.curr_delta = self.maps[self.momentum]

        # Create Outline
        self.drawing = self.process_img(self.img)
        self.show()

    def load_img(self, name):
        image = cv2.imread(name)  # reading the image
        return image

    def show(self):
        cv2.imshow('image', np.swapaxes(self.img, 0, 1))  # display image in screen
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def rescale(self, img, dim):
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return resized

    def translate(self, coord):
        ratio = (coord[0] / self.pseudoDim[1], coord[1] / self.pseudoDim[0])
        deltas = (int(ratio[0] * self.dim[0]), int(ratio[1] * self.dim[1]))
        return self.startX + deltas[0], self.startY + deltas[1]

    def process_img(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 50, 75)
        canny = self.rescale(canny, self.pseudoDim)
        r, res = cv2.threshold(canny, 50, 255, cv2.THRESH_BINARY_INV)

        return res

    def rgb_to_hex(self, rgb):
        return '%02x%02x%02x' % rgb

    def execute(self, commands):
        press = 0  # flag indicating whether we are putting pressure on paper
        self.paint.reset()
        for (i, comm) in enumerate(commands):
            if type(comm) == str:
                if comm == 'UP':
                    press = 0
                if comm == 'DOWN':
                    press = 1
            else:
                if press == 1:
                    if pg.size()[0] / 4 <= comm[0] <= (pg.size()[0] * 3 / 4):
                        self.paint.paint_Imai(comm[0], comm[1] - 100)
                else:
                    if pg.size()[0] / 4 <= comm[0] <= (pg.size()[0] * 3 / 4):
                        self.paint.setOldxy(comm[0], comm[1] - 100)

        return

    def createPath(self):
        # check for closest point. Go there. Add click down. Change curr. Remove from set and tree. Then, begin
        new_pos = tuple(self.KDTree.search_nn(self.curr_pos)[0].data)  # check for closest point

        self.commands.append(new_pos)
        self.commands.append("DOWN")
        self.curr_pos = new_pos
        self.KDTree = self.KDTree.remove(list(new_pos))  # removing the point so it will not be visited again in KNN
        self.hashSet.remove(new_pos)  # remove the point from the list

        while len(self.hashSet) > 0:  # repeat until it visits all the points
            prev_direction = self.momentum
            candidate = self.checkMomentum(self.curr_pos)
            if self.isValid(candidate):
                new = tuple(map(operator.add, self.curr_pos, candidate))
                new_pos = self.translate(new)
                if prev_direction == self.momentum and type(self.commands[-1]) != str:  # the direction didn't change
                    self.commands.pop()
                self.commands.append(new_pos)
            else:
                self.commands.append("UP")
                new = tuple(self.KDTree.search_nn(self.curr_pos)[0].data)
                new_pos = self.translate(new)
                self.commands.append(new_pos)
                self.commands.append("DOWN")
            self.curr_pos = new  # set the current point to the next available nearest point
            self.KDTree = self.KDTree.remove(list(new))  # remove the point
            self.hashSet.remove(new)
            print('Making path...number points left: ', len(self.hashSet))

        return

    def isValid(self, delta):
        return len(delta) == 2

    def checkMomentum(self, point):
        # Returns best next relative move w.r.t. momentum and if in hashset
        self.curr_delta = self.maps[self.momentum]
        moments = self.maps.values()
        deltas = [d for d in moments if (tuple(map(operator.add, point, d)) in self.hashSet)]
        deltas.sort(key=self.checkDirection, reverse=True)
        if len(deltas) > 0:
            best = deltas[0]
            self.momentum = [item[0] for item in self.maps.items() if item[1] == best][0]
            return best
        return [-1]

    def checkDirection(self, element):
        return self.dot(self.curr_delta, element)

    def dot(self, pt1, pt2):
        pt1 = self.unit(pt1)
        pt2 = self.unit(pt2)
        return pt1[0] * pt2[0] + pt1[1] * pt2[1]

    def unit(self, point):
        norm = (point[0] ** 2 + point[1] ** 2)
        norm = np.sqrt(norm)
        return point[0] / norm, point[1] / norm

    def run(self):
        colorsCount = 1

        color = self.rescale(self.img, self.pseudoDim)
        collapsed = np.sum(color, axis=2) / 3
        fill = np.argwhere(collapsed < 230)  # color 2-d indices
        fill = np.swapaxes(fill, 0, 1)  # swap to index into color
        RGB = color[fill[0], fill[1], :]

        k_means = KMeans(n_clusters=self.num_colors).fit(RGB)  # Using k-means algorithm to cluster colors
        colors = k_means.cluster_centers_  # The colors that the program will use
        labels = k_means.labels_  # the cluster type of each pixel

        fill = np.swapaxes(fill, 0, 1).tolist()  # swap back to make dictionary
        label_2_index = defaultdict(list)

        for i, j in zip(labels, fill):  # linking cluster label with the color using dictionary
            label_2_index[i].append(j)

        for (i, color) in enumerate(colors):  # iterating over the clustered colors of the image
            self.paint.color = "#" + self.rgb_to_hex((round(color[2]), round(color[1]), round(color[0])))

            print("CHANGING COLOR")


            points = label_2_index[i]  # extracting all points with paint.color

            # making the set all of points with paint.color
            index_tuples = map(tuple, points)
            self.hashSet = set(index_tuples)  # storing all points with paint.color into the set

            # using KNN algorithm to make paths
            self.KDTree = create(points)  # create a model
            self.commands = []  # will be used to store if the pen should be down or up
            self.curr_pos = (pg.size()[0] / 2, pg.size()[1] / 2)  # start at center of the screen
            startPoint = self.translate(self.curr_pos)  # find where the point should be drawn
            self.commands.append(startPoint)
            self.commands.append("UP")

            self.createPath()

            print('Number of colors left is', self.num_colors - colorsCount)
            colorsCount += 1

            self.execute(self.commands)


if __name__ == '__main__':
    Paint()