import ast
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
from PIL import Image
from typing import List, Tuple
from sklearn.cluster import KMeans
import sys

from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import (
        QApplication,
        QDialog,
        QFileDialog,
        QCheckBox,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QDoubleSpinBox,
        QSpinBox,
        QVBoxLayout,
        QWidget
    )

from PyQt5.QtGui import QPalette, QColor

# import pdb; pdb.set_trace()

####### The following are some PyQt5 functions/classes required for the program's GUI

def get_number_of_clusters() -> int:

    win = QDialog()

    layout = QVBoxLayout()

    label = QLabel("Please select the number of colours")
    # label.setAlignment(Qt.AlignCenter)
    layout.addWidget(label)

    spinbox = QSpinBox()

    spinbox.setRange(8, 64)
    spinbox.setValue(16)
    
    layout.addWidget(spinbox)

    dlg = QMessageBox()
    dlg.setStandardButtons(QMessageBox.Ok)

    layout.addWidget(dlg)

    win.setLayout(layout)
    win.setGeometry(100,100,200,100)
    win.show()

    button = dlg.exec()

    if button == QMessageBox.Ok:

       return spinbox.value()

def choose_file():

    qfd = QFileDialog()
    qfd.setFileMode(QFileDialog.AnyFile)
    qfd.setNameFilter("*.png *.xpm *.tif *.jpg")

    if qfd.exec_():
        file_name = qfd.selectedFiles()

    return file_name

def set_pixel_area() -> Tuple[float,float]:

    win = QDialog()

    layout = QHBoxLayout()

    label = QLabel("Set pixel length scales")
    label.setAlignment(Qt.AlignCenter)
    layout.addWidget(label)

    xslider = QDoubleSpinBox()
    xslider.setValue(0.005)
    # xslider.setRange(0.001,0.010)
    # xslider.setMaximum(0.010)
    xslider.setDecimals(4)
    xslider.setSingleStep(0.0002)
    xslider.setSuffix(" mm")
    # xslider.setTickPosition(QSlider.TicksBelow)
    # xslider.setTickInterval(0.001)

    layout.addWidget(xslider)

    yslider = QDoubleSpinBox()
    yslider.setValue(0.005)
    # yslider.setRange(0.001,0.010)
    yslider.setDecimals(4)
    yslider.setSingleStep(0.0002)
    yslider.setSuffix(" mm")
    # yslider.setTickPosition(QSlider.TicksRight)
    # yslider.setTickInterval(0.001)

    layout.addWidget(yslider)

    dlg = QMessageBox()
    dlg.setStandardButtons(QMessageBox.Ok)

    layout.addWidget(dlg)

    win.setLayout(layout)
    win.setGeometry(100,100,200,100)
    win.show()

    button = dlg.exec()

    if button == QMessageBox.Ok:

        delta_x = xslider.value()
        delta_y = yslider.value()

    return (delta_x, delta_y)

class Color(QWidget):

    def __init__(self, r, g, b):

        super(Color, self).__init__()

        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(r, g, b))
        self.setPalette(palette)

def display_colours(colors, blues, purples, background, blacks):

    win = QDialog()

    win.setWindowTitle("Automatic Colour Classification")

    dialogLayout = QVBoxLayout()

    grid = QGridLayout()

    grid.addWidget(QLabel("Blues"), 0, 0)
    grid.addWidget(QLabel("Purples"), 0, 1)
    grid.addWidget(QLabel("Background"), 0, 2)
    grid.addWidget(QLabel("Blacks"), 0, 3)

    for n, color_index in enumerate(blues):

        r, g, b = colors[color_index]
        color_label = Color(r, g, b)
        grid.addWidget(color_label, n, 0)

    for n, color_index in enumerate(purples):

        r, g, b = colors[color_index]
        color_label = Color(r, g, b)
        grid.addWidget(color_label, n, 1)

    for n, color_index in enumerate(background):

        r, g, b = colors[color_index]
        color_label = Color(r, g, b)
        grid.addWidget(color_label, n, 2)

    for n, color_index in enumerate(blacks):

        r, g, b = colors[color_index]
        color_label = Color(r, g, b)
        grid.addWidget(color_label, n, 3)

    dialogLayout.addLayout(grid)

    dlg = QMessageBox()
    dlg.setWindowTitle("Is this ok?")
    dlg.setStandardButtons(QMessageBox.Yes)
    dlg.addButton(QMessageBox.No)
    dlg.setDefaultButton(QMessageBox.Yes)

    dialogLayout.addWidget(dlg)

    win.setLayout(dialogLayout)
    win.show()

    button = dlg.exec()

    if button == QMessageBox.Yes:

        return

    elif button == QMessageBox.No:

        print(f'Houston, we have a problem!')



def get_color_list(colors, title, noncheckable = None) -> List[bool]:

    win = QDialog()

    win.setWindowTitle(title)

    dialogLayout = QVBoxLayout()
    grid = QGridLayout()

    n_colors = len(colors)

    for n, (r, g, b) in enumerate(colors):

        color_label = Color(r, g, b)
        color_button = QCheckBox(parent = win, text=repr(n))

        if noncheckable is not None:

            if noncheckable[n]:

                color_button.setCheckable(False)
                color_button.setDown(True)

        grid.addWidget(color_label, n, 0)
        grid.addWidget(color_button, n, 1)

    dialogLayout.addLayout(grid)

    dlg = QMessageBox()
    dlg.setStandardButtons(QMessageBox.Ok)

    dialogLayout.addWidget(dlg)

    win.setLayout(dialogLayout)
    win.setGeometry(100,100,200,100)
    win.show()

    button = dlg.exec()

    n_color = 0

    selected_colours = [False for i in range(n_colors)]

    if button == QMessageBox.Ok:

       for widget in win.children():

           if isinstance(widget, QCheckBox):

               if widget.isChecked():

                   selected_colours[n_color] = True

               n_color += 1

    return selected_colours

####### The following are some PyQt5 functions/classes required for the program's GUI

app = QApplication(sys.argv)

# from plot_colortable import plot_colortable 
# from Widgets import MainWindow

n_clusters = get_number_of_clusters()

# n_clusters = int(input("Please enter the desired number of colours: "))
# file_name = input("Enter the file name of the image to analyse: ")
# file_name += '.tif'

image_file = choose_file()
file_name = image_file[0]

# pixel_area = input("Enter the pixel area (default 0.005x0.005 mm^2): ")

# if pixel_area == '':
#    pixel_area = 0.005 * 0.005

delta_x, delta_y = set_pixel_area()
pixel_area = delta_x * delta_y

image = np.asarray(Image.open(file_name))

n_pixels_y, n_pixels_x, _ = image.shape

image_data = image.reshape((-1,3))

kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0)

kmeans.fit(image_data)

values, counts = np.unique(kmeans.labels_, return_counts=True)

colour_labels = np.rint(kmeans.cluster_centers_).astype(int)
colours = kmeans.cluster_centers_/255.

dark = []

for n, colour in enumerate(kmeans.cluster_centers_):
    gray = 0.299 * colour[0] + 0.587 * colour[1] + 0.114 * colour[2]
    dark.append(1./gray)

colour_intensity = np.array(dark)

names = []
for i, label in enumerate(kmeans.cluster_centers_):
    names.append("Colour #: " + repr(i))

counts = counts / counts.sum()

labels = []
for n, count in enumerate(counts):
    label = f"{count*100.:3.1f} % #: {n}"
    labels.append(label)

pixel_colours = (colours*255.).astype(int)

# figure = plot_colortable(colors, names)

pixel_cluster = kmeans.fit_predict(image_data) # this classifies each pixel

# create an figure showing the original image, the clusters of colours (as a pie chart) 
# and the colour labels

fig, (ax0, ax1) = plt.subplots(1,2)

ax0.imshow(image)

wedges, texts = ax1.pie(counts, labels=labels, colors = colours)

# ax2.legend(wedges, labels,
#           title="Colours",
#           loc="center left")
          # bbox_to_anchor=(1.2, 0, 0.5, 1))

# ax2.axis('off')

plt.show()

# do an automatic color classification and check with user if this is fine

blue_colours = []
purple_colours = []
background_colours = []
black_colours = []

for n, (r, g, b) in enumerate(pixel_colours):

    if r >= 180 and g >= 180 and b >= 180:

        background_colours.append(n)

    elif r <= 30 and g <= 30 and b <= 30:

        black_colours.append(n)

    elif (b/r) >= 1.3:

        blue_colours.append(n)

    else:

        purple_colours.append(n)

display_colours(pixel_colours, blue_colours, purple_colours, background_colours, black_colours)

title = "Check the 'blue colours' as identified in the pie chart"
blue_colour_index = get_color_list(pixel_colours, title)
#blue_colours = ast.literal_eval(input("blue colours (e.g. [1, 2, 5]): "))
blue_colours = []
other_colours = [False for i in range(n_clusters)]
for i in range(n_clusters):
    if blue_colour_index[i]: 
        blue_colours.append(i)
    else:
        other_colours[i] = True

# establish an 'intensity' order among the blue colours

blue_colour_intensity = np.zeros((n_clusters), dtype=float)

for n in range(n_clusters):
    if n in blue_colours:
       blue_colour_intensity[n] = colour_intensity[n]

blue_colour_intensity /= np.max(blue_colour_intensity)

title = "Check the 'other sample colours' (i.e. exclude background colors)"
# purple_colours = ast.literal_eval(input("purple colours (e.g. [3, 4, 6]): "))

purple_colour_index = get_color_list(pixel_colours, title, blue_colour_index)

purple_colours = []
for i in range(n_clusters):
    if purple_colour_index[i]:
       purple_colours.append(i)

app.quit()

s_counts = []
s_labels = []
s_colours = []

for n, (count, colour) in enumerate(zip(counts, colours)):

    if n in blue_colours or n in purple_colours:

       s_counts.append(count)
       s_colours.append(colour)
       s_labels.append(f'{n}')

sample_counts = np.array(s_counts)
sample_colours = np.array(s_colours)

sample_labels = []
for n, label in enumerate(s_labels):

    percentage = 100. * sample_counts[n] / float(sample_counts.sum())

    sample_labels.append(label + f' {percentage:2.3f}')
 
# with the above information we can estimate the level of compactness of the color at each pixel

n_pixels = n_pixels_x * n_pixels_y
compactness = np.zeros((n_pixels), dtype = float)
sample = -1 * np.ones((n_pixels), dtype = int)

for i in range(n_pixels):

    if pixel_cluster[i] in blue_colours or pixel_cluster[i] in purple_colours:
       sample[i] = pixel_cluster[i]

    if pixel_cluster[i] in blue_colours:

       nx = i % n_pixels_x
       ny = int( i / n_pixels_x ) % n_pixels_y

       if (nx > 0 and nx < (n_pixels_x-1)) and \
          (ny > 0 and ny < (n_pixels_y-1)):

          for ix in range(-1,2):
              for iy in range(-1,2):

                  # if ix == iy == 0: continue

                  j = (nx + ix) + n_pixels_x * ( ny + iy )

                  if pixel_cluster[j] in blue_colours:

                     compactness[i] += blue_colour_intensity[pixel_cluster[j]]

max_compactness = np.max(compactness)

compactness /= max_compactness

c_values, c_counts = np.unique(sample, return_counts=True)

sort_index = np.argsort(c_values)

sample_values, n_sample_pixels = c_values[sort_index[1:]], c_counts[sort_index[1:]]

for n, value in enumerate(sample_values):

    percentage = 100. * n_sample_pixels[n] / n_sample_pixels.sum()
    area = n_sample_pixels[n] * pixel_area

    txt = f'Colour {value:3d}  Percentage: {percentage: 5.3f},  Area: {area:5.3f} (mm^2)'

    print(txt)

sample_total_area = n_sample_pixels.sum() * pixel_area
total_area = n_pixels * pixel_area
fractional_area = 100. * sample_total_area / total_area
txt = f'\nTotal area occupied by sample: {sample_total_area:5.3f}'
print(txt)
txt = f'\nFractional area occupied by sample: {fractional_area:5.3f}'
print(txt)

cmap = plt.get_cmap('plasma')

compactness_map = cmap(compactness.reshape((n_pixels_y, n_pixels_x)))

new_im = []

for index in kmeans.labels_:
    new_im.append(pixel_colours[index])

new_image = np.array(new_im).reshape((n_pixels_y,n_pixels_x,3))

fig, axs = plt.subplots(2,2)

axs[0,0].imshow(image)
axs[0,0].axis("off")

axs[0,1].pie(sample_counts, labels=sample_labels, colors = sample_colours)

axs[1,0].imshow(new_image)
axs[1,0].axis("off")

axs[1,1].imshow(compactness_map)
axs[1,1].axis("off")

plt.show()

fig, (ax0, ax1) = plt.subplots(1,2, constrained_layout=True)
ax0.imshow(image, aspect="auto")
ax0.axis('off')
pos = ax1.imshow(compactness_map,cmap=cmap, aspect="auto")
ax1.axis('off')
ax2 = ax1.twinx()
ax2.tick_params(which="both", right=False, labelright=False)
plt.colorbar(pos,ax=ax2)
# ax1.figure.colorbar(pos)
plt.show()

# finally we are going to subdivide the compactness into four bands [0:0.25], [0.25:0.50], [0.5:0.75], etc

band_1 = 0
band_2 = 0
band_3 = 0
band_4 = 0

n_finite_compactness = 0

for i in range(n_pixels):

    value = compactness[i]

    if sample[i] > 0:

       if 0.0 < value and value < 0.25:
          band_1 += 1
       elif value < 0.50:
          band_2 += 1
       elif value < 0.75:
          band_3 += 1
       elif value <= 1.:
          band_4 += 1

n_sample = n_sample_pixels.sum()

compactness_band_1 = 100 * float(band_1) / float(n_sample)
compactness_band_2 = 100 * float(band_2) / float(n_sample)
compactness_band_3 = 100 * float(band_3) / float(n_sample)
compactness_band_4 = 100 * float(band_4) / float(n_sample)

c_band_1_area = band_1 * pixel_area
c_band_2_area = band_2 * pixel_area
c_band_3_area = band_3 * pixel_area
c_band_4_area = band_4 * pixel_area

print('\n')
txt = f'0 % <= Compactness < 25 %  Percentage of sample: {compactness_band_1: 5.3f}, \
      Area: {c_band_1_area:5.3f} (mm^2)'
print(txt)
txt = f'25 % <= Compactness < 50 %  Percentage of sample: {compactness_band_2: 5.3f}, \
      Area: {c_band_2_area:5.3f} (mm^2)'
print(txt)
txt = f'50 % <= Compactness < 75 %  Percentage of sample: {compactness_band_3: 5.3f}, \
      Area: {c_band_3_area:5.3f} (mm^2)'
print(txt)
txt = f'75 % <= Compactness <= 100 %  Percentage of sample: {compactness_band_4: 5.3f}, \
      Area: {c_band_4_area:5.3f} (mm^2)'
print(txt)

