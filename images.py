import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image

path = "C:/Users/ofri/Desktop/תואר שני/ראיה בעזרת למידה עמוקה/examples/%d/%s.png"
f, ax = plt.subplots(4, 6, sharex='col',figsize=(30,20))

coords = [[20,250,450,350],[550,230,1300,350],[500,230,500,350],[650,230,1200,350]]

rects = [patches.Rectangle((50,100),40,30,linewidth=1,edgecolor='w',facecolor='none'),
    patches.Rectangle((50,100),40,30,linewidth=1,edgecolor='w',facecolor='none'),
    patches.Rectangle((50,100),40,30,linewidth=1,edgecolor='w',facecolor='none'),
    patches.Rectangle((50,100),40,30,linewidth=1,edgecolor='w',facecolor='none')]


for i in range(4):

    coord = coords[i]
    p = patches.Rectangle((coord[2], coord[0]), coord[3], coord[1], linewidth=1, edgecolor='w', facecolor='none')
    orig_path = path % (i+1, 'orig')
    low_path = path % (i + 1, 'low')
    deconv_path = path % (i + 1, 'deconv')
    norm_path = path % (i + 1, 'dnorm')
    edsr_path = path % (i + 1, 'edsr')

    orig = Image.open(orig_path).convert("RGB")
    ax[i][0].set_axis_off()
    ax[i][0].imshow(orig)
    ax[i][0].add_patch(p)

    orig2 = plt.imread(orig_path)
    ax[i][1].imshow(orig2[coord[0]:coord[0]+coord[1],coord[2]:coord[2]+coord[3]])
    ax[i][1].set_axis_off()

    low = plt.imread(low_path)
    r = low.shape[0] / orig2.shape[0]
    ax[i][2].imshow(low[int(coord[0]*r):int(coord[0]*r) + int(coord[1]*r), int(coord[2]*r):int(coord[2]*r) + int(coord[3]*r)])
    ax[i][2].set_axis_off()

    edsr = plt.imread(deconv_path)
    r = edsr.shape[0] / orig2.shape[0]
    ax[i][3].imshow(edsr[int(coord[0] * r):int(coord[0] * r) + int(coord[1] * r),
                    int(coord[2] * r):int(coord[2] * r) + int(coord[3] * r)])
    ax[i][3].set_axis_off()

    deconv = plt.imread(deconv_path)
    r = deconv.shape[0] / orig2.shape[0]
    ax[i][4].imshow(deconv[int(coord[0]*r):int(coord[0]*r) + int(coord[1]*r), int(coord[2]*r):int(coord[2]*r) + int(coord[3]*r)])
    ax[i][4].set_axis_off()

    norm = plt.imread(norm_path)
    r = norm.shape[0] / orig2.shape[0]
    ax[i][5].imshow(norm[int(coord[0]*r):int(coord[0]*r) + int(coord[1]*r), int(coord[2]*r):int(coord[2]*r) + int(coord[3]*r)])
    ax[i][5].set_axis_off()


# image = Image.open(img_loc).convert("RGB")
#
# f, (ax1, ax2) = plt.subplots(1, 2, sharex='col')
# ax1.imshow(img, cmap='gray', vmin=0, vmax=255), ax1.set_title('Original+ detected circles')
# ax2.imshow(imageEdges, cmap='gray', vmin=0, vmax=255), ax2.set_title('Canny edges')
#
# circle = []
# for y, x, r in circles:
#     circle.append(plt.Circle((x, y), r, color=(1, 0, 0), fill=False))
#     ax1.add_artist(circle[-1])
plt.show()
print('done!')
