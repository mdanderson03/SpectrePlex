from autocyplex import *


os.chdir(r'E:\folder_structure\np_arrays')

images = full_array = np.load('z_stack.npy', allow_pickle=False)
imwrite('test_save.tif', images[5][5], photometric='minisblack')
io.imshow(images[5][5])
io.show()


