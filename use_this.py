from autocif import *
import serial
import time

mm_app_path = 'C:/Program Files/Micro-Manager-2.0gamma'
config_file = 'C:/Users/CyCIF PC/Desktop/backup config files/auto_cycif.cfg'
#start_headless(mm_app_path, config_file, timeout=15000)

# arduino = serial.Serial(port='COM5', baudrate=9600, timeout=5)
# time.sleep(7)

bridge = Bridge()
core = bridge.get_core()

global f_start
global f_end
global f_step


f_start = 5800    # original value: -15328
f_end = 5900      # original value: -15368
f_step = 10         # original value: -8
brenner = []


# cycif.syr_obj_switch(0)  # binary state. 0 is pool, 1 is pipettor

def focus(image, metadata):
    # append z
    z = f_start + metadata['Axes']['z'] * f_step  ### is z change in z here?

    # focus_score(image,z) outputs f_score and z
    #focus_score(image, z)

    # image in numpy array and z in integer with micron units
    # output should something like this:
    # [[fscore1,z1],[fscore2,z2],...]
    brenner.append(focus_score(image, z))

    # null the image to save memory
    # plt.imshow(image)
    # plt.show()
    # image[:100, :100] = 0
    image[:, :] = []

    return


def focus_score(image, z):
    # focus score using Brenner's score function
    # Note: Uniform background is a bit mandatory
    a = image[2:, :]
    b = image[:-2, :]
    c = a - b
    c = c * c
    f_score_shadow = c.sum()

    # check to see if this works. and it does.
    # print(f_score_shadow)
    # print(z)
    return f_score_shadow, z


def gauss(x, A, x0, sig, y0):
    # fit to a gaussian
    y = y0 + (A * np.exp(-((x - x0) / sig) ** 2))
    return y


def autofocus_fit():
    brenner_temp = np.array(brenner) # brenner's a global variable. there's no reason to re-call it
    f_score_temp = brenner_temp[:, 0]
    z = brenner_temp[:, 1]

    # print(brenner_temp)
    # print(f_score_temp)
    # print(z)

    # curve fitted with bounds relating to the inputs
    # let's force it such that z_ideal is within our range.
    parameters, covariance = curve_fit(gauss, z, f_score_temp,
                                       bounds=[(min(f_score_temp) / 4, min(z), 0, 0),
                                                (max(f_score_temp) * 4, max(z), (max(z)-min(z)), max(f_score_temp))])

    # a previous iteration of bounds that are more general
    # bounds = [(min(f_score_temp) / 4, min(z) / 2, f_start / 2, 0),
    #           (max(f_score_temp) * 4, max(z) * 2, f_start * 2, max(f_score_temp))])

    print('Z focus is located at: (microns)')
    print(parameters[1])

    # for a sanity check, let's plot this
    fit_f_score_gauss = gauss(z, *parameters)
    plt.plot(z, f_score_temp, 'o', label='data')
    plt.plot(z, fit_f_score_gauss, '-', label='fit')
    plt.title(['fstart,fend,fstep: ',str(f_start), ' ',str(f_end),' ',str(f_step)])
    plt.legend()
    plt.grid()
    plt.show()
    return parameters[1]

######################

with Acquisition(directory='C:/Users/CyCIF PC/Desktop/test_images',
                 name='z_stack_DAPI',
                 show_display=False,
                 image_process_fn=focus) as acq:
    events = multi_d_acquisition_events(channel_group='Color',
                                        channels=['DAPI'],
                                        z_start=f_start,
                                        z_end=f_end,
                                        z_step=f_step,
                                        order='zc')
    acq.acquire(events)

## HERE IS WHERE FITS GO
autofocus_fit()


# print(z_ideal)

# core.set_position(-2000)

# cycif.order_execute([61, 60], arduino)
# cycif.prim_secondary_cycle(0, 1, arduino)
# cycif.post_acquistion_cycle(2)


# cycif.syr_obj_switch(1)
# list_of_orders = [70, 11, 21, 33, 20, 61]
# cycif.order_execute(list_of_orders, arduino)
# time.sleep(3600)
# list_of_orders = [49, 71, 85, 70, 43, 71, 60]
# cycif.order_execute(list_of_orders, arduino)
# cycif.syr_obj_switch(0)
