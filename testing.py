from autocif import *
microscope = autocif.cycif()


microscope.surf2focused_surf(core, magellan, ['DAPI'])
microscope.micro_magellan_acq()


