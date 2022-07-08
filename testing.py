from autocif import *
core = Core()
magellan = Magellan()
microscope = autocif.cycif()


microscope.auto_expose(core, magellan, 50, 6500, ['DAPI'])
