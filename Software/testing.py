import numpy as np


class testing:

    def __init__(self, variable):
        self.attribute = variable

a = testing('first')
a.attribute = 'second'

print(a.attribute)