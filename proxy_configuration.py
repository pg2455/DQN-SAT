#! /usr/bin/env python2
import sys 
if "linux" in sys.platform:
    from configuration import *
else:
    from configuration_2 import *
