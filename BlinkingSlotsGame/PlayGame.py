from BlinkDetection import Blinking
import threading
import importlib


blinkDetection = Blinking()
module = importlib.import_module('SlotsMachine')

t1 = threading.Thread(target = blinkDetection.start)
t2 = threading.Thread(target = module.main)

t2.start()
t1.start()

t2.join() 
t1.join()