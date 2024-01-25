from multiprocessing import Process
import os
import time

import remi.gui as gui
from remi import start, App


class MyApp(App):
    def __init__(self, *args):

        self.container = gui.VBox(width=120, height=100)
        self.lbl = gui.Label('Hello world!')
        self.bt = gui.Button('Press me!')

        # setting the listener for the onclick event of the button
        self.bt.set_on_click_listener(self.on_button_pressed)

        # appending a widget to another, the first argument is a string key
        self.container.append(self.lbl)
        self.container.append(self.bt)

        self.printer = "printer"
        print(self.printer)
        super(MyApp, self).__init__(*args)




    def main(self):
        print("main is looped")
        container = self.container

        return container

    # listener function
    def on_button_pressed(self, widget):
        self.lbl.set_text('Button pressed!')
        self.bt.set_text('Hi!')
        print("button pressed")
        self.printer = "New"
        self.container = gui.VBox(width=220, height=100)
        self.lbl = gui.Label('SUB')
        # appending a widget to another, the first argument is a string key
        self.container.append(self.lbl)
        #super(MyApp, self).__init__(*args)






# starts the webserver
#start(MyApp)
start(MyApp, address='127.0.0.2', port=8081, multiple_instance=True, enable_file_cache=True, update_interval=0.0001, start_browser=True)

'''
def splitfun2(feature, Threshold="optional"):
    print(feature)

class thresholdsplitter():
    def __init__(self, Threshold):
        self.Threshold=Threshold
    def doit(self, feature):
        print(feature)
        print("Threshold yeah")
        print(self.Threshold)
        newval = feature + self.Threshold
        return newval

    def redoer(self):
        newval1 = self.newval
        print (newval1)
x = thresholdsplitter(1)
x.doit(10)
x.redoer()



def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def f(name):
    info('function f')
    print('hello', name)

def abc(name):
    # Wait for 5 seconds
    time.sleep(5)
    print('hello', name)


info('main line')
p = Process(target=abc, args=('bob',))
p1 = Process(target=f, args=("bib",))

if __name__ == '__main__':

    p.start()
    p1.start()
    p.join()
    p1.join()
'''