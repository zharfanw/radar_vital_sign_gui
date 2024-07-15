from threading import Thread, Event
from time import sleep

event = Event()

def modify_variable(var):
    while True:
        
        print(var)
        sleep(1)
        if event.is_set():
            break
        
    print('Stop printing')


my_var = [1, 2, 3]
t = Thread(target=modify_variable, args=(my_var, ))
t.start()
while True:
    try:
        for i in range(len(my_var)):
            my_var[i] += 1
        sleep(.5)
    except KeyboardInterrupt:
        event.set()
        break
t.join()
print(my_var)