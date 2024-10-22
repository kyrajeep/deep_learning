from multiprocessing import Process, freeze_support
import os
import time

processes = []
num_processes = os.cpu_count()

def square_nums():
    for i in range(100):
        i = i
        time.sleep(0.1)
#create processes
def main():
    for i in range(num_processes):
        p = Process(target = square_nums)
        processes.append(p)
    
    for p in processes:
        p.start()
    
    for p in processes:
        p.join()
if __name__ == '__main__':
    freeze_support()
    main()