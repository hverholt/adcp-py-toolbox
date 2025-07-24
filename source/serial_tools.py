import serial
import time


# Control flags
stop_thread = False
record_data = False


def send_break(ser):
    ser.write(b'@@@@@@')
    time.sleep(0.15)
    ser.write(b'K1W%!Q')
    time.sleep(0.4)
    ser.write(b'K1W%!Q')


def record_data_to_file(data, savepath, clean_data=True):
    if clean_data:
        lines = data.split('\n')
        lines = [line for line in lines if line.strip().startswith('$')]
        data = '\n'.join(lines)
    
    with open(savepath, 'a') as file:
        file.write(data)
