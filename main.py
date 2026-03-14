import machine
import time
import gc
import cv2 as cv
from ulab import numpy as np
import micropython

frame_buffer = bytearray(80000)
mask_buffer = bytearray(40000)


spi = None
i2c = None
cs = None

@micropython.viper
def redmask(src: ptr8, dst: ptr8, pixels: int): # src=camera bytes, dst=output mask, pixels=how many to check
    for i in range(pixels):
        idx = i*2 # each pixel is 2 bytes
        c16 = (src[idx] << 8) | src[idx + 1]

        r = ((c16 >> 11) & 0x1F) * 8
        g = ((c16 >> 5) & 0x3F) * 4
        b = (c16 & 0x1F) * 8
        
        if r > (g+50) and r > (b+50) and r > 100:
            dst[i] = 255 # white in mask
        else:
            dst[i] = 0 # black in mask

@micropython.viper
def greenmask(src: ptr8, dst: ptr8, pixels: int):
    for i in range(pixels):
        idx = i * 2
        c16 = (src[idx] << 8) | src[idx + 1]

        r = ((c16 >> 11) & 0x1F) * 8
        g = ((c16 >> 5) & 0x3F) * 4
        b = (c16 & 0x1F) * 8

        if g > (r + 30) and g > (b + 20) and g > 60:
            dst[i] = 255
        else:
            dst[i] = 0

@micropython.viper
def bluemask(src: ptr8, dst: ptr8, pixels: int):
    for i in range(pixels):
        idx = i * 2
        c16 = (src[idx] << 8) | src[idx + 1]

        r = ((c16 >> 11) & 0x1F) * 8
        g = ((c16 >> 5) & 0x3F) * 4
        b = (c16 & 0x1F) * 8

        if b > (r + 40) and b > (g + 30) and b > 80:
            dst[i] = 255
        else:
            dst[i] = 0

def setup():
    global spi, i2c, cs
    spi = machine.SPI(0,baudrate=4000000,sck=machine.Pin(2),mosi=machine.Pin(3),miso=machine.Pin(4))
    i2c = machine.I2C(0,sda=machine.Pin(8),scl=machine.Pin(9),freq=50000)

    time.sleep_ms(100) # wait 100ms for the buses to stabilize

    cs = machine.Pin(5, machine.Pin.OUT)
    cs.value(1)

def w_reg(reg, val):
    cs.value(0)
    spi.write(bytes([reg | 0x80, val])) # set write bit
    cs.value(1)

def r_reg(reg):
    cs.value(0)
    spi.write(bytes([reg & 0x7F])) # set read bit
    val = spi.read(1)[0] # read 1 byte back, grab the value from the returned list
    cs.value(1)
    return val

def init_cam():
    w_reg(0x07, 0x80) # set hardware reset bit
    time.sleep_ms(100)
    w_reg(0x07, 0x00) # clear  bit
    time.sleep_ms(100)

    regs = [
        (0xff, 0x01), (0x12, 0x80), (0xff, 0x00), (0x2c, 0xff), (0x2e, 0xdf),
        (0xff, 0x01), (0x11, 0x01), (0x12, 0x00), (0x3c, 0x32), (0xff, 0x00),
        (0x44, 0x00), (0x12, 0x40), (0x13, 0x00), (0x11, 0x03), (0x14, 0x00),
        (0x0c, 0x00), (0x3e, 0x00), (0x0d, 0x00), (0xff, 0x01), (0x12, 0x40),
        (0x47, 0x01), (0x4b, 0x09), (0x10, 0x00), (0xff, 0x00), (0xda, 0x08),
        (0xd7, 0x03), (0xdf, 0x02), (0x33, 0x40), (0x3c, 0x00), (0xba, 0x01),
        (0xbb, 0x20), (0xff, 0x00), (0xe0, 0x04), (0x12, 0x00), (0x5a, 0x50), (0x3c, 0x3c)]

    for r, v in regs:
        i2c.writeto_mem(0x30, r, bytes([v]))
        if r == 0x12 and v == 0x80:
            time.sleep_ms(50)
        else:
            time.sleep_ms(2)

def findblobs(targetcolor):
    w_reg(0x04, 0x01) # clear flag
    w_reg(0x04, 0x02) # capture new frame

    start_time = time.ticks_ms()
    while not (r_reg(0x41) & 0x08): # capture is done when bit 3 is set
        if time.ticks_diff(time.ticks_ms(), start_time) > 1000: # if we've waited more than 1 second...
            return # ...camera is stuck, give up on this frame

    size = r_reg(0x42)|(r_reg(0x43) << 8)|((r_reg(0x44) & 0x7f) << 16)

    if size < 5000: # if the image is too small, skip
        w_reg(0x04, 0x01)
        return

    width = 320
    height = 240

    crop_height = height // 2 # only get middle slit of frame
    start_row = (height - crop_height) // 2

    ptot = width * crop_height

    cs.value(0) # listen
    spi.write(bytes([0x3C])) # burst read
    spi.readinto(memoryview(frame_buffer)[0:start_row * width * 2]) # read the top rows not needed
    spi.readinto(memoryview(frame_buffer)[0:crop_height * width * 2]) # overwrite with middle rows
    cs.value(1) # the end

    if targetcolor == "RED":
        redmask(frame_buffer, mask_buffer, ptot)
    elif targetcolor == "GREEN":
        greenmask(frame_buffer, mask_buffer, ptot)
    elif targetcolor == "BLUE":
        bluemask(frame_buffer, mask_buffer, ptot)

    maskedimg = np.frombuffer(memoryview(mask_buffer)[0:ptot], dtype=np.uint8).reshape((crop_height, width))

    contours = cv.findContours(maskedimg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE )[0] #outermost contour

    best = None
    max_area = 0

    for i in contours:
        area = cv.contourArea(i)

        if area > 500:
            perimeter = cv.arcLength(i, True) # treat as closed shape
            if perimeter == 0: continue # avoiding divide by zero

            radius = perimeter / (2 * 3.14159) # estimate radius as if the blob were a perfect circle
            if radius > 0:
                roundness = area / (3.14159 *(radius**2))

                if 0.79 < roundness < 1.5 and area > max_area:
                    max_area = area
                    M = cv.moments(i)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"]) + start_row # shift
                        best = (targetcolor, cx, cy, max_area, roundness)

    if best != None: #if there is a blob
        color, cx, cy, final_size, roundness = best
        mem = gc.mem_alloc() // 1000
        memtot = (gc.mem_free() // 1000) + mem
        print(f"target: {color} | x:{cx} y:{cy} | size:{final_size} | roundness:{roundness:.2f} | mem:{mem}kb / {memtot}kb")
    
    del maskedimg #FREE RAM
    del contours
    gc.collect()

def boot_camera():
    while True:
        setup() # set up pins
        init_cam() # config registers
        print("camera working...")
        time.sleep(.5)
        return
            
boot_camera() # initialize the camera — blocks here until it successfully starts

color_list = ["GREEN", "RED", "BLUE"] # the three colors to cycle through in order
i = 0 # start at index 0 = "RED"

while True:
    try:
        target = color_list[i] # pick the current color to hunt for
        findblobs(target) # capture a frame and look for that color
        i = (i + 1) % 3

        gc.collect()

    except MemoryError:
        print("memory low...")
        gc.collect()
        time.sleep(.5)
