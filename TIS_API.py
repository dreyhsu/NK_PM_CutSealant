import ctypes 
import tisgrabber as tis
import cv2
import numpy as np
import time
import os

def IC_Initial(DLL_Location) : 
    
    ic = ctypes.cdll.LoadLibrary(DLL_Location)
    tis.declareFunctions(ic)
    ic.IC_InitLibrary(0)       
    
    grabbers = []
    device_names = []
    devicecount = ic.IC_GetDeviceCount()
    for i in range(0, devicecount):
        print("Device count = ", devicecount)
        print("Device {}".format(tis.D(ic.IC_GetDevice(i))))
        uniquename = tis.D(ic.IC_GetUniqueNamefromList(i))
        device_names.append(uniquename)
        print("Unique Name : {}".format(uniquename))
        if os.path.isfile(uniquename + ".xml"):
            hGrabber = ic.IC_CreateGrabber()
            ic.IC_LoadDeviceStateFromFile(hGrabber, tis.T(uniquename + ".xml"))
            grabbers.append(hGrabber)
        else:
            hGrabber = ic.IC_ShowDeviceSelectionDialog(None)                   
            ic.IC_SaveDeviceStateToFile(hGrabber, tis.T(uniquename + ".xml"))
            grabbers.append(hGrabber)
    return ic, grabbers, device_names

def IC_Release(ic, grabbers):
    for grabber in grabbers:
        if(ic.IC_IsDevValid(grabber)):
            ic.IC_StopLive(grabber)
            ic.IC_ReleaseGrabber(grabber)
            

def IC_StartCapture(ic, grabbers):      
    for grabber in grabbers:
        if(ic.IC_IsDevValid(grabber)):
            ic.IC_StartLive(grabber, 0)
    return ic, grabbers

def IC_GetImage(ic, grabbers, camera_idx):
    if(ic.IC_IsDevValid(grabbers[camera_idx])):            
        if ic.IC_SnapImage(grabbers[camera_idx], 2000) == tis.IC_SUCCESS:
            # Declare variables of image description
            Width = ctypes.c_long()
            Height = ctypes.c_long()
            BitsPerPixel = ctypes.c_int()
            colorformat = ctypes.c_int()
            # Query the values of image description
            ic.IC_GetImageDescription(grabbers[camera_idx], Width, Height,
                                        BitsPerPixel, colorformat)

            # Calculate the buffer size
            bpp = int(BitsPerPixel.value / 8.0)
            buffer_size = Width.value * Height.value * BitsPerPixel.value

            # Get the image data
            imagePtr = ic.IC_GetImagePtr(grabbers[camera_idx])

            imagedata = ctypes.cast(imagePtr,
                                    ctypes.POINTER(ctypes.c_ubyte *
                                                    buffer_size))
                    
            # Create the numpy array
            image = np.ndarray(buffer=imagedata.contents,
                                dtype=np.uint8,
                                shape=(Height.value,
                                        Width.value,
                                        bpp))
            #framecount = framecount+1
            #nowTime = time.time();
            #timediff = nowTime - eTime
            #TotalCount  = TotalCount+1
            #if timediff>1:
            #    fps = framecount/timediff
            #    print('Camera1 frame rate ', fps )
            #    eTime = nowTime
            #    framecount=0
            return image               
        else:
            print("No frame received in 2 seconds.")
            return 0
    else:
        #ic.IC_MsgBox(tis.T("No device opened"), tis.T("Simple Live Video"))
        print("Camera1 not opened")
        return 0
     





