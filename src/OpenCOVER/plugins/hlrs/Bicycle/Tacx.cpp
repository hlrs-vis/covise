/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "Tacx.h"
#include "UDPComm.h"
#include <config/CoviseConfig.h>

static float zeroAngle = 1152.;

int Tacx::usbOpenDevice(libusb_device_handle **device, int vendor, const char *vendorName, int product, const char *productName)
{
    libusb_device_handle *handle = NULL;
    int errorCode = USB_ERROR_NOTFOUND;
    static int didUsbInit = 0;
    int ret;
    if (!didUsbInit)
    {
        didUsbInit = 1;
        int r;
        r = libusb_init(NULL);
        if (r < 0)
                return r;
    }

    handle = libusb_open_device_with_vid_pid(NULL, 0x16c0, 0x0763);
    if (handle==NULL)
    {
        fprintf(stderr, "Error finding USB device\n");
        return -1;
    }
    ret = libusb_claim_interface(handle, 2);
    if (ret < 0)
    {
        fprintf(stderr, "Error claiming interface: %s\n", libusb_error_name(ret));
        return -1;
    }
    if (handle)
    {
        memset(tmp, 0, sizeof(tmp));
        tmp[0] = 2;
        int bytesTransferred=0;
        ret = libusb_bulk_transfer(handle, LIBUSB_ENDPOINT_OUT, tmp, 4,&bytesTransferred, 5000);
        if (ret < 0 || bytesTransferred!=4)
        {
            printf("error writing:\n%s %d\n", libusb_error_name(ret),bytesTransferred);
            libusb_close(handle);
            handle = NULL;
            return -1;
        }
        else
        {
            printf("success: bulk write %d bytes\n", bytesTransferred);
        }
    }

    *device = handle;

    return 0;
}

Tacx::Tacx()
    : udp(NULL)
{
    handle = NULL;
    n = 0;
    on = 1;
    init();
}
Tacx::~Tacx()
{
    delete udp;
}

void Tacx::init()
{
    if (covise::coCoviseConfig::isOn("Bicycle.udp", true))
    {
        const std::string host = covise::coCoviseConfig::getEntry("value", "Bicycle.serverHost", "141.58.8.174");
        unsigned short localPort = covise::coCoviseConfig::getInt("Bicycle.localPort", 31445);
        unsigned short serverPort = covise::coCoviseConfig::getInt("Bicycle.serverPort", 31444);
        std::cerr << "Tacx config: UDP: serverHost: " << host << ", localPort: " << localPort << ", serverPort: " << serverPort << std::endl;
        udp = new UDPComm(host.c_str(), serverPort, localPort);
    }
    else
    {
        //usb_init();             // initialize libusb
        if (usbOpenDevice(&handle, USBDEV_Tacx_VENDOR, "Tacx?VR", USBDEV_VRInterface_PRODUCT, "VR-Interface") != 0)
        {
            fprintf(stderr, "Could not find USB device \"SPI-USB\" with vid=0x%x pid=0x%x\n", USBDEV_Tacx_VENDOR, USBDEV_VRInterface_PRODUCT);

            // not open, thus don't close itusb_close(handle);
            nBytes = -5;
            return;
        }
    }

    nBytes = 0;

    return;
}

void Tacx::update()
{
    if (udp)
    {
        int status = udp->receive(&vrdata, 64);
        if (status == 64)
        {
/*            fprintf(stderr, "\r");
            fprintf(stderr, "Tasten: %1d ", vrdata.tasten);
            fprintf(stderr, "Lenkwinkel: %6d ", vrdata.Lenkwinkel);
            fprintf(stderr, "Drehzahl: %6d ", vrdata.drehzahl);
            fprintf(stderr, "Drehzahl: %6f ", getSpeed());
            fprintf(stderr, "Angle: %6f ", getAngle());
            fprintf(stderr, "Trittfrequenz: %6d ", vrdata.trittfrequenz);*/
            //fprintf(stderr,"ti: %1d ",vrdata.trittfreqenzimpuls);
            /*	unsigned char *tmpc = (unsigned char *)&vrdata;
		int i;
		fprintf(stderr,"\r");
		for(i = 0;i<64;i++)
		{
		fprintf(stderr,"%2x ",tmpc[i]);
		}*/

            //fprintf(stderr,"\n");
            vrdataout.unknown0 = 0x00010801;
            //vrdataout.force = 0xF959;
            vrdataout.unknown1 = 0;
            vrdataout.unknown2 = 0x05145702;
            ret = udp->send(&vrdataout, 12);
            if (ret != 12)
            {
                std::cerr << "Tacx: send: err=" << ret << std::endl;
            }
        }
        else if (status == -1)
        {
            std::cerr << "Tacx::update: error" << std::endl;
        }
        else
        {
            std::cerr << "Tacx::update: received invalid no. of bytes: recv=" << status << ", got=" << status << std::endl;
        }
    }
    else
    {
        if (nBytes == -5)
        { // reopen USB device

            if (usbOpenDevice(&handle, USBDEV_Tacx_VENDOR, "Tacx?VR", USBDEV_VRInterface_PRODUCT, "VR-Interface") != 0)
            {
                fprintf(stderr, "Could not find USB device \"SPI-USB\" with vid=0x%x pid=0x%x\n", USBDEV_Tacx_VENDOR, USBDEV_VRInterface_PRODUCT);

                // not open, don't close usb_close(handle);
                nBytes = -5;
                return;
            }
            else
            {
                fprintf(stderr, "found USB device \"SPI-USB\" with vid=0x%x pid=0x%x\n", USBDEV_Tacx_VENDOR, USBDEV_VRInterface_PRODUCT);
                // init
                nBytes = 0;
                memset(tmp, 0, sizeof(tmp));
                tmp[0] = 2;
                int bytesTransferred=0;
                ret = libusb_bulk_transfer(handle, LIBUSB_ENDPOINT_OUT, tmp, 4,&bytesTransferred, 5000);
                if (ret < 0 || bytesTransferred!=4)
                {
                    printf("error writing:\n%s %d\n", libusb_error_name(ret),bytesTransferred);
                    libusb_close(handle);
                    handle = NULL;
                }
                else
                {
                    printf("success: bulk write %d bytes\n", bytesTransferred);
                }
            }
        }
        else
        {
            // Running a sync read
            static int errorCounter = 0;
            int bytesTransferred=0;
            int retry = 0;
            do {
                ret = libusb_bulk_transfer(handle, LIBUSB_ENDPOINT_IN, (unsigned char *)&vrdata, 64,&bytesTransferred, 200);
                if (ret == LIBUSB_ERROR_PIPE) {
                    libusb_clear_halt(handle, LIBUSB_ENDPOINT_IN);
                }
                retry++;
            } while ((ret == LIBUSB_ERROR_PIPE) && (retry<10));
            if (ret < 0 || bytesTransferred!=4)
            {
                    errorCounter++;
                    printf("error reading:\n%s count: %d ret: %d\n", libusb_error_name(ret),errorCounter, ret);
                    if (errorCounter > 2)
                    {
                        libusb_close(handle);
                        nBytes = -5;
                        return;
                    }
            }
            else
            {
                fprintf(stderr, "\r");
                fprintf(stderr, "Tasten: %1d ", vrdata.tasten);
                fprintf(stderr, "Lenkwinkel: %6d ", vrdata.Lenkwinkel);
                fprintf(stderr, "Drehzahl: %6d ", vrdata.drehzahl);
                fprintf(stderr, "Drehzahl: %6f ", getSpeed());
                fprintf(stderr, "Angle: %6f ", getAngle());
                fprintf(stderr, "Trittfrequenz: %6d ", vrdata.trittfrequenz);
                //fprintf(stderr,"ti: %1d ",vrdata.trittfreqenzimpuls);
                /*	unsigned char *tmpc = (unsigned char *)&vrdata;
		int i;
		fprintf(stderr,"\r");
		for(i = 0;i<64;i++)
		{
		fprintf(stderr,"%2x ",tmpc[i]);
		}*/

                //fprintf(stderr,"\n");
                errorCounter = 0;
                vrdataout.unknown0 = 0x00010801;
                //vrdataout.force = 0xF959;
                vrdataout.unknown1 = 0;
                vrdataout.unknown2 = 0x05145702;
                int bytesTransferred=0;
                ret = libusb_bulk_transfer(handle, LIBUSB_ENDPOINT_OUT, (unsigned char *)&vrdataout, 12,&bytesTransferred, 5000);
                if (ret < 0 || bytesTransferred!=4)
                {
                    printf("error writing:\n%s %d\n", libusb_error_name(ret),bytesTransferred);
                    libusb_close(handle);
                    handle = NULL;
                    nBytes = -5;
                    return;
                }
                else
                {
                    //printf("success: bulk write %d bytes\n", bytesTransferred);
                }
            }
            n++;
        }
    }
}

float Tacx::getRPM()
{
       return vrdata.drehzahl;
} 

float Tacx::getAngle()
{
    float angle = (vrdata.Lenkwinkel - zeroAngle) / 300.0;
    if (angle < 0.) {
       return -angle*angle;
    } 
    else
    {   
       return angle*angle;
    }
} // -1 - 1 min-max

int Tacx::getButtons()
{
    if (vrdata.tasten & 0x4)
    {
        // arrow up
        zeroAngle = vrdata.Lenkwinkel;
        std::cerr << "reset Lenkwinkel to " << vrdata.Lenkwinkel << std::endl;
    }

    return vrdata.tasten & 0xb;
}
