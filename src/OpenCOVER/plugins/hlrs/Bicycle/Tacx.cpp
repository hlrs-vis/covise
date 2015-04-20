/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "Tacx.h"
#include "UDPComm.h"
#include <config/CoviseConfig.h>

static float zeroAngle = 1152.;

int Tacx::usbGetStringAscii(usb_dev_handle *dev, int index, int langid, char *buf, int buflen)
{
    char buffer[2000];
    int rval, i;

    if ((rval = usb_control_msg(dev, USB_ENDPOINT_IN, USB_REQ_GET_DESCRIPTOR, (USB_DT_STRING << 8) + index, langid, buffer, sizeof(buffer), 1000)) < 0)
        return rval;
    if (buffer[1] != USB_DT_STRING)
        return 0;
    if ((unsigned char)buffer[0] < rval)
        rval = (unsigned char)buffer[0];
    rval /= 2;
    /* lossy conversion to ISO Latin1 */
    for (i = 1; i < rval; i++)
    {
        if (i > buflen) /* destination buffer overflow */
            break;
        buf[i - 1] = buffer[2 * i];
        if (buffer[2 * i + 1] != 0) /* outside of ISO Latin1 range */
            buf[i - 1] = '?';
    }
    buf[i - 1] = 0;
    return i - 1;
}

int Tacx::usbOpenDevice(usb_dev_handle **device, int vendor, const char *vendorName, int product, const char *productName)
{
    struct usb_bus *bus;
    struct usb_device *dev;
    usb_dev_handle *handle = NULL;
    int errorCode = USB_ERROR_NOTFOUND;
    static int didUsbInit = 0;

    if (!didUsbInit)
    {
        didUsbInit = 1;
        usb_init();
    }
    usb_find_busses();
    usb_find_devices();
    for (bus = usb_get_busses(); bus; bus = bus->next)
    {
        for (dev = bus->devices; dev; dev = dev->next)
        {
            if (dev->descriptor.idVendor == vendor && dev->descriptor.idProduct == product)
            {
                char string[256];
                int len;
                handle = usb_open(dev); /* we need to open the device in order to query strings */
                if (!handle)
                {
                    errorCode = USB_ERROR_ACCESS;
                    fprintf(stderr, "Warning: cannot open USB device: %s\n", usb_strerror());
                    continue;
                }
                if (vendorName == NULL && productName == NULL)
                { /* name does not matter */
                    break;
                }
                /* now check whether the names match: */
                len = usbGetStringAscii(handle, dev->descriptor.iManufacturer, 0x0409, string, sizeof(string));
                if (len < 0)
                {
                    errorCode = USB_ERROR_IO;
                    fprintf(stderr, "Warning: cannot query manufacturer for device: %s\n", usb_strerror());
                }
                else
                {
                    errorCode = USB_ERROR_NOTFOUND;
                    /* fprintf(stderr, "seen device from vendor ->%s<-\n", string); */
                    if (strcmp(string, vendorName) == 0)
                    {
                        len = usbGetStringAscii(handle, dev->descriptor.iProduct, 0x0409, string, sizeof(string));
                        if (len < 0)
                        {
                            errorCode = USB_ERROR_IO;
                            fprintf(stderr, "Warning: cannot query product for device: %s\n", usb_strerror());
                        }
                        else
                        {
                            errorCode = USB_ERROR_NOTFOUND;
                            /* fprintf(stderr, "seen product ->%s<-\n", string); */
                            if (strcmp(string, productName) == 0)
                                break;
                        }
                    }
                }
                usb_close(handle);
                handle = NULL;
            }
        }
        if (handle)
            break;
    }

    if (handle && (usb_set_configuration(handle, MY_CONFIG) < 0))
    {
        printf("error setting config #%d: %s\n", MY_CONFIG, usb_strerror());
        usb_close(handle);
        handle = NULL;
        errorCode = USB_ERROR_CONFIG;
    }
    else
    {
        printf("success: set configuration #%d\n", MY_CONFIG);
    }

    if (handle && (usb_claim_interface(handle, 0) < 0))
    {
        printf("error claiming interface #%d:\n%s\n", MY_INTF, usb_strerror());
        usb_close(handle);
        handle = NULL;
        errorCode = USB_ERROR_CLAIM;
    }
    else
    {
        printf("success: claim_interface #%d\n", MY_INTF);
    }

    // init
    if (handle)
    {
        memset(tmp, 0, sizeof(tmp));
        tmp[0] = 2;
        ret = usb_bulk_write(handle, EP_OUT, tmp, 4, 5000);
        if (ret < 0)
        {
            printf("error writing:\n%s\n", usb_strerror());
            usb_close(handle);
            handle = NULL;
            errorCode = USB_ERROR_INIT;
        }
        else
        {
            printf("success: bulk write %d bytes\n", ret);
        }
    }

    if (handle != NULL)
    {
        errorCode = 0;
        *device = handle;
    }

    return errorCode;
}

Tacx::Tacx()
    : udp(NULL)
{
    handle = NULL;
    errorCode = USB_ERROR_NOTFOUND;
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
        const std::string host = covise::coCoviseConfig::getEntry("value", "Bicycle.serverHost", "141.58.8.171");
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
                ret = usb_bulk_write(handle, EP_OUT, tmp, 4, 5000);
                if (ret < 0)
                {
                    printf("error writing:\n%s\n", usb_strerror());
                    usb_close(handle);
                    nBytes = -5;
                    return;
                }
                else
                {
                    printf("success: bulk write %d bytes\n", ret);
                }
            }
        }
        else
        {
            // Running a sync read
            static int errorCounter = 0;
            ret = usb_bulk_read(handle, EP_IN, (char *)&vrdata, 64, 200);
            if (ret < 0)
            {
                errorCounter++;
                printf("error reading:\n%s count: %d ret: %d\n", usb_strerror(), errorCounter, ret);
                if (errorCounter > 2)
                {
                    usb_close(handle);
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
                ret = usb_bulk_write(handle, EP_OUT, (char *)&vrdataout, 12, 5000);
                if (ret < 0)
                {
                    printf("error writing:\n%s\n", usb_strerror());
                    usb_close(handle);
                    nBytes = -5;
                    return;
                }
                else
                {
                    //printf("success: bulk write %d bytes\n", ret);
                }
            }
            n++;
        }
    }
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
