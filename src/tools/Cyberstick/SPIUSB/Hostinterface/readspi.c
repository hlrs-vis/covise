/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <conio.h>

#include "lusb0_usb.h"

#define USBDEV_SHARED_VENDOR 0x16C0 /* VOTI */
#define USBDEV_SHARED_PRODUCT 0x05DC /* Obdev's free shared PID */

#define USB_ERROR_NOTFOUND 1
#define USB_ERROR_ACCESS 2
#define USB_ERROR_IO 3
#define USB_LED_ON 1
#define USB_LED_OFF 0
#define USB_DATA_OUT 2
#define USB_INIT 3

static int usbGetStringAscii(usb_dev_handle *dev, int index, int langid, char *buf, int buflen)
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

static int usbOpenDevice(usb_dev_handle **device, int vendor, char *vendorName, int product, char *productName)
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
    if (handle != NULL)
    {
        errorCode = 0;
        *device = handle;
    }
    return errorCode;
}

int main(int argc, char **argv)
{
    struct usb_bus *bus;
    struct usb_device *dev;

    usb_dev_handle *handle = NULL;
    int errorCode = USB_ERROR_NOTFOUND;

    int nBytes;
    char buffer[512];
    int n = 0;
    int on = 1;

    usb_init(); // initialize libusb

    if (usbOpenDevice(&handle, USBDEV_SHARED_VENDOR, "obdev.at", USBDEV_SHARED_PRODUCT, "SPI-USB") != 0)
    {
        fprintf(stderr, "Could not find USB device \"SPI-USB\" with vid=0x%x pid=0x%x\n", USBDEV_SHARED_VENDOR, USBDEV_SHARED_PRODUCT);
        exit(1);
    }
    // if(strcmp(argv[1], "on") == 0) {
    nBytes = usb_control_msg(handle,
                             USB_TYPE_VENDOR | USB_RECIP_DEVICE | USB_ENDPOINT_IN,
                             USB_INIT, 0, 0, (char *)buffer, sizeof(buffer), 5000);
    // } else if(strcmp(argv[1], "off") == 0) {
    //     nBytes = usb_control_msg(handle,
    //         USB_TYPE_VENDOR | USB_RECIP_DEVICE | USB_ENDPOINT_IN,
    //         USB_LED_OFF, 0, 0, (char *)buffer, sizeof(buffer), 5000);
    // }
    while (1)
    {
        Sleep(500);
        nBytes = usb_control_msg(handle,
                                 USB_TYPE_VENDOR | USB_RECIP_DEVICE | USB_ENDPOINT_IN,
                                 USB_DATA_OUT, 0, 0, (char *)buffer, sizeof(buffer), 5000);
        if (nBytes > 0)
        {
            buffer[nBytes] = '\0';
            if (nBytes > 0)
            {
                int i;
                for (i = 0; i < nBytes; i++)
                {
                    int byte = buffer[i];
                    fprintf(stderr, "%x,", byte & 0xff);
                }
                fprintf(stderr, "\n");
            }
        }
        else if (nBytes == -5)
        { // reopen USB device

            while (usbOpenDevice(&handle, USBDEV_SHARED_VENDOR, "obdev.at", USBDEV_SHARED_PRODUCT, "SPI-USB") != 0)
            {
                fprintf(stderr, "Could not find USB device \"SPI-USB\" with vid=0x%x pid=0x%x\n", USBDEV_SHARED_VENDOR, USBDEV_SHARED_PRODUCT);
                Sleep(1000);
                //    exit(1);
            }
            fprintf(stderr, "found USB device \"SPI-USB\" with vid=0x%x pid=0x%x\n", USBDEV_SHARED_VENDOR, USBDEV_SHARED_PRODUCT);
        }
        else
        {
            if (n > 100)
            {
                if (on)
                {
                    nBytes = usb_control_msg(handle,
                                             USB_TYPE_VENDOR | USB_RECIP_DEVICE | USB_ENDPOINT_IN,
                                             USB_LED_OFF, 0, 0, (char *)buffer, sizeof(buffer), 5000);
                    on = 0;
                    n = 0;
                }
                else
                {
                    nBytes = usb_control_msg(handle,
                                             USB_TYPE_VENDOR | USB_RECIP_DEVICE | USB_ENDPOINT_IN,
                                             USB_LED_ON, 0, 0, (char *)buffer, sizeof(buffer), 5000);
                    on = 1;
                    n = 1;
                }
            }
            n++;
        }
    }
    return 0;
}