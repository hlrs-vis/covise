/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>
#ifdef WIN32
#include <windows.h>
#include "lusb0_usb.h"
#include <conio.h>
#else
#include <usb.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define USBDEV_Tacx_VENDOR 13665
#define USBDEV_VRInterface_PRODUCT 6450

#define USB_ERROR_NOTFOUND 1
#define USB_ERROR_ACCESS 2
#define USB_ERROR_IO 3
#define USB_ERROR_CONFIG 4
#define USB_ERROR_CLAIM 5
#define USB_ERROR_INIT 6
#define EP_OUT 0x02
#define EP_IN 0x82

// Device configuration and interface id.
#define MY_CONFIG 1
#define MY_INTF 0

class UDPComm;

#pragma pack(push, 1)
struct VRData
{
    unsigned int id1; //1-4
    unsigned int id2; //5-8
    unsigned int id3; //9-12
    unsigned char unknown1; // 13
    unsigned char tasten; // 14
    unsigned int unknown2; // 15-18
    unsigned short Lenkwinkel; // 19-20
    unsigned short unknown3; // 21-22
    unsigned int unknown4; // 23-26
    unsigned int unknown5; // 27-30
    unsigned short unknown6; // 31-32
    unsigned int drehzahl; //33-36
    unsigned short unknown7; // 37-38
    unsigned int unknown8; // 39-42
    unsigned char trittfreqenzimpuls; // 43
    unsigned char unknown9; // 44
    unsigned short trittfrequenz; // 45-46
    unsigned int unknown10; // 47-50
    unsigned int unknown11; // 51-54
    unsigned int unknown12; // 55-58
    unsigned int unknown13; // 59-62
    unsigned short unknown14; // 63-64
};
#pragma pack(pop)

#pragma pack(push, 1)
struct VRDataOut
{
    unsigned int unknown0; //1-4
    unsigned short force; // 5-6
    unsigned short unknown1; // 7-8
    unsigned int unknown2; //5-8
};
#pragma pack(pop)

class PLUGINEXPORT Tacx
{
public:
    Tacx();
    ~Tacx();
    void update();
    float getAngle();
    float getSpeed()
    {
        return vrdata.drehzahl / 1000.0;
    };
    int getButtons();
    void setForce(float f)
    {
        vrdataout.force = ((unsigned short)(((float)0xf00) * f + 0x100));
        fprintf(stderr, "force=0x%x\n", vrdataout.force);
    }; // 0 - 1 min-max

private:
    void init();

    UDPComm *udp;

    int usbGetStringAscii(usb_dev_handle *dev, int index, int langid, char *buf, int buflen);
    int usbOpenDevice(usb_dev_handle **device, int vendor, const char *vendorName, int product, const char *productName);
    usb_dev_handle *handle;
    int errorCode;

    int nBytes;
    int n;
    int on;
    char tmp[64];
    int ret;

    VRData vrdata;
    VRDataOut vrdataout;
};
