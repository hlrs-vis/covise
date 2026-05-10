/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coRawDevice.h"
#include <stdio.h>
#include <util/unixcompat.h>

#pragma comment(lib, "hid.lib")
#pragma comment(lib, "setupapi.lib")

using namespace opencover;
coRawDevice::coRawDevice(int n)
{
    buttonNumber = n;
}

coRawDevice::coRawDevice(const char *deviceName)
{
    buttonNumber = 0;
    if (deviceName == NULL)
        return;
    fprintf(stderr, "looking for:%s\n", deviceName);
    for (int i = 0; i < coRawDeviceManager::instance()->numDevices(); i++)
    {
        if(strlen(coRawDeviceManager::instance()->rawDevices[i].deviceName)>4)
        {
            RAW_MOUSE &dev = coRawDeviceManager::instance()->rawDevices[i];
            fprintf(stderr, "try        :%s\n", dev.deviceName + 4);
            if ( strncasecmp(deviceName, dev.deviceName + 4, strlen(deviceName)) == 0)
            {
                fprintf(stderr, "found ID:%d ", i);
                buttonNumber = i;

                dev.HIDdevice_handle= CreateFile(
                    dev.deviceName,
                    GENERIC_READ,
                    FILE_SHARE_READ | FILE_SHARE_WRITE,
                    nullptr,
                    OPEN_EXISTING,
                    0,
                    nullptr);

                if (dev.HIDdevice_handle == INVALID_HANDLE_VALUE)
                {
                    printf("CreateFile failed: %lu\n", GetLastError());
                }
                std::wcout << "ProductName: " << coRawDeviceManager::instance()->GetDeviceProductName(dev.HIDdevice_handle) << "\t\t";

                break;
            }
        }
        else
        {
            fprintf(stderr, "ignoring        :%s\n", coRawDeviceManager::instance()->rawDevices[i].deviceName);
        }
    }
}

coRawDevice::~coRawDevice()
{
}

int coRawDevice::getX()
{
    return coRawDeviceManager::instance()->rawDevices[buttonNumber].x;
}
int coRawDevice::getY()
{
    return coRawDeviceManager::instance()->rawDevices[buttonNumber].y;
}
int coRawDevice::getWheelCount()
{
    return coRawDeviceManager::instance()->rawDevices[buttonNumber].z;
}
bool coRawDevice::getButton(int i)
{
    return coRawDeviceManager::instance()->is_raw_device_button_pressed(buttonNumber, i) != 0;
}
int coRawDevice::getNumValues()
{
    return coRawDeviceManager::instance()->rawDevices[buttonNumber].numValues;
}

float coRawDevice::getValue(int i)
{
    return coRawDeviceManager::instance()->rawDevices[buttonNumber].values[i];
}

unsigned int coRawDevice::getButtonBits()
{
    unsigned int bits = 0;
    for (int i = 0; i < MAX_RAW_MOUSE_BUTTONS; i++)
    {
        if (coRawDeviceManager::instance()->is_raw_device_button_pressed(buttonNumber, i))
        {
            if (i == 1)
                bits |= (1 << 2);
            else if (i == 2)
                bits |= (1 << 1);
            else
                bits |= (1 << i);
        }
    }
    return bits;
}

//============================================================
//	numDevices
//============================================================

int coRawDeviceManager::numDevices()
{
    return nInputDevices;
}

//============================================================
//	is_rm_rdp_device
//============================================================

BOOL coRawDeviceManager::is_rm_rdp_device(char cDeviceString[])
{
    int i;
    char cRDPString[] = "\\??\\Root#RDP_MOU#0000#";

    if (strlen(cDeviceString) < 22)
    {
        return 0;
    }

    for (i = 0; i < 22; i++)
    {
        if (cRDPString[i] != cDeviceString[i])
        {
            return 0;
        }
    }

    return 1;
}

//============================================================
//	register_raw_device
//============================================================

BOOL coRawDeviceManager::register_raw_device(void)
{
    // This function registers to receive the WM_INPUT messages
    RAWINPUTDEVICE Rid[10]; 

    //register to get wm_input messages
    Rid[0].usUsagePage = 0x01;
    Rid[0].usUsage = 0x01;
    Rid[0].dwFlags = RIDEV_INPUTSINK; // RIDEV_NOLEGACY;   // adds HID device and also ignores legacy device messages
    Rid[0].hwndTarget = handle_;
    //register to get wm_input messages
    Rid[1].usUsagePage = 0x01;
    Rid[1].usUsage = 0x02;
    Rid[1].dwFlags = RIDEV_INPUTSINK; // RIDEV_NOLEGACY;   // adds HID device and also ignores legacy device messages
    Rid[1].hwndTarget = handle_;
    //register to get wm_input messages
    Rid[2].usUsagePage = 0x01;
    Rid[2].usUsage = 0x04;
    Rid[2].dwFlags = RIDEV_INPUTSINK; // RIDEV_NOLEGACY;   // adds HID device and also ignores legacy device messages
    Rid[2].hwndTarget = handle_;
    //register to get wm_input messages
    Rid[3].usUsagePage = 0x01;
    Rid[3].usUsage = 0x05;
    Rid[3].dwFlags = RIDEV_INPUTSINK; // RIDEV_NOLEGACY;   // adds HID device and also ignores legacy device messages
    Rid[3].hwndTarget = handle_;
    //register to get wm_input messages
    Rid[4].usUsagePage = 0x01;
    Rid[4].usUsage = 0x06;
    Rid[4].dwFlags = RIDEV_INPUTSINK; // RIDEV_NOLEGACY;   // adds HID device and also ignores legacy device messages
    Rid[4].hwndTarget = handle_;
    //register to get wm_input messages
    Rid[5].usUsagePage = 0x01;
    Rid[5].usUsage = 0x07;
    Rid[5].dwFlags = RIDEV_INPUTSINK; // RIDEV_NOLEGACY;   // adds HID device and also ignores legacy device messages
    Rid[5].hwndTarget = handle_;

    // Register to receive the WM_INPUT message for any change in device (buttons, wheel, and movement will all generate the same message)
    if (!/* RegisterRawInputDevices*/ (*_RRID)(Rid, 6, sizeof(Rid[0])))
        return 0;

    return 1;
}

//============================================================
//	read_raw_input
//============================================================

BOOL coRawDeviceManager::read_raw_input(PRAWINPUT raw)
{
    for (int i=0; i < nInputDevices; i++)
    {
        auto &dev = rawDevices[i];
        if (dev.device_handle == raw->header.hDevice)
        {
            if(dev.type == RIM_TYPEMOUSE)
            {
                //fprintf(stderr,"MOUSEMessage %d\n",i);
                // Update the values for the specified device
                if (dev.is_absolute)
                {
                    dev.x = raw->data.mouse.lLastX;
                    dev.y = raw->data.mouse.lLastY;
                }
                else
                { // relative
                    dev.x += raw->data.mouse.lLastX;
                    dev.y += raw->data.mouse.lLastY;
                }
                //fprintf(stderr,"Raw Device Event: num: %d Flags: %d\n",i,raw->data.mouse.usButtonFlags);
                if (raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_1_DOWN)
                    dev.buttonpressed[0] = 1;
                if (raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_1_UP)
                    dev.buttonpressed[0] = 0;
                if (raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_2_DOWN)
                    dev.buttonpressed[1] = 1;
                if (raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_2_UP)
                    dev.buttonpressed[1] = 0;
                if (raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_3_DOWN)
                    dev.buttonpressed[2] = 1;
                if (raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_3_UP)
                    dev.buttonpressed[2] = 0;
                if (raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_4_DOWN)
                    dev.buttonpressed[3] = 1;
                if (raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_4_UP)
                    dev.buttonpressed[3] = 0;
                if (raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_5_DOWN)
                    dev.buttonpressed[4] = 1;
                if (raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_5_UP)
                    dev.buttonpressed[4] = 0;

                if (raw->data.mouse.usFlags & MOUSE_MOVE_ABSOLUTE)
                    dev.is_absolute = 1;
                else if (raw->data.mouse.usFlags & MOUSE_MOVE_RELATIVE)
                    dev.is_absolute = 0;
                if (raw->data.mouse.usFlags & MOUSE_VIRTUAL_DESKTOP)
                    dev.is_virtual_desktop = 1;
                else
                    dev.is_virtual_desktop = 0;

                if (raw->data.mouse.usButtonFlags & RI_MOUSE_WHEEL)
                { // If the current message has a device_wheel message
                    if ((SHORT)raw->data.mouse.usButtonData > 0)
                    {
                        dev.z++;
                    }
                    if ((SHORT)raw->data.mouse.usButtonData < 0)
                    {
                        dev.z--;
                    }
                }
            }
            else if(dev.type == RIM_TYPEKEYBOARD)
            {
#if 0
                fprintf(stderr,"KeyboardMessage %d \n",i);
                fprintf(stderr,"Kbd: make=%04x Flags:%04x Reserved:%04x ExtraInformation:%08x, msg=%04x VK=%04x i=%d\n", 
    raw->data.keyboard.MakeCode, 
    raw->data.keyboard.Flags, 
    raw->data.keyboard.Reserved, 
    raw->data.keyboard.ExtraInformation, 
    raw->data.keyboard.Message, 
    raw->data.keyboard.VKey,
					i);
#endif
                int value=0;
                if(raw->data.keyboard.Message == 0x100) // key press
                {
                    value = 1;
                }
                if(raw->data.keyboard.Message == 0x101) // key release
                {
                    value = 0;
                }
                
                
                    if(raw->data.keyboard.MakeCode == 0x49) // Page up
                    {
                        dev.buttonpressed[0] = value;
                    }
                    if(raw->data.keyboard.MakeCode == 0x51)// Page down
                    {
                        dev.buttonpressed[1] = value;
                    }
                    if(raw->data.keyboard.MakeCode == 0x34)// .
                    {
                        dev.buttonpressed[3] = value;
                    }
                    if(raw->data.keyboard.MakeCode == 0x3f)// F5
                    {
                        dev.buttonpressed[2] = value;
                    }
                    if(raw->data.keyboard.MakeCode == 0x3f)// Escape
                    {
                        dev.buttonpressed[2] = value;
                    }
            }
            else if(dev.type == RIM_TYPEHID)
            {
                if (dev.preparsed == nullptr)
                {
                    HidD_GetPreparsedData(dev.HIDdevice_handle, &dev.preparsed);
                    HidP_GetCaps(dev.preparsed, &dev.caps);
                    dev.numValues = dev.caps.NumberInputValueCaps;
                    dev.numIndices = dev.caps.NumberInputDataIndices;
                    for (int i = 0; i < dev.numValues; i++)
                    {
                        dev.values[i] = 0;
                    }
                }
                ULONG dataCount = dev.caps.NumberInputDataIndices;

                std::vector<HIDP_DATA> data(dataCount);

                for (int i = 0; i < MAX_RAW_MOUSE_BUTTONS; i++)
                    dev.buttonpressed[i] = 0;

                NTSTATUS status = HidP_GetData(
                    HidP_Input,
                    data.data(),
                    &dataCount,
                    dev.preparsed,
                    (PCHAR)raw->data.hid.bRawData,
                    raw->data.hid.dwSizeHid);
                for (ULONG i = 0; i < dataCount; ++i)
                {
                    const HIDP_DATA &d = data[i];

                    //printf("DataIndex: %u\n", d.DataIndex);

                    if (d.On == 1 || (d.On > 0 && d.On == d.RawValue && d.On <9))
                    {
                        if (d.DataIndex < MAX_RAW_MOUSE_BUTTONS)
                            dev.buttonpressed[d.DataIndex+(int)(d.On)] = 1;
                    }
                    else
                    {
                        if (d.DataIndex < MAX_RAW_MOUSE_VALUES)
                            dev.values[d.DataIndex] = (d.RawValue / 32767.0f) - 1.0f;
                    }
                }
                /* cerr << "values: " << endl;
                for (int i = 0; i < dev.numIndices; i++)
                {
                    cerr << " " << dev.values[i]; 
                }
                cerr << endl;*/
                /*
                //fprintf(stderr,"HIDMessage %d \n",i);
                USAGE buttons[128];
                ULONG buttonCount = 128;

                HidP_GetUsages(
                    HidP_Input,
                    0x09, // Button page
                    0,
                    buttons,
                    &buttonCount,
                    dev.preparsed,
                    (PCHAR)raw->data.hid.bRawData,
                    raw->data.hid.dwSizeHid);
                for (ULONG i = 0; i < buttonCount; i++)
                {
                    if (buttons[i] < MAX_RAW_MOUSE_BUTTONS)
                    {
                        dev.buttonpressed[buttons[i]] = 1;
                    }
                }
                for (int i = 0; i < MAX_RAW_MOUSE_VALUES; i++)
                {
                    //dev.values[i] = 0;
                    HidP_GetScaledUsageValue(
                        HidP_Input,
                        0x01, // Generic Desktop page
                        0,
                        0x30 + i, // X axis
                        &dev.values[i],
                        dev.preparsed,
                        (PCHAR)raw->data.hid.bRawData,
                        raw->data.hid.dwSizeHid);
                }*/
            }
            else
            {
                fprintf(stderr,"unknown HID type %d, %d \n",dev.type,i);
            }

        }
    }

    return 1;
}

//============================================================
//	is_raw_device_button_pressed
//============================================================

BOOL coRawDeviceManager::is_raw_device_button_pressed(int devicenum, int buttonnum)
{

    // It's ok to ask if buttons are pressed for unitialized mice - just tell 'em no button's pressed
    if (devicenum >= nInputDevices || buttonnum >= MAX_RAW_MOUSE_BUTTONS || rawDevices == NULL)
        return 0;

    return (rawDevices[devicenum].buttonpressed[buttonnum]);
}

//============================================================
//	is_raw_device_absolute
//============================================================
BOOL coRawDeviceManager::is_raw_device_absolute(int devicenum)
{
    return (rawDevices[devicenum].is_absolute);
}

//============================================================
//	is_raw_device_virtual_desktop
//============================================================
BOOL coRawDeviceManager::is_raw_device_virtual_desktop(int devicenum)
{
    return (rawDevices[devicenum].is_virtual_desktop);
}

//============================================================
//	get_raw_device_button_name
//============================================================

char *coRawDeviceManager::get_raw_device_button_name(int devicenum, int buttonnum)
{
    if (devicenum >= nInputDevices || buttonnum >= MAX_RAW_MOUSE_BUTTONS || rawDevices == NULL)
        return NULL;
    return (rawDevices[devicenum].button_name[buttonnum]);
}

//============================================================
//	processData
//============================================================

BOOL coRawDeviceManager::processData(HANDLE in_device_handle)
{
    // When the WM_INPUT message is received, the lparam must be passed to this function to keep a running tally of
    //     every device moves to maintain accurate results for get_raw_device_?_delta().
    // This function will take the HANDLE of the device and find the device in the rawDevices arrayand add the
    //      x and y devicemove values according to the information stored in the RAWINPUT structure.

    int dwSize;

    if (/* GetRawInputData */ (*_GRID)((HRAWINPUT)in_device_handle, RID_INPUT, NULL, &dwSize, sizeof(RAWINPUTHEADER)) == -1)
    {
        //fprintf(stderr, "ERROR: Unable to add to get size of raw input header.\n");
        // oops, probably lost a bluetooth device, try to restart
        setupDevices();
        return 0;
    }
    if (oldSize < dwSize)
    {
        oldSize = dwSize;
        free(lpb);
        lpb = (LPBYTE)malloc(sizeof(LPBYTE) * dwSize);
        if (lpb == NULL)
        {
            fprintf(stderr, "ERROR: Unable to allocate memory for raw input header.\n");
            return 0;
        }
    }

    if (/* GetRawInputData */ (*_GRID)((HRAWINPUT)in_device_handle, RID_INPUT, lpb, &dwSize, sizeof(RAWINPUTHEADER)) != dwSize)
    {
        //fprintf(stderr, "ERROR: Unable to add to get raw input header.\n");
        return 0;
    }

    read_raw_input((RAWINPUT *)lpb);

    return 1;
}

//============================================================
//	get_raw_device_x_delta
//============================================================

ULONG coRawDeviceManager::get_raw_device_x_delta(int devicenum)
{
    ULONG nReturn = 0;

    if (rawDevices != NULL && devicenum < nInputDevices)
    {
        nReturn = rawDevices[devicenum].x;
        if (!rawDevices[devicenum].is_absolute)
            rawDevices[devicenum].x = 0;
    }

    return nReturn;
}

//============================================================
//	get_raw_device_y_delta
//============================================================

ULONG coRawDeviceManager::get_raw_device_y_delta(int devicenum)
{
    ULONG nReturn = 0;

    if (rawDevices != NULL && devicenum < nInputDevices)
    {
        nReturn = rawDevices[devicenum].y;
        if (!rawDevices[devicenum].is_absolute)
            rawDevices[devicenum].y = 0;
    }

    return nReturn;
}

//============================================================
//	get_raw_device_z_delta
//============================================================

ULONG coRawDeviceManager::get_raw_device_z_delta(int devicenum)
{
    ULONG nReturn = 0;

    if (rawDevices != NULL && devicenum < nInputDevices)
    {
        nReturn = rawDevices[devicenum].z;
        if (!rawDevices[devicenum].is_absolute)
            rawDevices[devicenum].z = 0;
    }

    return nReturn;
}
LRESULT CALLBACK
    MainWndProc(HWND hwnd, UINT nMsg, WPARAM wParam, LPARAM lParam)
{
    switch (nMsg)
    {

    case WM_INPUT:
    {
        coRawDeviceManager::instance()->processData((HRAWINPUT)lParam);
    }
    break;
    }
    return DefWindowProc(hwnd, nMsg, wParam, lParam);
}

coRawDeviceManager *coRawDeviceManager::inst = NULL;

coRawDeviceManager *coRawDeviceManager::instance()
{
    if (inst == NULL)
    {
        inst = new coRawDeviceManager();
    }
    return inst;
}

coRawDeviceManager::~coRawDeviceManager()
{
    int i, j;
    for (i = 0; i < nInputDevices; i++)
    {
        for (j = 0; j < MAX_RAW_MOUSE_BUTTONS; j++)
        {

            delete[] rawDevices[i].button_name[j];
        }
        delete[] rawDevices[i].deviceName;
    }

    delete[] rawDevices;
}

coRawDeviceManager::coRawDeviceManager()
{

    oldSize = 0;
    nInputDevices = 0;
    lpb = NULL;
    instance_ = GetModuleHandle(0);

    WNDCLASS wndclass;

    wndclass.style = CS_HREDRAW | CS_VREDRAW;
    wndclass.lpfnWndProc = MainWndProc;
    wndclass.cbClsExtra = 0;
    wndclass.cbWndExtra = 0;
    wndclass.hInstance = instance_;
    wndclass.hCursor = 0;
    wndclass.hIcon = 0;
    wndclass.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wndclass.lpszMenuName = 0;
    wndclass.lpszClassName = "deviceInputWindow";

    if (!RegisterClass(&wndclass))
    {
        fprintf(stderr, "RegisterClass Error:%d\n", GetLastError());
    }

    if (!(handle_ = CreateWindowEx(0,
                                   "deviceInputWindow",
                                   TEXT("deviceInputWindow"),
                                   WS_POPUP,
                                   0,
                                   0,
                                   100,
                                   100,
                                   0,
                                   0,
                                   instance_,
                                   0)))
    {
        return;
    }


    static BOOL bHasBeenInitialized = 0;

    // Return 0 if rawinput is not available
    HMODULE user32 = LoadLibrary("user32.dll");
    if (!user32)
        return;
    _RRID = (pRegisterRawInputDevices)GetProcAddress(user32, "RegisterRawInputDevices");
    if (!_RRID)
        return;
    _GRIDL = (pGetRawInputDeviceList)GetProcAddress(user32, "GetRawInputDeviceList");
    if (!_GRIDL)
        return;
    _GRIDIA = (pGetRawInputDeviceInfoA)GetProcAddress(user32, "GetRawInputDeviceInfoA");
    if (!_GRIDIA)
        return;
    _GRID = (pGetRawInputData)GetProcAddress(user32, "GetRawInputData");
    if (!_GRID)
        return;


    if (bHasBeenInitialized)
    {
        fprintf(stderr, "WARNING: rawdevice init called after initialization already completed.");
        bHasBeenInitialized = 1;
        return;
    }

    rawDevices = NULL;
    setupDevices();

    // finally, register to recieve raw input WM_INPUT messages
    if (!register_raw_device())
    {
        fprintf(stderr, "ERROR: Unable to register raw input (2).\n");
        return;
    }
    bHasBeenInitialized = 1;
}
void coRawDeviceManager::escape(std::string &data)
{
    std::string::size_type pos = 0;
    for (;;)
    {
        pos = data.find_first_of("\"&<>", pos);
        if (pos == std::string::npos) break;
        std::string replacement;
        switch ((data)[pos])
        {
        case '\"': replacement = "&quot;"; break;   
        case '&':  replacement = "&amp;";  break;   
        case '<':  replacement = "&lt;";   break;   
        case '>':  replacement = "&gt;";   break;   
        default: ;
        }
        data.replace(pos, 1, replacement);
        pos += replacement.size();
    };
}
void coRawDeviceManager::setupDevices()
{
    char buffer[80];
    int  i, j;
    PRAWINPUTDEVICELIST pRawInputDeviceList;
    int nSize;
    char *psName;
    fprintf(stderr,"List of available Raw Input Devices:\n");
    for (i = 0; i < nInputDevices; i++)
    {
        for (j = 0; j < MAX_RAW_MOUSE_BUTTONS; j++)
        {
            delete[] rawDevices[i].button_name[j];
        }
        delete[] rawDevices[i].deviceName;
    }
    delete[] rawDevices;
    //fprintf(stderr,"testit1\n");
    // 1st call to GetRawInputDeviceList: Pass NULL to get the number of devices.
    if (/* GetRawInputDeviceList */ (*_GRIDL)(NULL, &nInputDevices, sizeof(RAWINPUTDEVICELIST)) != 0)
    {
        fprintf(stderr, "ERROR: Unable to count raw input devices.\n");
        return;
    }
    
    //fprintf(stderr,"testit2\n");
    // Allocate the array to hold the DeviceList
    if ((pRawInputDeviceList = new RAWINPUTDEVICELIST[nInputDevices]) == NULL)
    {
        fprintf(stderr, "ERROR: Unable to allocate memory for raw input device list.\n");
        return;
    }

    // 2nd call to GetRawInputDeviceList: Pass the pointer to our DeviceList and GetRawInputDeviceList() will fill the array
    if (/* GetRawInputDeviceList */ (*_GRIDL)(pRawInputDeviceList, &nInputDevices, sizeof(RAWINPUTDEVICELIST)) == -1)
    {
        fprintf(stderr, "ERROR: Unable to get raw input device list.\n");
        return;
    }
    
    // Loop through all devices and count the mice
    for (i = 0; i < nInputDevices; i++)
    {
        /* Get the device name and use it to determine if it's the RDP Terminal Services virtual device. */

        // 1st call to GetRawInputDeviceInfo: Pass NULL to get the size of the device name
        if (/* GetRawInputDeviceInfo */ (*_GRIDIA)(pRawInputDeviceList[i].hDevice, RIDI_DEVICENAME, NULL, &nSize) != 0)
        {
            fprintf(stderr, "ERROR: Unable to get size of raw input device name.\n");
            return;
        }

        // Allocate the array to hold the name
        if ((psName = new char[nSize * sizeof(TCHAR)]) == NULL)
        {
            fprintf(stderr, "ERROR: Unable to allocate memory for device name.\n");
            return;
        }

        // 2nd call to GetRawInputDeviceInfo: Pass our pointer to get the device name
        if ((int)/* GetRawInputDeviceInfo */ (*_GRIDIA)(pRawInputDeviceList[i].hDevice, RIDI_DEVICENAME, psName, &nSize) < 0)
        {
            fprintf(stderr, "ERROR: Unable to get raw input device name %d.\n",i);
            psName = new char[100];
            strcpy(psName,"unknown(failed to get Name)");
            //return;
        }
        std::string xmlString = psName+4; // skip backslashes and questionmarks
        escape(xmlString);
        HANDLE h = CreateFile(
            psName,
            GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            nullptr,
            OPEN_EXISTING,
            0,
            nullptr);
        std::wcout << "ProductName: " << GetDeviceProductName(h) << "\t\t";
        CloseHandle(h);
        
        if(pRawInputDeviceList[i].dwType == RIM_TYPEMOUSE)
            fprintf(stderr, "Mouse%d: %s\n", i, xmlString.c_str());
        else if(pRawInputDeviceList[i].dwType == RIM_TYPEKEYBOARD)
            fprintf(stderr, "Keyboard%d: %s\n", i, xmlString.c_str());
        else if(pRawInputDeviceList[i].dwType == RIM_TYPEHID)
            fprintf(stderr, "HID%d: %s\n", i, xmlString.c_str());
        else
            fprintf(stderr, "Unknown type %d %d: %s\n", pRawInputDeviceList[i].dwType, i, xmlString.c_str());
        delete[] psName;
    }


    // Allocate the array for the raw mice
    if ((rawDevices = new RAW_MOUSE[nInputDevices]) == NULL)
    {
        fprintf(stderr, "ERROR: Unable to allocate memory for raw input mice.\n");
        return;
    }
    

    // Loop through all devices and set the device handles and initialize the device values
    for (i = 0; i < nInputDevices; i++)
    {
        // 1st call to GetRawInputDeviceInfo: Pass NULL to get the size of the device name
        if (/* GetRawInputDeviceInfo */ (*_GRIDIA)(pRawInputDeviceList[i].hDevice, RIDI_DEVICENAME, NULL, &nSize) != 0)
        {
            fprintf(stderr, "ERROR: Unable to get size of raw input device name (2).\n");
            return;
        }

        // Allocate the array to hold the name
        if ((rawDevices[i].deviceName = new char[nSize * sizeof(TCHAR)]) == NULL)
        {
            fprintf(stderr, "ERROR: Unable to allocate memory for raw input device name (2).\n");
            return;
        }

        // 2nd call to GetRawInputDeviceInfo: Pass our pointer to get the device name
        if ((int)/* GetRawInputDeviceInfo */ (*_GRIDIA)(pRawInputDeviceList[i].hDevice, RIDI_DEVICENAME, rawDevices[i].deviceName, &nSize) < 0)
        {
            fprintf(stderr, "ERROR: Unable to get raw input device name (2).\n");
            return;
        }

        rawDevices[i].device_handle = pRawInputDeviceList[i].hDevice;
        rawDevices[i].x = 0;
        rawDevices[i].y = 0;
        rawDevices[i].z = 0;
        rawDevices[i].is_absolute = 0;
        rawDevices[i].is_virtual_desktop = 0;
        rawDevices[i].type = pRawInputDeviceList[i].dwType;
		for(int n=0;n<MAX_RAW_MOUSE_BUTTONS;n++)
		rawDevices[i].buttonpressed[n] = 0;
    }

    delete[] pRawInputDeviceList;

    for (i = 0; i < nInputDevices; i++)
    {
        for (j = 0; j < MAX_RAW_MOUSE_BUTTONS; j++)
        {
            rawDevices[i].buttonpressed[j] = 0;

            // Create the name for this button
            sprintf(buffer, "Button %i", j);
            rawDevices[i].button_name[j] = new char[strlen(buffer) + 1];
            strcpy(rawDevices[i].button_name[j], buffer);
        }
    }

}

std::wstring coRawDeviceManager::GetDeviceProductName(HANDLE deviceHandle)
{
    wchar_t buffer[256];

    if (HidD_GetProductString(deviceHandle, buffer, sizeof(buffer)))
    {
        return buffer;
    }

    return L"Unknown Device";
}

void coRawDeviceManager::update() // read all pending messages if any and process them
{

    MSG msg;
    while (PeekMessage(&msg, handle_, 0, 0, PM_REMOVE))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}
