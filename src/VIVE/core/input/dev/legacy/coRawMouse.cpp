/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coRawMouse.h"
#include <stdio.h>

using namespace opencover;
coRawMouse::coRawMouse(int n)
{
    buttonNumber = n;
}

coRawMouse::coRawMouse(const char *deviceName)
{
    buttonNumber = 0;
    if (deviceName == NULL)
        return;
    int i;
    char *devName = new char[strlen(deviceName) + 1];
    strcpy(devName, deviceName);
    for (i = 0; i < strlen(devName); i++)
    {
        if (devName[i] == '\\')
            devName[i] = '#';
    }
    for (i = 0; i < coRawMouseManager::instance()->numMice(); i++)
    {
        fprintf(stderr, "looking for:%s\n", devName);
        fprintf(stderr, "try        :%s\n", coRawMouseManager::instance()->rawMice[i].mouseName + 4);
        if (strncasecmp(deviceName, coRawMouseManager::instance()->rawMice[i].mouseName + 4, strlen(deviceName)) == 0)
        {
            buttonNumber = i;

            fprintf(stderr, "found:%d\n", i);
            break;
        }
        if (strncasecmp(devName, coRawMouseManager::instance()->rawMice[i].mouseName + 4, strlen(devName)) == 0)
        {
            buttonNumber = i;
            break;
        }
    }
    delete[] devName;
}

coRawMouse::~coRawMouse()
{
}

int coRawMouse::getX()
{
    return coRawMouseManager::instance()->rawMice[buttonNumber].x;
}
int coRawMouse::getY()
{
    return coRawMouseManager::instance()->rawMice[buttonNumber].y;
}
int coRawMouse::getWheelCount()
{
    return coRawMouseManager::instance()->rawMice[buttonNumber].z;
}
bool coRawMouse::getButton(int i)
{
    return coRawMouseManager::instance()->is_raw_mouse_button_pressed(buttonNumber, i) != 0;
}

unsigned int coRawMouse::getButtonBits()
{
    unsigned int bits = 0;
    for (int i = 0; i < MAX_RAW_MOUSE_BUTTONS; i++)
    {
        if (coRawMouseManager::instance()->is_raw_mouse_button_pressed(buttonNumber, i))
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
//	numMice
//============================================================

int coRawMouseManager::numMice()
{
    return nnumMice;
}

//============================================================
//	is_rm_rdp_mouse
//============================================================

BOOL coRawMouseManager::is_rm_rdp_mouse(char cDeviceString[])
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
//	register_raw_mouse
//============================================================

BOOL coRawMouseManager::register_raw_mouse(void)
{
    // This function registers to receive the WM_INPUT messages
    RAWINPUTDEVICE Rid[1]; // Register only for mouse messages from wm_input.

    //register to get wm_input messages
    Rid[0].usUsagePage = 0x01;
    Rid[0].usUsage = 0x02;
    Rid[0].dwFlags = RIDEV_INPUTSINK; // RIDEV_NOLEGACY;   // adds HID mouse and also ignores legacy mouse messages
    Rid[0].hwndTarget = handle_;

    // Register to receive the WM_INPUT message for any change in mouse (buttons, wheel, and movement will all generate the same message)
    if (!/* RegisterRawInputDevices*/ (*_RRID)(Rid, 1, sizeof(Rid[0])))
        return 0;

    return 1;
}

//============================================================
//	read_raw_input
//============================================================

BOOL coRawMouseManager::read_raw_input(PRAWINPUT raw)
{
    // should be static when I get around to it

    int i;

    // mouse 0 is sysmouse, so if there is not sysmouse start loop @0
    i = 0;
    if (IncludeSysMouse)
        i++;

    for (; i < (nnumMice + excluded_sysmouse_devices_count); i++)
    {
        if (rawMice[i].device_handle == raw->header.hDevice)
        {
            // Update the values for the specified mouse
            if (IncludeIndividualMice)
            {
                if (rawMice[i].is_absolute)
                {
                    rawMice[i].x = raw->data.mouse.lLastX;
                    rawMice[i].y = raw->data.mouse.lLastY;
                }
                else
                { // relative
                    rawMice[i].x += raw->data.mouse.lLastX;
                    rawMice[i].y += raw->data.mouse.lLastY;
                }
                //fprintf(stderr,"Raw Mouse Event: num: %d Flags: %d\n",i,raw->data.mouse.usButtonFlags);
                if (raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_1_DOWN)
                    rawMice[i].buttonpressed[0] = 1;
                if (raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_1_UP)
                    rawMice[i].buttonpressed[0] = 0;
                if (raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_2_DOWN)
                    rawMice[i].buttonpressed[1] = 1;
                if (raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_2_UP)
                    rawMice[i].buttonpressed[1] = 0;
                if (raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_3_DOWN)
                    rawMice[i].buttonpressed[2] = 1;
                if (raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_3_UP)
                    rawMice[i].buttonpressed[2] = 0;
                if (raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_4_DOWN)
                    rawMice[i].buttonpressed[3] = 1;
                if (raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_4_UP)
                    rawMice[i].buttonpressed[3] = 0;
                if (raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_5_DOWN)
                    rawMice[i].buttonpressed[4] = 1;
                if (raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_5_UP)
                    rawMice[i].buttonpressed[4] = 0;

                if (raw->data.mouse.usFlags & MOUSE_MOVE_ABSOLUTE)
                    rawMice[i].is_absolute = 1;
                else if (raw->data.mouse.usFlags & MOUSE_MOVE_RELATIVE)
                    rawMice[i].is_absolute = 0;
                if (raw->data.mouse.usFlags & MOUSE_VIRTUAL_DESKTOP)
                    rawMice[i].is_virtual_desktop = 1;
                else
                    rawMice[i].is_virtual_desktop = 0;

                if (raw->data.mouse.usButtonFlags & RI_MOUSE_WHEEL)
                { // If the current message has a mouse_wheel message
                    if ((SHORT)raw->data.mouse.usButtonData > 0)
                    {
                        rawMice[i].z++;
                    }
                    if ((SHORT)raw->data.mouse.usButtonData < 0)
                    {
                        rawMice[i].z--;
                    }
                }
            }

            // Feed the values for every mouse into the system mouse
            if (IncludeSysMouse)
            {
                if (rawMice[i].is_absolute)
                {
                    rawMice[RAW_SYS_MOUSE].x = raw->data.mouse.lLastX;
                    rawMice[RAW_SYS_MOUSE].y = raw->data.mouse.lLastY;
                }
                else
                { // relative
                    rawMice[RAW_SYS_MOUSE].x += raw->data.mouse.lLastX;
                    rawMice[RAW_SYS_MOUSE].y += raw->data.mouse.lLastY;
                }

                // This is innacurate:  If 2 mice have their buttons down and I lift up on one, this will register the
                //   system mouse as being "up".  I checked out on my windows desktop, and Microsoft was just as
                //   lazy as I'm going to be.  Drag an icon with the 2 left mouse buttons held down & let go of one.

                if (raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_1_DOWN)
                    rawMice[RAW_SYS_MOUSE].buttonpressed[0] = 1;
                if (raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_1_UP)
                    rawMice[RAW_SYS_MOUSE].buttonpressed[0] = 0;
                if (raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_2_DOWN)
                    rawMice[RAW_SYS_MOUSE].buttonpressed[1] = 1;
                if (raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_2_UP)
                    rawMice[RAW_SYS_MOUSE].buttonpressed[1] = 0;
                if (raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_3_DOWN)
                    rawMice[RAW_SYS_MOUSE].buttonpressed[2] = 1;
                if (raw->data.mouse.usButtonFlags & RI_MOUSE_BUTTON_3_UP)
                    rawMice[RAW_SYS_MOUSE].buttonpressed[2] = 0;

                // If an absolute mouse is triggered, sys mouse will be considered absolute till the end of time.
                if (raw->data.mouse.usFlags & MOUSE_MOVE_ABSOLUTE)
                    rawMice[RAW_SYS_MOUSE].is_absolute = 1;
                // Same goes for virtual desktop
                if (raw->data.mouse.usFlags & MOUSE_VIRTUAL_DESKTOP)
                    rawMice[RAW_SYS_MOUSE].is_virtual_desktop = 1;

                if (raw->data.mouse.usButtonFlags & RI_MOUSE_WHEEL)
                { // If the current message has a mouse_wheel message
                    if ((SHORT)raw->data.mouse.usButtonData > 0)
                    {
                        rawMice[RAW_SYS_MOUSE].z++;
                    }
                    if ((SHORT)raw->data.mouse.usButtonData < 0)
                    {
                        rawMice[RAW_SYS_MOUSE].z--;
                    }
                }
            }
        }
    }

    return 1;
}

//============================================================
//	is_raw_mouse_button_pressed
//============================================================

BOOL coRawMouseManager::is_raw_mouse_button_pressed(int mousenum, int buttonnum)
{
    // It's ok to ask if buttons are pressed for unitialized mice - just tell 'em no button's pressed
    if (mousenum >= nnumMice || buttonnum >= MAX_RAW_MOUSE_BUTTONS || rawMice == NULL)
        return 0;
    return (rawMice[mousenum].buttonpressed[buttonnum]);
}

//============================================================
//	is_raw_mouse_absolute
//============================================================
BOOL coRawMouseManager::is_raw_mouse_absolute(int mousenum)
{
    return (rawMice[mousenum].is_absolute);
}

//============================================================
//	is_raw_mouse_virtual_desktop
//============================================================
BOOL coRawMouseManager::is_raw_mouse_virtual_desktop(int mousenum)
{
    return (rawMice[mousenum].is_virtual_desktop);
}

//============================================================
//	get_raw_mouse_button_name
//============================================================

char *coRawMouseManager::get_raw_mouse_button_name(int mousenum, int buttonnum)
{
    if (mousenum >= nnumMice || buttonnum >= MAX_RAW_MOUSE_BUTTONS || rawMice == NULL)
        return NULL;
    return (rawMice[mousenum].button_name[buttonnum]);
}

//============================================================
//	processData
//============================================================

BOOL coRawMouseManager::processData(HANDLE in_device_handle)
{
    // When the WM_INPUT message is received, the lparam must be passed to this function to keep a running tally of
    //     every mouse moves to maintain accurate results for get_raw_mouse_?_delta().
    // This function will take the HANDLE of the device and find the device in the rawMice arrayand add the
    //      x and y mousemove values according to the information stored in the RAWINPUT structure.

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
//	get_raw_mouse_x_delta
//============================================================

ULONG coRawMouseManager::get_raw_mouse_x_delta(int mousenum)
{
    ULONG nReturn = 0;

    if (rawMice != NULL && mousenum < nnumMice)
    {
        nReturn = rawMice[mousenum].x;
        if (!rawMice[mousenum].is_absolute)
            rawMice[mousenum].x = 0;
    }

    return nReturn;
}

//============================================================
//	get_raw_mouse_y_delta
//============================================================

ULONG coRawMouseManager::get_raw_mouse_y_delta(int mousenum)
{
    ULONG nReturn = 0;

    if (rawMice != NULL && mousenum < nnumMice)
    {
        nReturn = rawMice[mousenum].y;
        if (!rawMice[mousenum].is_absolute)
            rawMice[mousenum].y = 0;
    }

    return nReturn;
}

//============================================================
//	get_raw_mouse_z_delta
//============================================================

ULONG coRawMouseManager::get_raw_mouse_z_delta(int mousenum)
{
    ULONG nReturn = 0;

    if (rawMice != NULL && mousenum < nnumMice)
    {
        nReturn = rawMice[mousenum].z;
        if (!rawMice[mousenum].is_absolute)
            rawMice[mousenum].z = 0;
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
        coRawMouseManager::instance()->processData((HRAWINPUT)lParam);
    }
    break;
    }
    return DefWindowProc(hwnd, nMsg, wParam, lParam);
}

coRawMouseManager *coRawMouseManager::inst = NULL;

coRawMouseManager *coRawMouseManager::instance()
{
    if (inst == NULL)
    {
        inst = new coRawMouseManager();
    }
    return inst;
}

coRawMouseManager::~coRawMouseManager()
{
    int i, j;
    for (i = 0; i < nnumMice; i++)
    {
        for (j = 0; j < MAX_RAW_MOUSE_BUTTONS; j++)
        {

            delete[] rawMice[i].button_name[j];
        }
        delete[] rawMice[i].mouseName;
    }

    delete[] rawMice;
}

coRawMouseManager::coRawMouseManager()
{

    oldSize = 0;
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
    wndclass.lpszClassName = "mouseInputWindow";

    if (!RegisterClass(&wndclass))
    {
        fprintf(stderr, "RegisterClass Error:%d\n", GetLastError());
    }

    if (!(handle_ = CreateWindowEx(0,
                                   "mouseInputWindow",
                                   TEXT("mouseInputWindow"),
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

    IncludeSysMouse = true;
    IncludeRemoteDeskTopMouse = true;
    IncludeIndividualMice = true;

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

    excluded_sysmouse_devices_count = 0;
    nnumMice = 0;

    if (bHasBeenInitialized)
    {
        fprintf(stderr, "WARNING: rawmouse init called after initialization already completed.");
        bHasBeenInitialized = 1;
        return;
    }

    IncludeSysMouse = IncludeSysMouse;
    IncludeRemoteDeskTopMouse = IncludeRemoteDeskTopMouse;
    IncludeIndividualMice = IncludeIndividualMice;
    rawMice = NULL;
    nnumMice = 0;
    setupDevices();

    // finally, register to recieve raw input WM_INPUT messages
    if (!register_raw_mouse())
    {
        fprintf(stderr, "ERROR: Unable to register raw input (2).\n");
        return;
    }
    bHasBeenInitialized = 1;
}
void coRawMouseManager::setupDevices()
{
    fprintf(stderr, "setup Devices for raw input\n");
    int currentmouse = 0;
    char buffer[80];
    int nInputDevices, i, j;
    PRAWINPUTDEVICELIST pRawInputDeviceList;
    int nSize;
    char *psName;
    for (i = 0; i < nnumMice; i++)
    {
        for (j = 0; j < MAX_RAW_MOUSE_BUTTONS; j++)
        {
            delete[] rawMice[i].button_name[j];
        }
        delete[] rawMice[i].mouseName;
    }
    delete[] rawMice;
    // 1st call to GetRawInputDeviceList: Pass NULL to get the number of devices.
    if (/* GetRawInputDeviceList */ (*_GRIDL)(NULL, &nInputDevices, sizeof(RAWINPUTDEVICELIST)) != 0)
    {
        fprintf(stderr, "ERROR: Unable to count raw input devices.\n");
        return;
    }

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
        if (pRawInputDeviceList[i].dwType == RIM_TYPEMOUSE)
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
                fprintf(stderr, "ERROR: Unable to get raw input device name.\n");
                return;
            }
            fprintf(stderr, "Device%d: %s\n", i, psName);
            // Count this mouse for allocation if it's not an RDP mouse or if we want to include the rdp mouse
            if (is_rm_rdp_mouse(psName))
            {
                if (IncludeRemoteDeskTopMouse)
                    nnumMice++;
            }
            else
            { // It's an ordinary mouse
                nnumMice++;
                if (!IncludeIndividualMice)
                    excluded_sysmouse_devices_count++; // Don't count this in the final nnumMice value
            }
            delete[] psName;
        }
    }

    if (IncludeSysMouse)
        nnumMice++;

    // Allocate the array for the raw mice
    if ((rawMice = new RAW_MOUSE[nnumMice]) == NULL)
    {
        fprintf(stderr, "ERROR: Unable to allocate memory for raw input mice.\n");
        return;
    }

    // Define the sys mouse
    if (IncludeSysMouse)
    {
        rawMice[RAW_SYS_MOUSE].device_handle = 0;
        rawMice[RAW_SYS_MOUSE].x = 0;
        rawMice[RAW_SYS_MOUSE].y = 0;
        rawMice[RAW_SYS_MOUSE].z = 0;
        rawMice[RAW_SYS_MOUSE].is_absolute = 0;
        rawMice[RAW_SYS_MOUSE].is_virtual_desktop = 0;
        rawMice[RAW_SYS_MOUSE].mouseName = new char[9];
        strcpy(rawMice[RAW_SYS_MOUSE].mouseName, "sysMouse");

        currentmouse++;
    }

    // Loop through all devices and set the device handles and initialize the mouse values
    for (i = 0; i < nInputDevices; i++)
    {
        if (pRawInputDeviceList[i].dwType == RIM_TYPEMOUSE)
        {
            // 1st call to GetRawInputDeviceInfo: Pass NULL to get the size of the device name
            if (/* GetRawInputDeviceInfo */ (*_GRIDIA)(pRawInputDeviceList[i].hDevice, RIDI_DEVICENAME, NULL, &nSize) != 0)
            {
                fprintf(stderr, "ERROR: Unable to get size of raw input device name (2).\n");
                return;
            }

            // Allocate the array to hold the name
            if ((rawMice[currentmouse].mouseName = new char[nSize * sizeof(TCHAR)]) == NULL)
            {
                fprintf(stderr, "ERROR: Unable to allocate memory for raw input device name (2).\n");
                return;
            }

            // 2nd call to GetRawInputDeviceInfo: Pass our pointer to get the device name
            if ((int)/* GetRawInputDeviceInfo */ (*_GRIDIA)(pRawInputDeviceList[i].hDevice, RIDI_DEVICENAME, rawMice[currentmouse].mouseName, &nSize) < 0)
            {
                fprintf(stderr, "ERROR: Unable to get raw input device name (2).\n");
                return;
            }

            // Add this mouse to the array if it's not an RDPMouse or if we wish to include the RDP mouse
            if ((!is_rm_rdp_mouse(psName)) || IncludeRemoteDeskTopMouse)
            {
                rawMice[currentmouse].device_handle = pRawInputDeviceList[i].hDevice;
                rawMice[currentmouse].x = 0;
                rawMice[currentmouse].y = 0;
                rawMice[currentmouse].z = 0;
                rawMice[currentmouse].is_absolute = 0;
                rawMice[currentmouse].is_virtual_desktop = 0;

                currentmouse++;
            }
        }
    }

    delete[] pRawInputDeviceList;

    for (i = 0; i < nnumMice; i++)
    {
        for (j = 0; j < MAX_RAW_MOUSE_BUTTONS; j++)
        {
            rawMice[i].buttonpressed[j] = 0;

            // Create the name for this button
            sprintf(buffer, "Button %i", j);
            rawMice[i].button_name[j] = new char[strlen(buffer) + 1];
            strcpy(rawMice[i].button_name[j], buffer);
        }
    }

    nnumMice -= excluded_sysmouse_devices_count;
}
void coRawMouseManager::update() // read all pending messages if any and process them
{

    MSG msg;
    while (PeekMessage(&msg, handle_, 0, 0, PM_REMOVE))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}
