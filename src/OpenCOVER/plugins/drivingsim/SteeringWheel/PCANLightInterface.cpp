/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "PCANLightInterface.h"

// This Help function intent to load the requested PCANLight Dll
// using first the GetModuleHandle Win32 API function, in order to
// prevent the multiple attachment of a library to the Application.
// If the library is not attached, will be done with the Win32 Api
// LoadLibrary function.
//
HINSTANCE CANLight::GetDllHandle(HardwareType HWType)
{
    HINSTANCE CANLightDll;

    switch (HWType)
    {
    case ISA_1CH:
        CANLightDll = GetModuleHandle(TEXT("PCAN_ISA"));
        return (CANLightDll == NULL) ? LoadLibrary(TEXT("PCAN_ISA")) : CANLightDll;

    case ISA_2CH:
        CANLightDll = GetModuleHandle(TEXT("PCAN_2ISA"));
        return (CANLightDll == NULL) ? LoadLibrary(TEXT("PCAN_2ISA")) : CANLightDll;

    case PCI_1CH:
        CANLightDll = GetModuleHandle(TEXT("PCAN_PCI"));
        return (CANLightDll == NULL) ? LoadLibrary(TEXT("PCAN_PCI")) : CANLightDll;

    case PCI_2CH:
        CANLightDll = GetModuleHandle(TEXT("PCAN_2PCI"));
        return (CANLightDll == NULL) ? LoadLibrary(TEXT("PCAN_2PCI")) : CANLightDll;

    case PCC_1CH:
        CANLightDll = GetModuleHandle(TEXT("PCAN_PCC"));
        return (CANLightDll == NULL) ? LoadLibrary(TEXT("PCAN_PCC")) : CANLightDll;

    case PCC_2CH:
        CANLightDll = GetModuleHandle(TEXT("PCAN_2PCC"));
        return (CANLightDll == NULL) ? LoadLibrary(TEXT("PCAN_2PCC")) : CANLightDll;

    case USB:
        CANLightDll = GetModuleHandle(TEXT("PCAN_USB"));
        return (CANLightDll == NULL) ? LoadLibrary(TEXT("PCAN_USB")) : CANLightDll;

    case DNP:
        CANLightDll = GetModuleHandle(TEXT("PCAN_DNP"));
        return (CANLightDll == NULL) ? LoadLibrary(TEXT("PCAN_DNP")) : CANLightDll;

    case DNG:
        CANLightDll = GetModuleHandle(TEXT("PCAN_DNG"));
        return (CANLightDll == NULL) ? LoadLibrary(TEXT("PCAN_DNG")) : CANLightDll;

    // Hardware is not valid for this function
    //
    default:
        return NULL;
    }
}

// This Help function convert a given string into its equivalent
// Hexadecimal value as DWORD
//
/*
DWORD CANLight::HexTextToInt(LPCSTR ToConvert)
{
    DWORD iToReturn = 0;
    int iExp = 0;
    char chByte;

    // The string to convert is empty
    //
    if(ToConvert == "")
        return 0;
    // The string have more than 8 character (the equivalent value
    // exeeds the DWORD capacyty
    //
    if(ToConvert.GetLength() > 8)
        return 0;
    // We convert any character to its Upper case
    //
	ToConvert = ToConvert.MakeUpper();

    try
    {
        // We calculate the number using the Hex To Decimal formula
        //
        for(int i= ToConvert.GetLength()-1; i >= 0; i--){
            chByte = ToConvert[i];
            switch(int(chByte)){
                case 65:
                    iToReturn += (DWORD)(10*pow(16.0f,iExp));
                    break;
                case 66:
                    iToReturn += (DWORD)(11*pow(16.0f,iExp));
                    break;
                case 67:
                    iToReturn += (DWORD)(12*pow(16.0f,iExp));
                    break;
                case 68:
                    iToReturn += (DWORD)(13*pow(16.0f,iExp));
                    break;
                case 69:
                    iToReturn += (DWORD)(14*pow(16.0f,iExp));
                    break;
                case 70:
                    iToReturn += (DWORD)(15*pow(16.0f,iExp));
                    break;
                default:
                    if((int(chByte) <48)||(int(chByte)>57))
                        return -1;
                    iToReturn += (DWORD)(atoi(&chByte)*pow(16.0f,iExp));
                    break;

            }
            iExp++;
        }
    }
    catch(...)
    {
        // Error, return 0
        //
        return 0;
    }

    return iToReturn;
}
*/
// PCANLight Init function for non Plug and Play Hardware.
// This function make the following:
//		- Activate a Hardware
//		- Make a Register Test of 82C200/SJA1000
//		- Allocate a Send buffer and a Hardware handle
//		- Programs the configuration of the transmit/receive driver
//		- Set the Baudrate register
//		- Set the Controller in RESET condition
//
// "HWType" = Which hardware should be initialized
// "BTR0BTR1" = BTR0-BTR1 baudrate register
// "MsgType" = If the frame type is standard or extended
// "IO_Port" = Input/output Port Address of the hardware
// "Interrupt" = Interrupt number
// RETURN = A CANResult value - Error/status of the hardware after execute the function
//
CANResult CANLight::Init(HardwareType HWType, Baudrates BTR0BTR1, FramesType MsgType, DWORD IO_Port, WORD Interrupt)
{
    HINSTANCE CANLightDll;
    InitNPAP PCLight_Init;
    Hardware HWToInit = HW_ISA_SJA;
    CANResult Res;
    LPCSTR FunctionName;

    CANLightDll = GetDllHandle(HWType);

    switch (HWType)
    {
    case ISA_1CH:
        FunctionName = "CAN_Init";
        HWToInit = HW_ISA_SJA;
        break;

    case ISA_2CH:
        FunctionName = "CAN2_Init";
        HWToInit = HW_ISA_SJA;
        break;

    case DNG:
        FunctionName = "CAN_Init";
        HWToInit = HW_DONGLE_SJA;
        break;

    case DNP:
        FunctionName = "CAN_Init";
        HWToInit = HW_DONGLE_PRO;
        break;

    // Hardware is not valid for this function
    //
    default:
        return ERR_ILLHW;
    }

    if (CANLightDll != NULL)
    {
        PCLight_Init = (InitNPAP)GetProcAddress(CANLightDll, FunctionName);
        if (PCLight_Init != NULL)
            Res = (CANResult)PCLight_Init(BTR0BTR1, MsgType, HWToInit, IO_Port, Interrupt);
        return Res;
    }

    // Error: Dll does not exists or the Init function is not available
    //
    ::MessageBox(NULL, "Error: \"DLL could not be loaded!\"", "Error!", MB_ICONERROR);
    return ERR_NO_DLL;
}

// PCANLight Init function for Plug and Play Hardware.
// This function make the following:
//		- Activate a Hardware
//		- Make a Register Test of 82C200/SJA1000
//		- Allocate a Send buffer and a Hardware handle
//		- Programs the configuration of the transmit/receive driver
//		- Set the Baudrate register
//		- Set the Controller in RESET condition
//
// "HWType" = Which hardware should be initialized
// "BTR0BTR1" = BTR0-BTR1 baudrate register
// "MsgType" = If the frame type is standard or extended
// RETURN = A CANResult value - Error/status of the hardware after execute the function
//
CANResult CANLight::Init(HardwareType HWType, Baudrates BTR0BTR1, FramesType MsgType)
{
    HINSTANCE CANLightDll;
    InitPAP PCLight_Init;
    CANResult Res;
    LPCSTR FunctionName;

    // Hardware is not valid for this function
    //
    if ((HWType != PCI_1CH) && (HWType != PCI_2CH) && (HWType != PCC_1CH) && (HWType != PCC_2CH) && (HWType != USB))
        return ERR_ILLHW;

    CANLightDll = GetDllHandle(HWType);

    if ((HWType == PCI_2CH) || (HWType == PCC_2CH))
        FunctionName = "CAN2_Init";
    else
        FunctionName = "CAN_Init";

    if (CANLightDll != NULL)
    {
        PCLight_Init = (InitPAP)GetProcAddress(CANLightDll, FunctionName);
        if (PCLight_Init != NULL)
            Res = (CANResult)PCLight_Init(BTR0BTR1, MsgType);
        return Res;
    }

    // Error: Dll does not exists or the Init function is not available
    //
    ::MessageBox(NULL, "Error: \"DLL could not be loaded!\"", "Error!", MB_ICONERROR);
    return ERR_NO_DLL;
}

// PCANLight Close function.
// This function terminate and release all resources and the configured hardware:
//
// "HWType" = Which hardware should be finished
// RETURN = A CANResult value - Error/status of the hardware after execute the function
//
CANResult CANLight::Close(HardwareType HWType)
{
    HINSTANCE CANLightDll;
    C_S_RC_RF PCLight_Close;
    CANResult Res;
    LPCSTR FunctionName;

    CANLightDll = GetDllHandle(HWType);

    if ((HWType == ISA_2CH) || (HWType == PCI_2CH) || (HWType == PCC_2CH))
        FunctionName = "CAN2_Close";
    else
        FunctionName = "CAN_Close";

    if (CANLightDll != NULL)
    {
        PCLight_Close = (C_S_RC_RF)GetProcAddress(CANLightDll, FunctionName);
        if (PCLight_Close != NULL)
            Res = (CANResult)PCLight_Close();
        return Res;
    }

    // Error: Dll does not exists or the Init function is not available
    //
    ::MessageBox(NULL, "Error: \"DLL could not be loaded!\"", "Error!", MB_ICONERROR);
    return ERR_NO_DLL;
}

// PCANLight Status Function
// This function request the current status of the hardware (b.e. BUS-OFF)
//
// "HWType" = Which hardware should be asked for it Status
// RETURN = A CANResult value - Error/status of the hardware after execute the function
CANResult CANLight::Status(HardwareType HWType)
{
    HINSTANCE CANLightDll;
    C_S_RC_RF PCLight_Status;
    CANResult Res;
    LPCSTR FunctionName;

    CANLightDll = GetDllHandle(HWType);

    if ((HWType == ISA_2CH) || (HWType == PCI_2CH) || (HWType == PCC_2CH))
        FunctionName = "CAN2_Status";
    else
        FunctionName = "CAN_Status";

    if (CANLightDll != NULL)
    {
        PCLight_Status = (C_S_RC_RF)GetProcAddress(CANLightDll, FunctionName);
        if (PCLight_Status != NULL)
            Res = (CANResult)PCLight_Status();
        return Res;
    }

    // Error: Dll does not exists or the Init function is not available
    //
    ::MessageBox(NULL, "Error: \"DLL could not be loaded!\"", "Error!", MB_ICONERROR);
    return ERR_NO_DLL;
}

// PCANLight Write function
// This function Place a CAN message into the Transmit Queue of the CAN Hardware
//
// "HWType" = In which hardware should be written the CAN Message
// "MsgToSend" = The TPCANMsg message to be written
// RETURN = A CANResult value - Error/status of the hardware after execute the function
//
CANResult CANLight::Write(HardwareType HWType, TPCANMsg *MsgToSend)
{
    HINSTANCE CANLightDll;
    ReadWrite PCLight_Write;
    CANResult Res;
    LPCSTR FunctionName;

    // Not memory allocated  for the message
    //
    if (MsgToSend == NULL)
        return ERR_PARMVAL;

    CANLightDll = GetDllHandle(HWType);

    if ((HWType == ISA_2CH) || (HWType == PCI_2CH) || (HWType == PCC_2CH))
        FunctionName = "CAN2_Write";
    else
        FunctionName = "CAN_Write";

    if (CANLightDll != NULL)
    {
        PCLight_Write = (ReadWrite)GetProcAddress(CANLightDll, FunctionName);
        if (PCLight_Write != NULL)
            Res = (CANResult)PCLight_Write(MsgToSend);
        return Res;
    }

    // Error: Dll does not exists or the Init function is not available
    //
    ::MessageBox(NULL, "Error: \"DLL could not be loaded!\"", "Error!", MB_ICONERROR);
    return ERR_NO_DLL;
}

// PCANLight Read function
// This function get the next message or the next error from the Receive Queue of
// the CAN Hardware.
// REMARK:
//		- Check always the type of the received Message (MSGTYPE_STANDARD,MSGTYPE_RTR,
//		  MSGTYPE_EXTENDED,MSGTYPE_STATUS)
//		- The function will return ERR_OK always that you receive a CAN message successfully
//		  although if the messages is a MSGTYPE_STATUS message.
//		- When a MSGTYPE_STATUS mesasge is got, the ID and Length information of the message
//		  will be treated as indefined values. Actually information of the received message
//		  should be interpreted using the first 4 data bytes as follow:
//			*	Data0	Data1	Data2	Data3	Kind of Error
//				0x00	0x00	0x00	0x02	CAN_ERR_OVERRUN		0x0002	CAN Controller was read to late
//				0x00	0x00	0x00	0x04	CAN_ERR_BUSLIGHT	0x0004  Bus Error: An error counter limit reached (96)
//				0x00	0x00	0x00	0x08	CAN_ERR_BUSHEAVY	0x0008	Bus Error: An error counter limit reached (128)
//				0x00	0x00	0x00	0x10	CAN_ERR_BUSOFF		0x0010	Bus Error: Can Controller went "Bus-Off"
//		- If a CAN_ERR_BUSOFF status message is received, the CAN Controller must to be
//		  initialized again using the Init() function.  Otherwise, will be not possible
//		  to send/receive more messages.
//
// "HWType" = From which hardware should be read a CAN Message
// "Msg" = The TPCANMsg structure to store the CAN message
// RETURN = A CANResult value - Error/status of the hardware after execute the function
//
CANResult CANLight::Read(HardwareType HWType, TPCANMsg *MsgBuffer)
{
    HINSTANCE CANLightDll;
    ReadWrite PCLight_Read;
    CANResult Res;
    LPCSTR FunctionName;

    // Not memory allocated  for the message
    //
    if (MsgBuffer == NULL)
        return ERR_PARMVAL;

    CANLightDll = GetDllHandle(HWType);

    if ((HWType == ISA_2CH) || (HWType == PCI_2CH) || (HWType == PCC_2CH))
        FunctionName = "CAN2_Read";
    else
        FunctionName = "CAN_Read";

    if (CANLightDll != NULL)
    {
        PCLight_Read = (ReadWrite)GetProcAddress(CANLightDll, FunctionName);
        if (PCLight_Read != NULL)
            Res = (CANResult)PCLight_Read(MsgBuffer);
        return Res;
    }

    // Error: Dll does not exists or the Init function is not available
    //
    ::MessageBox(NULL, "Error: \"DLL could not be loaded!\"", "Error!", MB_ICONERROR);
    return ERR_NO_DLL;
}

// PCANLight VersionInfo function
// This function get the Version and copyright of the hardware as text
// (max. 255 characters)
//
// "HWType"  = Which hardware should be asked for its Version information
// "strInfo" = String variable to return the hardware information
// RETURN = A CANResult value - Error/status of the hardware after execute the function
//
CANResult CANLight::VersionInfo(HardwareType HWType, LPCSTR &strInfo)
{
    HINSTANCE CANLightDll;
    GetInfo PCLight_VersionInfo;
    CANResult Res;
    LPCSTR FunctionName;
    char ToRead[255];

    memset(ToRead, '\0', 254);

    CANLightDll = GetDllHandle(HWType);

    if ((HWType == ISA_2CH) || (HWType == PCI_2CH) || (HWType == PCC_2CH))
        FunctionName = "CAN2_VersionInfo";
    else
        FunctionName = "CAN_VersionInfo";

    if (CANLightDll != NULL)
    {
        PCLight_VersionInfo = (GetInfo)GetProcAddress(CANLightDll, FunctionName);
        if (PCLight_VersionInfo != NULL)
            Res = (CANResult)PCLight_VersionInfo(ToRead);
        strInfo = ToRead;
        return Res;
    }

    // Error: Dll does not exists or the Init function is not available
    //
    ::MessageBox(NULL, "Error: \"DLL could not be loaded!\"", "Error!", MB_ICONERROR);
    return ERR_NO_DLL;
}

// PCANLight SpecialFunktion function
// This function is an special function to be used "ONLY" for distributors
//
// "HWType" = Hardware within use this special function
// "DistributorCode" = Distributor Identification number
// "CodeNumber" = Number code
// RETURN = Literal Numeric value of the CANResult: 1 if the given parameters
// and the parameters in the hardware agree, 0 otherwise
//
CANResult CANLight::SpecialFunktion(HardwareType HWType, unsigned long DistributorCode, int CodeNumber)
{
    HINSTANCE CANLightDll;
    spcFunct PCLight_SpecialFunktion;
    CANResult Res;

    // Hardware is not valid for this function
    //
    if ((HWType != PCI_1CH) && (HWType != PCC_1CH) && (HWType != USB) && (HWType != DNG))
        return ERR_ILLHW;

    CANLightDll = GetDllHandle(HWType);

    if (CANLightDll != NULL)
    {
        PCLight_SpecialFunktion = (spcFunct)GetProcAddress(CANLightDll, TEXT("CAN_SpecialFunktion"));
        if (PCLight_SpecialFunktion != NULL)
            Res = (CANResult)PCLight_SpecialFunktion(DistributorCode, CodeNumber);
        return Res;
    }

    // Error: Dll does not exists or the Init function is not available
    //
    ::MessageBox(NULL, "Error: \"DLL could not be loaded!\"", "Error!", MB_ICONERROR);
    return ERR_NO_DLL;
}

// PCANLight GetDLL2_Version function
// This function work only for the PCI 2 Channels hardware. It is used to get
// the Version and copyright of the DLL as text (max. 255 characters)
//
// "Version" = String variable to return the hardware information
// RETURN = Literal Numeric value of the CANResult: 0xFFFFFFFF if the
// variable is not valid, otherwise is OK
//
CANResult CANLight::GetDLL2_Version(LPCSTR &Version)
{
    HINSTANCE CANLightDll;
    GetInfo PCLight_GetDLL2_Version;
    CANResult Res;
    char ToRead[255];

    memset(ToRead, '\0', 254);

    CANLightDll = GetDllHandle(PCI_2CH);

    if (CANLightDll != NULL)
    {
        PCLight_GetDLL2_Version = (GetInfo)GetProcAddress(CANLightDll, TEXT("GetDLL2_Version"));
        if (PCLight_GetDLL2_Version != NULL)
            Res = (CANResult)PCLight_GetDLL2_Version(ToRead);
        Version = ToRead;
        return Res;
    }

    // Error: Dll does not exists or the Init function is not available
    //
    ::MessageBox(NULL, "Error: \"DLL could not be loaded!\"", "Error!", MB_ICONERROR);
    return ERR_NO_DLL;
}

// PCANLight ResetClient function
// This function delete the both queues (Transmit,Receive) of the CAN Controller
// using a RESET
//
// "HWType" = Hardware to reset
// RETURN = A CANResult value - Error/status of the hardware after execute the function
CANResult CANLight::ResetClient(HardwareType HWType)
{
    HINSTANCE CANLightDll;
    C_S_RC_RF PCLight_ResetClient;
    CANResult Res;
    LPCSTR FunctionName;

    CANLightDll = GetDllHandle(HWType);

    if ((HWType == ISA_2CH) || (HWType == PCI_2CH) || (HWType == PCC_2CH))
        FunctionName = "CAN2_ResetClient";
    else
        FunctionName = "CAN_ResetClient";

    if (CANLightDll != NULL)
    {
        PCLight_ResetClient = (C_S_RC_RF)GetProcAddress(CANLightDll, FunctionName);
        if (PCLight_ResetClient != NULL)
            Res = (CANResult)PCLight_ResetClient();
        return Res;
    }

    // Error: Dll does not exists or the Init function is not available
    //
    ::MessageBox(NULL, "Error: \"DLL could not be loaded!\"", "Error!", MB_ICONERROR);
    return ERR_NO_DLL;
}

// PCANLigth MsgFilter function
// This function set the receive message filter of the CAN Controller.
// REMARK:
//		- A quick register of all messages is possible using the parameters From and To as 0
//		- Every call of this function maybe cause an extention of the receive filter of the
//		  CAN controller, which one can go briefly to RESET
//		- New in Ver 2.x:
//			* Standard frames will be put it down in the acc_mask/code as Bits 28..13
//			* Hardware driver for 82C200 must to be moved to Bits 10..0 again!
//	WARNING:
//		It is not guaranteed to receive ONLY the registered messages.
//
// "HWType" = Hardware which applay the filter to
// "From" = First/Start Message ID - It muss be smaller than the "To" parameter
// "To" = Last/Finish Message ID - It muss be bigger than the "From" parameter
// "MsgType" = Kind of Frame - Standard or Extended
// RETURN = A CANResult value - Error/status of the hardware after execute the function
//
CANResult CANLight::MsgFilter(HardwareType HWType, DWORD From, DWORD To, FramesType MsgType)
{
    HINSTANCE CANLightDll;
    MessageFilter PCLight_MsgFilter;
    CANResult Res;
    LPCSTR FunctionName;

    CANLightDll = GetDllHandle(HWType);

    if ((HWType == ISA_2CH) || (HWType == PCI_2CH) || (HWType == PCC_2CH))
        FunctionName = "CAN2_MsgFilter";
    else
        FunctionName = "CAN_MsgFilter";

    if (CANLightDll != NULL)
    {
        PCLight_MsgFilter = (MessageFilter)GetProcAddress(CANLightDll, FunctionName);
        if (PCLight_MsgFilter != NULL)
            Res = (CANResult)PCLight_MsgFilter(From, To, MsgType);
        return Res;
    }

    // Error: Dll does not exists or the Init function is not available
    //
    ::MessageBox(NULL, "Error: \"DLL could not be loaded!\"", "Error!", MB_ICONERROR);
    return ERR_NO_DLL;
}

// PCANLigth ResetFilter function
// This function close completely the Message Filter of the Hardware.
// They will be no more messages received.
//
// "HWType" = Hardware to reset its filter
// RETURN = A CANResult value - Error/status of the hardware after execute the function</returns>
//
CANResult CANLight::ResetFilter(HardwareType HWType)
{
    HINSTANCE CANLightDll;
    C_S_RC_RF PCLight_ResetFilter;
    CANResult Res;
    LPCSTR FunctionName;

    CANLightDll = GetDllHandle(HWType);

    if ((HWType == ISA_2CH) || (HWType == PCI_2CH) || (HWType == PCC_2CH))
        FunctionName = "CAN2_ResetFilter";
    else
        FunctionName = "CAN_ResetFilter";

    if (CANLightDll != NULL)
    {
        PCLight_ResetFilter = (C_S_RC_RF)GetProcAddress(CANLightDll, FunctionName);
        if (PCLight_ResetFilter != NULL)
            Res = (CANResult)PCLight_ResetFilter();
        return Res;
    }

    // Error: Dll does not exists or the Init function is not available
    //
    ::MessageBox(NULL, "Error: \"DLL could not be loaded!\"", "Error!", MB_ICONERROR);
    return ERR_NO_DLL;
}

// PCANLight SetUSBDeviceNr function
// This function set an identification number to the USB CAN hardware
//
// "DeviceNumber" = Value to be set as Device Number
// RETURN = A CANResult value - Error/status of the hardware after execute the function
//
CANResult CANLight::SetUSBDeviceNr(long DeviceNumber)
{
    HINSTANCE CANLightDll;
    USBSDevNr PCLight_SetUSBDeviceNr;
    CANResult Res;

    CANLightDll = GetDllHandle(USB);

    if (CANLightDll != NULL)
    {
        PCLight_SetUSBDeviceNr = (USBSDevNr)GetProcAddress(CANLightDll, TEXT("SetUSBDeviceNr"));
        if (PCLight_SetUSBDeviceNr != NULL)
            Res = (CANResult)PCLight_SetUSBDeviceNr(DeviceNumber);
        return Res;
    }

    // Error: Dll does not exists or the Init function is not available
    //
    ::MessageBox(NULL, "Error: \"DLL could not be loaded!\"", "Error!", MB_ICONERROR);
    return ERR_NO_DLL;
}

// PCANLight GetUSBDeviceNr function
// This function read the device number of a USB CAN Hardware
//
// "DeviceNumber" = Variable to return the Device Number value
// RETURN = A CANResult value - Error/status of the hardware after execute the function
//
CANResult CANLight::GetUSBDeviceNr(long *DeviceNumber)
{
    HINSTANCE CANLightDll;
    USBGDevNr PCLight_GetUSBDeviceNr;
    CANResult Res;

    CANLightDll = GetDllHandle(USB);

    if (CANLightDll != NULL)
    {
        PCLight_GetUSBDeviceNr = (USBGDevNr)GetProcAddress(CANLightDll, TEXT("GetUSBDeviceNr"));
        if (PCLight_GetUSBDeviceNr != NULL)
            Res = (CANResult)PCLight_GetUSBDeviceNr(DeviceNumber);
        return Res;
    }

    // Error: Dll does not exists or the Init function is not available
    //
    ::MessageBox(NULL, "Error: \"DLL could not be loaded!\"", "Error!", MB_ICONERROR);
    return ERR_NO_DLL;
}