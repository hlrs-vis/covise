/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

///////////////////////////////////////////////////////////////////////////////
//  Based on:
//  PCAN_ISA.h
//  PCAN_2ISA.h
//  PCAN_PCI.h
//  PCAN_2PCI.h
//  PCAN_PCC.h
//  PCAN_2PCC.h
//  PCAN_DNG.h
//  PCAN_DNP.h
//  PCAN_USB.h
//
//  Version 1.0
//
//  ~~~~~~~~~~~~
//
//  Idea:
//
//  ~~~~~~~~~~
//
//  PCANLight is a Unit in charge of make the managing of the different PCAN Hardware using the
//  PCANLight Dlls: pcan_isa,pcan_2isa,pcan_pci,pcan_2pci,pcan_pcc,pcan_2pcc,pcan_dng,pcan_dnp,
//  pcan_usb
//
//  In order  to offer  a simple  interface, some constant valuest were converted  to enumerate
//  types.  The class CANLight  make use of all Dlls and gives  an unique interface for all the
//  hardware types.  In this way, in this high level,  will  exists  only  one  occurrence of each
//  PCANLIGHT function.
//
//  ~~~~~~~~~~~~
//
//  PCAN-Light -API
//
//  ~~~~~~~~~~~~
//
//   Init() (Two versions, for P&P and Non P&P)
//   Close()
//   Status()
//   Write()
//   Read()
//   VersionInfo()
//   SpecialFunktion()
//   GetDLL2_Version()
//   ResetClient()
//   MsgFilter()
//   ResetFilter()
//   SetUSBDeviceNr()
//   GetUSBDeviceNr()
//
//  ------------------------------------------------------------------
//
//  Autor  : Keneth Wagner
//  Sprache: C++ (Visual Studio 2003)
//
//  ------------------------------------------------------------------
//  Copyright (C) 2006  PEAK-System Technik GmbH, Darmstadt
//
#ifndef PCANLightH
#define PCANLightH
#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x501 // This specifies WinXP or later - it is needed to access rawmouse from the user32.dll
#endif
#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#include <winsock2.h>
#include <io.h>
#ifndef PATH_MAX
#define PATH_MAX 512
#endif
#endif
#include <Pcan_pci.h>
#include <Math.h>

#pragma once

// A CAN Message
//
/*
typedef struct {
    DWORD ID;        // 11/29 Bit-ID
    BYTE  MSGTYPE;   // Kind of Message
    BYTE  LEN;       // Number of Data bytes (0..8)
    BYTE  DATA[8];   // Data bytes 0..7
} TPCANMsg;
*/

// Kind of Frame - Message Type
//
enum FramesType
{
    INIT_TYPE_ST = 0x00, //Standart Frame
    INIT_TYPE_EX = 0x01, //Extended Frame
};

// Maximal values for the ID of a CAN Message
//
enum MaxIDValues
{
    MAX_STANDARD_ID = 0x7FF,
    MAX_EXTENDED_ID = 0x1FFFFFFF,
};
/*
// Kind of CAN Message
//
enum MsgTypes
{
    MSGTYPE_STANDARD	= 0x00,		// Standard Frame (11 bit ID)
    MSGTYPE_RTR			= 0x01,		// Remote request
    MSGTYPE_EXTENDED	= 0x02,		// CAN 2.0 B Frame (29 Bit ID)
    MSGTYPE_STATUS		= 0x80,		// Status Message
};
*/
// PCAN Hardware enumeration
//
enum Hardware
{
    HW_INTERN = 0,
    HW_ISA = 1,
    HW_DONGLE = 2,
    HW_DONGLE_EPP = 3,
    HW_PHYT_ISA = 4,
    HW_DONGLE_SJA = 5,
    HW_DONGLE_SJA_EPP = 6,
    HW_DONGLE_PRO = 7,
    HW_DONGLE_PRO_EPP = 8,
    HW_ISA_SJA = 9,
};

// Hardware type corresponding to the different PCAN Light Dlls
//
enum HardwareType
{
    ISA_1CH = 0, // ISA 1 Channel
    ISA_2CH = 1, // ISA 2 Channels
    PCI_1CH = 2, // PCI 1 Channel
    PCI_2CH = 3, // PCI 2 Channels
    PCC_1CH = 4, // PCC 1 Channel
    PCC_2CH = 5, // PCC 2 Channels
    USB = 6, // USB
    DNP = 7, // DONGLE PRO
    DNG = 8, // DONGLE
};

// CAN Baudrates
//
enum Baudrates
{
    BAUD_1M = 0x0014, //   1 MBit/s
    BAUD_500K = 0x001C, // 500 kBit/s
    BAUD_250K = 0x011C, // 250 kBit/s
    BAUD_125K = 0x031C, // 125 kBit/s
    BAUD_100K = 0x432F, // 100 kBit/s
    BAUD_50K = 0x472F, //  50 kBit/s
    BAUD_20K = 0x532F, //  20 kBit/s
    BAUD_10K = 0x672F, //  10 kBit/s
    BAUD_5K = 0x7F7F, //   5 kBit/s
};

// CAN Error and status values
//
enum CANResult
{
    ERR_OK = 0x0000,
    ERR_XMTFULL = 0x0001, // Send buffer of the Controller ist full
    ERR_OVERRUN = 0x0002, // CAN-Controller was read to late
    ERR_BUSLIGHT = 0x0004, // Bus error: an Error count reached the limit
    ERR_BUSHEAVY = 0x0008, // Bus error: an Error count reached the limit
    ERR_BUSOFF = 0x0010, // Bus error: CAN_Controller went to 'Bus-Off'
    ERR_QRCVEMPTY = 0x0020, // RcvQueue is empty
    ERR_QOVERRUN = 0x0040, // RcvQueue was read to late
    ERR_QXMTFULL = 0x0080, // Send queue is full
    ERR_REGTEST = 0x0100, // RegisterTest of the 82C200/SJA1000 failed
    ERR_NOVXD = 0x0200, // Problem with Localization of the VxD
    ERRMASK_ILLHANDLE = 0x1C00, // Mask for all Handle errors
    ERR_HWINUSE = 0x0400, // Hardware is occupied by a net
    ERR_NETINUSE = 0x0800, // The Net is attached to a Client
    ERR_ILLHW = 0x1400, // Invalid Hardware handle
    ERR_ILLNET = 0x1800, // Invalid Net handle
    ERR_ILLCLIENT = 0x1C00, // Invalid Client handle
    ERR_RESOURCE = 0x2000, // Not generatably Resource (FIFO, Client, Timeout)
    ERR_PARMTYP = 0x4000, // Parameter not permitted
    ERR_PARMVAL = 0x8000, // Invalid Parameter value
    ERR_ANYBUSERR = ERR_BUSLIGHT | ERR_BUSHEAVY | ERR_BUSOFF, // All others error status <> 0 please ask by PEAK ......intern Driver errors.....
    ERR_NO_DLL = 0xFFFFFFFF // A Dll could not be loaded or a function was not found into the Dll
};

// Function pointers to load PCANLight functions
//
typedef DWORD(__stdcall *InitPAP)(WORD, int); // Init Plug And Play
typedef DWORD(__stdcall *InitNPAP)(WORD, int, int, DWORD, WORD); // Init Non Plug And Play
typedef DWORD(__stdcall *C_S_RC_RF)(); // Close,Status,ResetClient,ResetFilter
typedef DWORD(__stdcall *ReadWrite)(TPCANMsg *); // CAN_Read CAN_Write
typedef DWORD(__stdcall *USBSDevNr)(long); // SetUSBDeviceNumber
typedef DWORD(__stdcall *USBGDevNr)(long *); // GetUSBDeviceNumber
typedef DWORD(__stdcall *MessageFilter)(DWORD, DWORD, int); // Set MsgFilter
typedef DWORD(__stdcall *GetInfo)(LPSTR); // GetDllVersion, VersionInfo
typedef DWORD(__stdcall *spcFunct)(unsigned long, int); // SpecialFunktion

//---------------------------------------------------------------------------

class CANLight
{
private:
    static HINSTANCE GetDllHandle(HardwareType HWType);

public:
    static bool InitLibraries();
    //static DWORD HexTextToInt(LPCSTR ToConvert);
    static CANResult Init(HardwareType HWType, Baudrates BTR0BTR1, FramesType MsgType);
    static CANResult Init(HardwareType HWType, Baudrates BTR0BTR1, FramesType MsgType, DWORD IO_Port, WORD Interupt);
    static CANResult Close(HardwareType HWType);
    static CANResult Status(HardwareType HWType);
    static CANResult Write(HardwareType HWType, TPCANMsg *MsgToSend);
    static CANResult Read(HardwareType HWType, TPCANMsg *MsgToSend);
    static CANResult VersionInfo(HardwareType HWType, LPCSTR &strInfo);
    static CANResult SpecialFunktion(HardwareType HWType, unsigned long DistributorCode, int CodeNumber);
    static CANResult GetDLL2_Version(LPCSTR &Version);
    static CANResult ResetClient(HardwareType HWType);
    static CANResult MsgFilter(HardwareType HWType, DWORD From, DWORD To, FramesType MsgType);
    static CANResult ResetFilter(HardwareType HWType);
    static CANResult SetUSBDeviceNr(long DeviceNumber);
    static CANResult GetUSBDeviceNr(long *DeviceNumber);
};
#endif