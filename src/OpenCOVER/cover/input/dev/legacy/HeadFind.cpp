/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//---------------------------------------------------------------------------
//
//  Module: HEFD_COM.c
//
//  Purpose:
//     The application sends and reads DATA from and to the Eyefinder
//     via the serial interface
//     API.  It implements the COMM API of Windows 3.1.
//
//---------------------------------------------------------------------------

#include <stdio.h>
#include "HeadFind.h"
#include <algorithm>

// data structures
char HeadFinderInfo[255] = "";
//
HEFDCOMINFO headfinderInfo;

char RxData[MAXBLOCK + 1] = { 0 };

#define fifoDepth 5

int headX, headY, headZ;
int fifoX[fifoDepth], fifoY[fifoDepth], fifoZ[fifoDepth];
int fifoPos;

int useHeadFinder, isHeadFinder;
BYTE HeadFinderPort = 2; // variable: 1=COM1  2=COM2
DWORD BaudRate = CBR_115200; // variable: C-i(eye-tracked) = CBR_115200   C-s(spot-tracked) = CBR_19200

// global stuff

HWND hHEFDCOMWnd;
char gszHEFDCOMClass[] = "HEFDCOMWndClass";
char gszAppName[] = "HeadFinder";
HANDLE ghAccel;

int sendit = 0;
int receiveBild = 0, receiveBild4 = 0, receiveSoll;
int searchCount;
int continues = 0;

DWORD BaudTable[] = {
    CBR_110, CBR_300, CBR_600, CBR_1200, CBR_2400,
    CBR_4800, CBR_9600, CBR_14400, CBR_19200, CBR_38400,
    CBR_56000, CBR_115200, CBR_256000 // CBR_128000
};

DWORD ParityTable[] = {
    NOPARITY, EVENPARITY, ODDPARITY, MARKPARITY, SPACEPARITY
};

DWORD StopBitsTable[] = {
    ONESTOPBIT, ONE5STOPBITS, TWOSTOPBITS
};

/*
 * initHeadFinder
 */
int initHeadFinder(void *hWnd)
{
    SetHeadFinderPort(HeadFinderPort);
    SetBaudRate(BaudRate);
    //
    hHEFDCOMWnd = (HWND)hWnd;
    useHeadFinder = FALSE;
    isHeadFinder = FALSE;
    fifoPos = 0;
    //
    CreateHEFDCOMInfo((HWND)hWnd);
    if (!OpenConnection((HWND)hWnd))
        return (0);
    //
    isHeadFinder = TRUE;
    WriteCommBlock((HWND)hWnd, "0", 1);
    //WriteCommBlock((HWND)hWnd,"Q",1);
    useHF(TRUE);
    return (1);
}

/*
 * closeHeadFinder
 */
int closeHeadFinder(void *hWnd)
{
    CloseConnection((HWND)hWnd);
    DestroyHEFDCOMInfo((HWND)hWnd);

    useHeadFinder = FALSE;
    isHeadFinder = FALSE;
    return TRUE;
}

/*
 * SetHeadFinderPort
 */
void SetHeadFinderPort(BYTE Port)
{
    Port = Port == 1 || Port == 2 ? Port : 1;
    HeadFinderPort = Port;
}

/*
 * SetBaudRate
 */
void SetBaudRate(DWORD ABaudRate)
{
    ABaudRate = ABaudRate == 115200 || ABaudRate == 19200 ? ABaudRate : 115200;
    BaudRate = ABaudRate;
}

/*
 * NextHeadFinderSetting
 */
void NextHeadFinderSetting()
{
    struct tagSetting
    {
        BYTE HeadFinderPort;
        DWORD BaudRate;
    } Setting[4] = { 1, 19200, 2, 19200, 1, 115200, 2, 115200 };
    //
    int CurrentSetting;
    //
    for (int i = 0; i < 4; i++)
    {
        if (Setting[i].HeadFinderPort == HeadFinderPort && Setting[i].BaudRate == BaudRate)
        {
            CurrentSetting = i;
            break;
        }
    }
    //
    CurrentSetting++;
    CurrentSetting = CurrentSetting % 4;
    //
    SetHeadFinderPort(Setting[CurrentSetting].HeadFinderPort);
    SetBaudRate(Setting[CurrentSetting].BaudRate);
}

//---------------------------------------------------------------------------
//  LRESULT NEAR CreateHEFDCOMInfo( HWND hWnd )
//
//  Description:
//     Creates the HEFDCOM information structure and sets
//     menu option availability.  Returns -1 if unsuccessful.
//
//  Parameters:
//     HWND  hWnd
//        Handle to main window.
//
//  Win-32 Porting Issues:
//     - Needed to initialize TERMWND( headfinderInfo ) for secondary thread.
//     - Needed to create/initialize overlapped structures used in reads &
//       writes to COMM device.
//
//---------------------------------------------------------------------------

LRESULT NEAR CreateHEFDCOMInfo(HWND hWnd)
{
    // initialize HEFDCOM info structure

    headfinderInfo.idComDev = 0;
    headfinderInfo.fConnected = FALSE;
    headfinderInfo.fLocalEcho = FALSE;
    headfinderInfo.fAutoWrap = TRUE;
    headfinderInfo.bPort = HeadFinderPort;
    headfinderInfo.dwBaudRate = BaudRate;
    headfinderInfo.bByteSize = 8;
    headfinderInfo.bFlowCtrl = 0;
    headfinderInfo.bParity = NOPARITY;
    headfinderInfo.bStopBits = ONESTOPBIT;
    headfinderInfo.fXonXoff = FALSE;
    headfinderInfo.rgbFGColor = RGB(0, 0, 0);
    headfinderInfo.fUseCNReceive = TRUE;
    headfinderInfo.fDisplayErrors = TRUE;
    headfinderInfo.osWrite.Offset = 0;
    headfinderInfo.osWrite.OffsetHigh = 0;
    headfinderInfo.osRead.Offset = 0;
    headfinderInfo.osRead.OffsetHigh = 0;
    headfinderInfo.hTermWnd = hWnd;

    // create I/O event used for overlapped reads / writes

    headfinderInfo.osRead.hEvent = CreateEvent(NULL, // no security
                                               TRUE, // explicit reset req
                                               FALSE, // initial event reset
                                               NULL // no name
                                               );
    if (headfinderInfo.osRead.hEvent == NULL)
    {
        return (-1);
    }

    headfinderInfo.osWrite.hEvent = CreateEvent(NULL, // no security
                                                TRUE, // explicit reset req
                                                FALSE, // initial event reset
                                                NULL // no name
                                                );

    if (NULL == headfinderInfo.osWrite.hEvent)
    {
        CloseHandle(headfinderInfo.osRead.hEvent);
        return (-1);
    }

    return TRUE;
} // end of CreateHEFDCOMInfo()

//---------------------------------------------------------------------------
//  BOOL NEAR DestroyHEFDCOMInfo( HWND hWnd )
//
//  Description:
//     Destroys block associated with HEFDCOM window handle.
//
//  Parameters:
//     HWND hWnd
//        handle to HEFDCOM window
//
//  Win-32 Porting Issues:
//     - Needed to clean up event objects created during initialization.
//
//---------------------------------------------------------------------------

BOOL NEAR DestroyHEFDCOMInfo(HWND hWnd)
{
    // force connection closed (if not already closed)
    if (headfinderInfo.fConnected)
        CloseConnection(hWnd);

    // clean up event objects
    //VH	CloseHandle( headfinderInfo.osRead.hEvent );
    //VH	CloseHandle( headfinderInfo.osWrite.hEvent );
    //VH	CloseHandle( headfinderInfo.hPostEvent );

    return TRUE;
} // end of DestroyHEFDCOMInfo()

//---------------------------------------------------------------------------
//  BOOL NEAR OpenConnection( HWND hWnd )
//
//  Description:
//     Opens communication port specified in the HEFDCOMINFO struct.
//     It also sets the CommState and notifies the window via
//     the fConnected flag in the HEFDCOMINFO struct.
//
//  Parameters:
//     HWND hWnd
//        handle to HEFDCOM window
//
//  Win-32 Porting Issues:
//     - OpenComm() is not supported under Win-32.  Use CreateFile()
//       and setup for OVERLAPPED_IO.
//     - Win-32 has specific communication timeout parameters.
//     - Created the secondary thread for event notification.
//
//---------------------------------------------------------------------------

BOOL NEAR OpenConnection(HWND hWnd)
{
    char szPort[15], szTemp[10];
    BOOL fRetVal;

    HANDLE hCommWatchThread;
    DWORD dwThreadID;
    COMMTIMEOUTS CommTimeOuts;

    // load the COM prefix string and append port number

    strcpy(szTemp, "COM");
    wsprintf(szPort, "%s%d", (LPSTR)szTemp, headfinderInfo.bPort);

    // open COMM device

    if ((headfinderInfo.idComDev = CreateFile(szPort, GENERIC_READ | GENERIC_WRITE,
                                              0, // exclusive access
                                              NULL, // no security attrs
                                              OPEN_EXISTING,
                                              FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, // overlapped I/O
                                              NULL)) == (HANDLE)-1)
        return (FALSE);
    else
    {
        // get any early notifications

        SetCommMask(headfinderInfo.idComDev, EV_RXCHAR);

        // setup device buffers

        SetupComm(headfinderInfo.idComDev, 4096, 4096);

        // purge any information in the buffer

        PurgeComm(headfinderInfo.idComDev, PURGE_TXABORT | PURGE_RXABORT | PURGE_TXCLEAR | PURGE_RXCLEAR);

        // set up for overlapped I/O

        CommTimeOuts.ReadIntervalTimeout = 0xFFFFFFFF;
        CommTimeOuts.ReadTotalTimeoutMultiplier = 0;
        CommTimeOuts.ReadTotalTimeoutConstant = 1000;
        // CBR_9600 is approximately 1byte/ms. For our purposes, allow
        // double the expected time per character for a fudge factor.
        CommTimeOuts.WriteTotalTimeoutMultiplier = 2 * CBR_9600 / headfinderInfo.dwBaudRate;
        CommTimeOuts.WriteTotalTimeoutConstant = 0;
        SetCommTimeouts(headfinderInfo.idComDev, &CommTimeOuts);
    }

    fRetVal = SetupConnection(hWnd);

    if (fRetVal)
    {
        headfinderInfo.fConnected = TRUE;

        // Create a secondary thread
        // to watch for an event.

        if (NULL == (hCommWatchThread = CreateThread((LPSECURITY_ATTRIBUTES)NULL,
                                                     0,
                                                     (LPTHREAD_START_ROUTINE)CommWatchProc,
                                                     (LPVOID)&headfinderInfo,
                                                     0, &dwThreadID)))
        {
            headfinderInfo.fConnected = FALSE;
            CloseHandle(headfinderInfo.idComDev);
            fRetVal = FALSE;
        }
        else
        {
            headfinderInfo.dwThreadID = dwThreadID;
            headfinderInfo.hWatchThread = hCommWatchThread;

            // assert DTR
            EscapeCommFunction(headfinderInfo.idComDev, SETDTR);
        }
    }
    else
    {
        headfinderInfo.fConnected = FALSE;
        CloseHandle(headfinderInfo.idComDev);
    }

    return (fRetVal);

} // end of OpenConnection()

//---------------------------------------------------------------------------
//  BOOL NEAR SetupConnection( HWND hWnd )
//
//  Description:
//     This routines sets up the DCB based on settings in the
//     HEFDCOM info structure and performs a SetCommState().
//
//  Parameters:
//     HWND hWnd
//        handle to HEFDCOM window
//
//  Win-32 Porting Issues:
//     - Win-32 requires a slightly different processing of the DCB.
//       Changes were made for configuration of the hardware handshaking
//       lines.
//
//---------------------------------------------------------------------------

BOOL NEAR SetupConnection(HWND hWnd)
{
    BOOL fRetVal;
    BYTE bSet;
    DCB dcb;

    dcb.DCBlength = sizeof(DCB);

    GetCommState(headfinderInfo.idComDev, &dcb);

    dcb.BaudRate = headfinderInfo.dwBaudRate;
    dcb.ByteSize = headfinderInfo.bByteSize;
    dcb.Parity = headfinderInfo.bParity;
    dcb.StopBits = headfinderInfo.bStopBits;

    // setup hardware flow control
    bSet = (BYTE)((headfinderInfo.bFlowCtrl & FC_DTRDSR) != 0);
    dcb.fOutxDsrFlow = bSet;
    if (bSet)
        dcb.fDtrControl = DTR_CONTROL_HANDSHAKE;
    else
        dcb.fDtrControl = DTR_CONTROL_ENABLE;

    bSet = (BYTE)((headfinderInfo.bFlowCtrl & FC_RTSCTS) != 0);
    dcb.fOutxCtsFlow = bSet;
    if (bSet)
        dcb.fRtsControl = RTS_CONTROL_HANDSHAKE;
    else
        dcb.fRtsControl = RTS_CONTROL_ENABLE;

    // setup software flow control
    bSet = (BYTE)((headfinderInfo.bFlowCtrl & FC_XONXOFF) != 0);

    dcb.fInX = dcb.fOutX = bSet;
    dcb.XonChar = ASCII_XON;
    dcb.XoffChar = ASCII_XOFF;
    dcb.XonLim = 100;
    dcb.XoffLim = 100;

    // other various settings

    dcb.fBinary = TRUE;
    dcb.fParity = TRUE;

    fRetVal = SetCommState(headfinderInfo.idComDev, &dcb);

    return (fRetVal);

} // end of SetupConnection()

//---------------------------------------------------------------------------
//  BOOL NEAR CloseConnection( HWND hWnd )
//
//  Description:
//     Closes the connection to the port.  Resets the connect flag
//     in the HEFDCOMINFO struct.
//
//  Parameters:
//     HWND hWnd
//        handle to HEFDCOM window
//
//  Win-32 Porting Issues:
//     - Needed to stop secondary thread.  SetCommMask() will signal the
//       WaitCommEvent() event and the thread will halt when the
//       CONNECTED() flag is clear.
//     - Use new PurgeComm() API to clear communications driver before
//       closing device.
//
//---------------------------------------------------------------------------

BOOL NEAR CloseConnection(HWND hWnd)
{
    // set connected flag to FALSE

    headfinderInfo.fConnected = FALSE;

    // disable event notification and wait for thread to halt
    SetCommMask(headfinderInfo.idComDev, 0);

    // block until thread has been halted

    //HH while(THREADID(npHEFDCOMInfo) != 0);
    CloseHandle(headfinderInfo.hWatchThread);
    headfinderInfo.hWatchThread = 0;

    // drop DTR
    EscapeCommFunction(headfinderInfo.idComDev, CLRDTR);

    // purge any outstanding reads/writes and close device handle
    PurgeComm(headfinderInfo.idComDev, PURGE_TXABORT | PURGE_RXABORT | PURGE_TXCLEAR | PURGE_RXCLEAR);
    //VH
    CloseHandle(headfinderInfo.idComDev); //SB

    // change the selectable items in the menu
    return TRUE;
} // end of CloseConnection()

//---------------------------------------------------------------------------
//  int NEAR ReadCommBlock( HWND hWnd, LPSTR lpszBlock, int nMaxLength )
//
//  Description:
//     Reads a block from the COM port and stuffs it into
//     the provided buffer.
//
//  Parameters:
//     HWND hWnd
//        handle to HEFDCOM window
//
//     LPSTR lpszBlock
//        block used for storage
//
//     int nMaxLength
//        max length of block to read
//
//  Win-32 Porting Issues:
//     - ReadComm() has been replaced by ReadFile() in Win-32.
//     - Overlapped I/O has been implemented.
//
//---------------------------------------------------------------------------

int NEAR ReadCommBlock(HWND hWnd, LPSTR lpszBlock, int nMaxLength)
{
    BOOL fReadStat;
    COMSTAT ComStat;
    DWORD dwErrorFlags;
    DWORD dwLength;
    DWORD dwError;
    char szError[10];

    // only try to read number of bytes in queue
    ClearCommError(headfinderInfo.idComDev, &dwErrorFlags, &ComStat);
    dwLength = std::min((DWORD)nMaxLength, ComStat.cbInQue);

    if (dwLength > 0)
    {
        fReadStat = ReadFile(headfinderInfo.idComDev, lpszBlock,
                             dwLength, &dwLength, &headfinderInfo.osRead);
        if (!fReadStat)
        {
            if (GetLastError() == ERROR_IO_PENDING)
            {
                OutputDebugString("\n\rIO Pending");
                // We have to wait for read to complete.
                // This function will timeout according to the
                // CommTimeOuts.ReadTotalTimeoutConstant variable
                // Every time it times out, check for port errors
                while (!GetOverlappedResult(headfinderInfo.idComDev,
                                            &headfinderInfo.osRead, &dwLength, TRUE))
                {
                    dwError = GetLastError();
                    if (dwError == ERROR_IO_INCOMPLETE)
                        // normal result if not finished
                        continue;
                    else
                    {
                        // an error occurred, try to recover
                        wsprintf(szError, "<CE-%u>", dwError);
                        ClearCommError(headfinderInfo.idComDev, &dwErrorFlags, &ComStat);
                        if ((dwErrorFlags > 0) && headfinderInfo.fDisplayErrors)
                        {
                            wsprintf(szError, "<CE-%u>", dwErrorFlags);
                        }
                        break;
                    }
                }
            }
            else
            {
                // some other error occurred
                dwLength = 0;
                ClearCommError(headfinderInfo.idComDev, &dwErrorFlags, &ComStat);
                if ((dwErrorFlags > 0) && headfinderInfo.fDisplayErrors)
                {
                    wsprintf(szError, "<CE-%u>", dwErrorFlags);
                }
            }
        }
    }

    return (dwLength);

} // end of ReadCommBlock()

//---------------------------------------------------------------------------
//  BOOL NEAR WriteCommBlock( HWND hWnd, BYTE *pByte )
//
//  Description:
//     Writes a block of data to the COM port specified in the associated
//     HEFDCOM info structure.
//
//  Parameters:
//     HWND hWnd
//        handle to HEFDCOM window
//
//     BYTE *pByte
//        pointer to data to write to port
//
//  Win-32 Porting Issues:
//     - WriteComm() has been replaced by WriteFile() in Win-32.
//     - Overlapped I/O has been implemented.
//
//---------------------------------------------------------------------------

BOOL NEAR WriteCommBlock(HWND hWnd, LPSTR lpByte, DWORD dwBytesToWrite)
{

    BOOL fWriteStat;
    DWORD dwBytesWritten;
    DWORD dwErrorFlags;
    DWORD dwError;
    DWORD dwBytesSent = 0;
    COMSTAT ComStat;
    char szError[128];

    fWriteStat = WriteFile(headfinderInfo.idComDev, lpByte, dwBytesToWrite,
                           &dwBytesWritten, &headfinderInfo.osWrite);

    // Note that normally the code will not execute the following
    // because the driver caches write operations. Small I/O requests
    // (up to several thousand bytes) will normally be accepted
    // immediately and WriteFile will return true even though an
    // overlapped operation was specified

    if (!fWriteStat)
    {
        if (GetLastError() == ERROR_IO_PENDING)
        {
            // We should wait for the completion of the write operation
            // so we know if it worked or not

            // This is only one way to do this. It might be beneficial to
            // place the write operation in a separate thread
            // so that blocking on completion will not negatively
            // affect the responsiveness of the UI

            // If the write takes too long to complete, this
            // function will timeout according to the
            // CommTimeOuts.WriteTotalTimeoutMultiplier variable.
            // This code logs the timeout but does not retry
            // the write.

            while (!GetOverlappedResult(headfinderInfo.idComDev,
                                        &headfinderInfo.osWrite, &dwBytesWritten, TRUE))
            {
                dwError = GetLastError();
                if (dwError == ERROR_IO_INCOMPLETE)
                {
                    // normal result if not finished
                    dwBytesSent += dwBytesWritten;
                    continue;
                }
                else
                {
                    // an error occurred, try to recover
                    wsprintf(szError, "<CE-%u>", dwError);
                    //               WriteHEFDCOMBlock( hWnd, szError, lstrlen( szError ) ) ;
                    ClearCommError(headfinderInfo.idComDev, &dwErrorFlags, &ComStat);
                    if ((dwErrorFlags > 0) && headfinderInfo.fDisplayErrors)
                    {
                        wsprintf(szError, "<CE-%u>", dwErrorFlags);
                        //                  WriteHEFDCOMBlock( hWnd, szError, lstrlen( szError ) ) ;
                    }
                    break;
                }
            }

            dwBytesSent += dwBytesWritten;

            if (dwBytesSent != dwBytesToWrite)
                wsprintf(szError, "\nProbable Write Timeout: Total of %ld bytes sent", dwBytesSent);
            else
                wsprintf(szError, "\n%ld bytes written", dwBytesSent);

            OutputDebugString(szError);
        }
        else
        {
            // some other error occurred
            ClearCommError(headfinderInfo.idComDev, &dwErrorFlags, &ComStat);
            if ((dwErrorFlags > 0) && headfinderInfo.fDisplayErrors)
            {
                wsprintf(szError, "<CE-%u>", dwErrorFlags);
            }
            return (FALSE);
        }
    }
    return (TRUE);

} // end of WriteCommBlock()

//---------------------------------------------------------------------------
//  VOID NEAR GoModalDialogBoxParam( HINSTANCE hInstance,
//                                   LPCSTR lpszTemplate, HWND hWnd,
//                                   DLGPROC lpDlgProc, LPARAM lParam )
//
//  Description:
//     It is a simple utility function that simply performs the
//     MPI and invokes the dialog box with a DWORD paramter.
//
//  Parameters:
//     similar to that of DialogBoxParam() with the exception
//     that the lpDlgProc is not a procedure instance
//
//---------------------------------------------------------------------------

VOID NEAR GoModalDialogBoxParam(HINSTANCE hInstance, LPCSTR lpszTemplate,
                                HWND hWnd, DLGPROC lpDlgProc, LPARAM lParam)
{
    DLGPROC lpProcInstance;

    lpProcInstance = (DLGPROC)MakeProcInstance((FARPROC)lpDlgProc,
                                               hInstance);
    DialogBoxParam(hInstance, lpszTemplate, hWnd, lpProcInstance, lParam);
    FreeProcInstance((FARPROC)lpProcInstance);

} // end of GoModalDialogBoxParam()

//---------------------------------------------------------------------------
//  BOOL FAR PASCAL AboutDlgProc( HWND hDlg, UINT uMsg,
//                                WPARAM wParam, LPARAM lParam )
//
//  Description:
//     Simulates the Windows System Dialog Box.
//
//  Parameters:
//     Same as standard dialog procedures.
//
//---------------------------------------------------------------------------

//************************************************************************
//  DWORD FAR PASCAL CommWatchProc( LPSTR lpData )
//
//  Description:
//     A secondary thread that will watch for COMM events.
//
//  Parameters:
//     LPSTR lpData
//        32-bit pointer argument
//
//  Win-32 Porting Issues:
//     - Added this thread to watch the communications device and
//       post notifications to the associated window.
//
//************************************************************************

DWORD FAR PASCAL CommWatchProc(LPSTR lpData)
{
    DWORD dwEvtMask;
    OVERLAPPED os;
    int nLength, len;
    BYTE abIn[MAXBLOCK + 1];
    BYTE recBuf[MAXBLOCK + 1];
    int i;
    static int recBufAnz = 0;
    char ch;
    short int *coord;

    memset(&os, 0, sizeof(OVERLAPPED));

    // create I/O event used for overlapped read

    os.hEvent = CreateEvent(NULL, // no security
                            TRUE, // explicit reset req
                            FALSE, // initial event reset
                            NULL); // no name
    if (os.hEvent == NULL)
    {
        MessageBox(NULL, "Failed to create event for thread!", "HEFDCOM Error!",
                   MB_ICONEXCLAMATION | MB_OK);
        return (FALSE);
    }

    if (!SetCommMask(headfinderInfo.idComDev, EV_RXCHAR))
        return (FALSE);

    while (headfinderInfo.fConnected)
    {
        dwEvtMask = 0;

        WaitCommEvent(headfinderInfo.idComDev, &dwEvtMask, NULL);
        if ((dwEvtMask & EV_RXCHAR) == EV_RXCHAR)
        {
            do
            {
                if (nLength = ReadCommBlock(hHEFDCOMWnd, (LPSTR)abIn, MAXBLOCK))
                {
                    // Prüfe empfangene Daten - reagiere je nach Inhalt
                    memcpy(recBuf + recBufAnz, abIn, nLength);
                    recBufAnz += nLength;
                    len = recBufAnz;
                    i = 0;

                    while (i < len)
                    {
                        while ((i < len) && (recBuf[i] != 0x80))
                            i++;
                        if (i == len)
                        {
                            recBufAnz = 0;
                            break;
                        }
                        if (i < len - 1)
                        {
                            if ((recBuf[i + 1]) == 0x83)
                            {
                                if ((len - i) > 6)
                                { // noch genügend Zeichen im Puffer -> auswerten
                                    i += 2;
                                    // i steht jetzt auf x
                                    // Byte vertauschen (big endian)
                                    ch = recBuf[i];
                                    recBuf[i] = recBuf[i + 1];
                                    recBuf[i + 1] = ch;
                                    coord = (short int *)(recBuf + i);
                                    headX = *coord;
                                    i += 2;
                                    ch = recBuf[i];
                                    recBuf[i] = recBuf[i + 1];
                                    recBuf[i + 1] = ch;
                                    coord = (short int *)(recBuf + i);
                                    headY = *coord;
                                    i += 2;
                                    ch = recBuf[i];
                                    recBuf[i] = recBuf[i + 1];
                                    recBuf[i + 1] = ch;
                                    coord = (short int *)(recBuf + i);
                                    headZ = *coord;
                                    fifoX[fifoPos] = headX;
                                    fifoY[fifoPos] = headY;
                                    fifoZ[fifoPos] = headZ;
                                    fifoPos++; // fifoPos &= fifoDepth;
                                    if (fifoPos == fifoDepth)
                                        fifoPos = 0;
                                    i += 2;
                                    if (i == len)
                                        recBufAnz = 0;
                                    // Koordinaten stehen jetzt in x,y und z zur Verfügung
                                    // im Beispiel werden Sie nur angezeigt:
                                    //char WndTxtBuf[80]="";
                                    //                      sprintf(WndTxtBuf,"HeadFinder - Augenposition | x: %d  y: %d  z: %d",x,y,z);
                                    //                    SetWindowText(hHEFDCOMWnd,WndTxtBuf);
                                }
                                else //if ((len-i)>6)
                                { //Länge reicht nicht - empfangene Daten abspeichern; aufheben
                                    memcpy(recBuf, recBuf + i, len - i);
                                    recBufAnz = len - i;
                                    recBuf[len - i] = 0;
                                    break;
                                }
                            }
                            else // if (recBuf[i+1]) == 0x83)
                            {
                                i++;
                            }
                        }
                        else // if (i<len-1)
                        { // i==len-1
                            recBuf[0] = 0x80;
                            recBufAnz = 1;
                            break;
                        }
                    } // while (i<len)

                    // force a paint
                }
            } while (nLength > 0);
        }
    } // while ( CONNECTED( npHEFDCOMInfo ) )

    // get rid of event handle

    CloseHandle(os.hEvent);

    // clear information in structure (kind of a "we're done flag")
    ExitThread(0); //HH

    return (TRUE);

} // end of CommWatchProc()
/*----------------------------------------------------------------------------------*/

//---------------------------------------------------------------------------
//  End of File: HEFDCOM.c
//---------------------------------------------------------------------------

void useHF(bool value)
{
    useHeadFinder = value;
}

void getViewerPosition(int &x, int &y, int &z)
{
    int i;

    x = 0;
    y = 0;
    z = 0;
    if (isHeadFinder && useHeadFinder)
    {
        for (i = 0; i < fifoDepth; i++)
        {
            x += fifoX[i];
            y += fifoY[i];
            z += fifoZ[i];
        }
        x /= fifoDepth;
        y /= fifoDepth;
        z /= fifoDepth;
    }
}
