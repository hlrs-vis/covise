/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Porsche.h"

bool Porsche::readBlock()
{
    unsigned char header1 = 0, header2 = 0;
    bufferempty = false;
    int numtries = 0;
    int ret;
    while ((ret = Porsche::readChars(&header2, 1)) != 1 || header1 != 5 || header2 != 205)
    {

        if (ret > 0)
        {
            header1 = header2;
        }
        else
        {
            bufferempty = true;
        }
        numtries++;
        if (numtries > 2)
            std::cerr << "sync" << std::endl;
        if (numtries > 1000)
        {
            std::cerr << "abort1" << std::endl;
            return false;
        }
    }
    int numRead = 0;
    numtries = 0;
    while (numRead < 1)
    {
        numRead = Porsche::readChars(&numFloats, 1);

        if (numRead == 0)
        {
            bufferempty = true;
        }
        numtries++;
        if (numtries > 1000)
        {
            std::cerr << "abort2" << std::endl;
            return false;
        }
    }
    numRead = 0;
    if (numFloats > MAXFLOATS)
    {
        std::cerr << "numFloats wrong " << numFloats << std::endl;
        return false;
    }
    int toRead = numFloats * 4;
    numtries = 0;
    while (toRead > 0)
    {
        ret = Porsche::readChars(((unsigned char *)floats) + numRead, toRead);

        if (ret != toRead)
        {
            bufferempty = true;
        }
        numRead += ret;
        toRead -= ret;
        numtries++;
        if (numtries > 1000)
        {
            std::cerr << "abort2" << std::endl;
            return false;
        }
    }
    unsigned char footer = 1;
    numRead = 0;
    numtries = 0;
    while (numRead < 1)
    {
        numRead = Porsche::readChars(&footer, 1);
        if (numRead == 0)
        {
            bufferempty = true;
        }
        numtries++;
        if (numtries > 1000)
            return false;
    }
    if (footer == 128)
    {
        return true;
    }
    else
    {
        std::cerr << "footer: " << (int)footer << std::endl;
    }
    return false;
}
Porsche::Porsche(const char *devName, int baudrate)
{
    isOpen = false;
    numFloats = 0;
    bufferempty = false;
#ifdef WIN32
    DCB dcb;
    BOOL fSuccess;

    serialDev = CreateFile(devName,
                           GENERIC_READ | GENERIC_WRITE,
                           0, // must be opened with exclusive-access
                           NULL, // no security attributes
                           OPEN_EXISTING, // must use OPEN_EXISTING
                           0, // not overlapped I/O
                           NULL // hTemplate must be NULL for comm devices
                           );

    if (serialDev == INVALID_HANDLE_VALUE)
    {
        // Handle the error.
        printf("could not open com port %s with error %d.\n", devName, GetLastError());
        return;
    }

    // Build on the current configuration, and skip setting the size
    // of the input and output buffers with SetupComm.

    fSuccess = GetCommState(serialDev, &dcb);

    if (!fSuccess)
    {
        // Handle the error.
        printf("GetCommState failed with error %d.\n", GetLastError());
        return;
    }

    /* First, get the parameters which can be set in the Preferences */
    switch (baudrate)
    {
    case 1200:
        dcb.BaudRate = CBR_1200;
        break;
    case 2400:
        dcb.BaudRate = CBR_2400;
        break;
    case 4800:
        dcb.BaudRate = CBR_4800;
        break;
    case 9600:
        dcb.BaudRate = CBR_9600;
        break;
    case 19200:
        dcb.BaudRate = CBR_19200;
        break;
    case 38400:
        dcb.BaudRate = CBR_38400;
        break;
    case 57600:
        dcb.BaudRate = CBR_57600;
        break;
    case 115200:
        dcb.BaudRate = CBR_115200;
        break;
    default:
        dcb.BaudRate = CBR_19200;
        break;
    }
    // Fill in rest of DCB:  8 data bits, no parity, and 1 stop bit.
    dcb.ByteSize = 8; // data size, xmit, and rcv
    dcb.Parity = NOPARITY; // no parity bit
    dcb.StopBits = ONESTOPBIT; // one stop bit
    dcb.fAbortOnError = FALSE;

    fSuccess = SetCommState(serialDev, &dcb);

    if (!fSuccess)
    {
        // Handle the error.
        printf("SetCommState failed with error %d.\n", GetLastError());
        return;
    }
    isOpen = true;
    return;
#else
    int NoDelayFlag = 0;
    int BaudRate = baudrate;
    int Parity = 'N';
    int DataBits = 8;
    int StopBits = 1;
    //int Flags=0;
    //int SerialChan;
    int SerialSpeed = 0;
    int SerialBits = 0;
#ifdef _OLD_TERMIOS
    struct termio TermPar;
#else
    struct termios TermPar;
#endif

    if (NoDelayFlag == true)
    {
        if ((serialDev = open(devName, O_RDWR | O_NDELAY)) == -1)
            return;
    }
    else
    {
        if ((serialDev = open(devName, O_RDWR)) == -1)
            return;
    }
    /* Set the parameters via ioctl. */

    /* First, get the parameters which can be set in the Preferences */
    switch (BaudRate)
    {
    case 1200:
        SerialSpeed |= B1200;
        break;
    case 2400:
        SerialSpeed |= B2400;
        break;
    case 4800:
        SerialSpeed |= B4800;
        break;
    case 9600:
        SerialSpeed |= B9600;
        break;
    case 19200:
        SerialSpeed |= B19200;
        break;
    case 38400:
        SerialSpeed |= B38400;
        break;
    default:
        break;
    }

    if (DataBits == 7)
        SerialBits = CS7;
    else
        SerialBits = CS8;

    if (ioctl(serialDev, TCGETA, &TermPar) == -1)
        return;

    /* change values and flags in term_par struct */
    TermPar.c_oflag &= ~OPOST; /* no character conversion */
    TermPar.c_cc[VMIN] = 0; /* 0 for any setup purposes,> 0 for streaming  */
    TermPar.c_cc[VTIME] = (char)1; /* 1 normally */

    TermPar.c_lflag = 0;
    TermPar.c_oflag = 0;
    TermPar.c_iflag = 0;
    TermPar.c_iflag &= ~IXOFF;
#ifdef __linux__
    TermPar.c_line = 0;
#else
    TermPar.c_line = LDISC1;
#endif

#ifdef _OLD_TERMIOS
    TermPar.c_cflag = (SerialSpeed | SerialBits | CREAD | CLOCAL);
#else
    TermPar.c_cflag = (SerialBits | CREAD | CLOCAL);
    TermPar.c_ispeed = SerialSpeed;
    TermPar.c_ospeed = SerialSpeed;
#endif

    switch (StopBits)
    {
    case 1:
        TermPar.c_cflag &= ~CSTOPB;
        break;
    default:
        TermPar.c_cflag |= CSTOPB;
        break; /* Default: set CSTOPB */
    };

    switch (Parity)
    {
    case 'N':
        TermPar.c_cflag &= ~PARENB;
        break;
    case 'O':
        TermPar.c_cflag |= (PARENB | PARODD);
        break;
    case 'E':
    {
        TermPar.c_cflag |= PARENB;
        TermPar.c_cflag &= ~PARODD;
    };
    };

    /* Put back values */
    if ((ioctl(serialDev, TCSETAF, &TermPar)) == -1)
        return;
    isOpen = true;
#endif
}

Porsche::~Porsche()
{
    if (isOpen)
    {
        isOpen = false;
#ifdef WIN32
        CloseHandle(serialDev);
#else
        if (serialDev != -1)
        {
            if (ioctl(serialDev, TCSETAF, &oldconfig) == -1)
                fprintf(stderr, "\n Fehler beim Wiederherstellen der alten Konfig. !\n");
        }
        if (close(serialDev) == -1)
        {
            serialDev = -1;
        }
        else
        {
            serialDev = -1;
        }
#endif
    }
}

int Porsche::readChars(unsigned char *buf, int n)
{
#ifdef WIN32
    BOOL bResult;
    DWORD nBytesRead = 0;
    bResult = ReadFile(serialDev, (char *)buf, n, &nBytesRead, NULL);

    if (!bResult)
    {
        // Handle the error.
        printf("ReadFile failed with error %d.\n", GetLastError());
        return 0;
    }
    return nBytesRead;
#else
    return read(serialDev, buf, n);
#endif
}
int Porsche::writeChars(const unsigned char *buf, int n)
{
#ifdef WIN32
    DWORD nBytesRead = 0;
    BOOL bResult;
    bResult = WriteFile(serialDev, buf, n, &nBytesRead, NULL);
    return nBytesRead;
#else
    return write(serialDev, buf, n);
#endif
}
