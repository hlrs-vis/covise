/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <util/common.h>

#include "serialio.h"
#include "serialcom.h"
#ifdef _WIN32

HANDLE hCom;
/*************************************************************************/
/**			Oeffentliche Funktionen				**/
/*************************************************************************/

/** Initialisiert die serielle Schnittstelle so, dass eine 	**/
/** Zeichenweise E/A stattfindet ( raw Modus )		   	**/
/**  Datenbits : 8						**/
/**  Baudrate  : baud_rate					**/
/**  Stopbits  : 1						**/
/**  Startbits : 1	  ( wird automatisch gesendet )		**/
/**  Paritaet  : keine						**/
/**  Handshake : HW  						**/

bool Init(char *device, int baud_rate)
{

    /* Set the parameters via ioctl. */

    DCB dcb;
    BOOL fSuccess;
    //device ==  "COM2";

    hCom = CreateFile(device,
                      GENERIC_READ | GENERIC_WRITE,
                      0, // must be opened with exclusive-access
                      NULL, // no security attributes
                      OPEN_EXISTING, // must use OPEN_EXISTING
                      0, // not overlapped I/O
                      NULL // hTemplate must be NULL for comm devices
                      );

    if (hCom == INVALID_HANDLE_VALUE)
    {
        // Handle the error.
        printf("could not open com port %s with error %d.\n", device, GetLastError());
        return (false);
    }

    // Build on the current configuration, and skip setting the size
    // of the input and output buffers with SetupComm.

    fSuccess = GetCommState(hCom, &dcb);

    if (!fSuccess)
    {
        // Handle the error.
        printf("GetCommState failed with error %d.\n", GetLastError());
        return (false);
    }

    /* First, get the parameters which can be set in the Preferences */
    switch (baud_rate)
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

    default:
        dcb.BaudRate = CBR_19200;
        break;
    }
    // Fill in rest of DCB:  8 data bits, no parity, and 1 stop bit.

    dcb.ByteSize = 8; // data size, xmit, and rcv
    dcb.Parity = NOPARITY; // no parity bit
    dcb.StopBits = ONESTOPBIT; // one stop bit

    fSuccess = SetCommState(hCom, &dcb);

    if (!fSuccess)
    {
        // Handle the error.
        printf("SetCommState failed with error %d.\n", GetLastError());
        return (false);
    }

    return (true);
}

bool close_port()
{
    return (CloseHandle(hCom) != 0);
}

/** Wartet die Antwort ab ( max max_time s) und wertet sie aus **/

bool get_answer(unsigned n, unsigned char *out)
{
    // Attempt a synchronous read operation.
    DWORD nBytesRead = 0;
    BOOL bResult;
    bResult = ReadFile(hCom, out, n, &nBytesRead, NULL);

    return (bResult && (n == nBytesRead));
}

/** Wartet die Antwort ab ( max max_time s) und wertet sie aus **/

bool getDivisionAnswer(unsigned n, unsigned char *out)
{
    unsigned int Bytes_read = 0;
    BOOL bResult;
    DWORD nBytesRead = 0;
    while (Bytes_read < n)
    {

        bResult = ReadFile(hCom, out, n - Bytes_read, &nBytesRead, NULL);
        Bytes_read += nBytesRead;
    }
    return 1;
}

/** Sendet Byte fuer Byte einen Stream (command) auf ein Device **/

bool send_command(char *command, int c_length)
{
    // Attempt a synchronous read operation.
    DWORD nBytesRead = 0;
    BOOL bResult;
    bResult = WriteFile(hCom, command, c_length, &nBytesRead, NULL);

    return (bResult && (c_length == nBytesRead));
}

#else // ! WIN32

#include <sys/types.h> /* open */
#include <sys/stat.h> /* open */
#include <sys/ioctl.h>
#include <fcntl.h> /* open */
#include <termios.h> /* tcsetattr */
#include <limits.h> /* sginap */

#ifndef STANDALONE
//#include "VRLogitechTracker.h"
#endif

#include "serialcom.h"

int fd = -1; /** Filedeskriptor ( id fuer Device ) **/
struct termios oldconfig; /** Zwischenspeicher fuer die alte Konfiguration **/
/** der Schnittstelle.				 **/

/*************************************************************************/
/**			Oeffentliche Funktionen				**/
/*************************************************************************/

/** Initialisiert die serielle Schnittstelle so, dass eine 	**/
/** Zeichenweise E/A stattfindet ( raw Modus )		   	**/
/**  Datenbits : 8						**/
/**  Baudrate  : baud_rate					**/
/**  Stopbits  : 1						**/
/**  Startbits : 1	  ( wird automatisch gesendet )		**/
/**  Paritaet  : keine						**/
/**  Handshake : HW  						**/

bool Init(char *device, int baud_rate)
{
    int NoDelayFlag = 0;
    int BaudRate = baud_rate;
    int Parity = 'N';
    int DataBits = 8;
    int StopBits = 1;
    //int Flags=0;
    int SerialChan;
    int SerialSpeed = 0;
    int SerialBits = 0;
    struct termios TermPar;

    //cerr<<"[CoviseInterfaceSerial::Open] Opening Termio chan on "<<device
    //       <<" with baudrate" <<BaudRate<<"\n";

    if (NoDelayFlag == true)
    {
        if ((SerialChan = open(device, O_RDWR | O_NDELAY)) == -1)
            return (false);
    }
    else
    {
        if ((SerialChan = open(device, O_RDWR)) == -1)
            return (false);
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

    if (ioctl(SerialChan, TCGETA, &TermPar) == -1)
        return false;

    /* change values and flags in term_par struct */
    TermPar.c_oflag &= ~OPOST; /* no character conversion */
    TermPar.c_cc[VMIN] = 0; /* 0 for any setup purposes,> 0 for streaming  */
    TermPar.c_cc[VTIME] = (char)1; /* 1 normally */

    TermPar.c_lflag = 0;
    TermPar.c_oflag = 0;
    TermPar.c_iflag = 0;
    TermPar.c_iflag &= ~IXOFF;
#ifdef __linux__
    TermPar.c_line = N_TTY;
#elif defined(__APPLE__)
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
    if ((ioctl(SerialChan, TCSETAF, &TermPar)) == -1)
        return false;
    fd = SerialChan;
    return true;
}

/** Interne Funktion um die alte Konfiguration der Schnittstelle **/
/** wiederherzustellen.						 **/

void restore_config()
{
    if (ioctl(fd, TCSETAF, &oldconfig) == -1)
        fprintf(stderr, "\n Fehler beim Wiederherstellen der alten Konfig. !\n");
}

bool close_port()
{
    if (fd != -1) /** Device wurde geoeffnet **/
        restore_config();

    if (close(fd) == -1)
    {
        fd = -1;
        return 0;
    }
    else
    {
        fd = -1;
        return 1;
    }
}

/** Wartet die Antwort ab ( max max_time s) und wertet sie aus **/

bool get_answer(unsigned n, unsigned char *out)
{
    time_t start_time = time(NULL);
    int Bytes_read = 0;
    int max_time = 1;
    int ret;

    while (Bytes_read < n)
    {
        //fprintf(stderr,"Reading %d\n",fd);
        ret = read(fd, &out[Bytes_read], n - Bytes_read);
        Bytes_read += ret;
        //fprintf(stderr,"ret: %d, Bytes_read: %d\n",ret,Bytes_read);
        if (time(NULL) > start_time + max_time)
        {
            // fprintf(stderr, "\n Timeout : Keine Antwort erhalten !\n");
            return 0;
        }
    }
    return 1;

    /*  while ( 1 )
     {
       for ( Bytes_read = 0; Bytes_read < 25; Bytes_read++ )
       {
         int i;
         read(fd, &out[0], 1);
         printf( "%2x ", out[0] );
       }
       printf( "\n" );
     } */
}

/** Wartet die Antwort ab ( max max_time s) und wertet sie aus **/

bool getDivisionAnswer(unsigned n, unsigned char *out)
{
    time_t start_time = time(NULL);
    int Bytes_read = 0;
    int max_time = 1;

    while (Bytes_read < n)
    {
        Bytes_read += read(fd, &out[Bytes_read], n - Bytes_read);
        if (time(NULL) > start_time + max_time)
        {
            //fprintf(stderr, "\n Timeout : Keine Antwort erhalten !\n");
            return 0;
        }
    }
    return 1;

    /*  while ( 1 )
     {
       for ( Bytes_read = 0; Bytes_read < 25; Bytes_read++ )
       {
         int i;
         read(fd, &out[0], 1);
         printf( "%2x ", out[0] );
       }
       printf( "\n" );
     } */
}

/** Sendet Byte fuer Byte einen Stream (command) auf ein Device **/

bool send_command(char *command, int c_length)
{
    int Bytes_send = 0,
        n;

    if (c_length > 0)
    {
        while (Bytes_send < c_length)
        {
            if ((n = write(fd, command, 1)) == -1)
            {
                fprintf(stderr, "\n Fehler beim Schreiben !\n");
                return 0;
            }
            else
            {
                command += n;
                Bytes_send += n;
            }
        }
        return 1;
    }
    return 1;
}
#endif
