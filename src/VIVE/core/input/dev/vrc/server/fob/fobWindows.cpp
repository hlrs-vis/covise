/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *                                                                      *
 *                                                                      *
 *                            (C) 1999                                  *
 *                                                                      *
 *                                                                      *
 *                       D-70469 Stuttgart                              *
 *                            Germany                                   *
 *                                                                      *
 *                                                                      *
 *Descriptionflock of birds low level driver class for windows          *
 *no! extra process for reading the                                     *
 *serial port                                                           *
 *                                                                      *
 *AuthorL. Frenzel, D. Rainer, U. Woessner                              * 
 *                                                                      *
 *DateMai 99                                                            *
 *                                                                      *
 ************************************************************************/
#include <covise/covise.h>
#ifndef WIN32
#include <strings.h>
#include <time.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <termio.h>
#include <termios.h>
#include <sys/types.h>
#include <sys/resource.h>
#include <sys/prctl.h>
#else
#endif
#include <signal.h>
#ifdef __linux__
#define sigset signal
#endif

#include "fob.h"

#define PERMS 0666

#define INCHES_IN_MM 25.4

#undef VERBOSE

fob::fob(const char *portname, int baudrate, int nb, int fmode)
{
#ifdef VERBOSE
    fprintf(stderr, "fob::fob\n");
#endif
    _stopping = false;
    connected = false;
    numReceivers = 0;
    numERCs = 0;
    askedForSystemStatus = 0;
    numBytesPerData = 0; // size of the output
    numBirds = nb;

    serverRunning = false;

    receivers = new birdReceiver[maxNumReceivers];
    terminate = new bool;

    *terminate = false;
    memset(receivers, 0, maxNumReceivers * sizeof(birdReceiver));

    /* Set the parameters via ioctl. */

    DCB dcb;
    BOOL fSuccess;
    //device ==  "COM2";
    COMMTIMEOUTS timeouts;
    timeouts.ReadIntervalTimeout = 300;
    timeouts.ReadTotalTimeoutMultiplier = 300;
    timeouts.ReadTotalTimeoutConstant = 300;
    timeouts.WriteTotalTimeoutMultiplier = 300;
    timeouts.WriteTotalTimeoutConstant = 300;
    serialChannel = CreateFile(portname,
                               GENERIC_READ | GENERIC_WRITE,
                               0, // must be opened with exclusive-access
                               NULL, // no security attributes
                               OPEN_EXISTING, // must use OPEN_EXISTING
                               1, // 0 meansnot overlapped I/O
                               NULL // hTemplate must be NULL for comm devices
                               );
    if (serialChannel == INVALID_HANDLE_VALUE)
    {
        // Handle the error.
        printf("could not open com port %s with error %d.\n", portname, GetLastError());
    }
    if (!SetCommTimeouts(serialChannel, &timeouts))
    {
        fprintf(stderr, "SetCommTimeouts() Failed\n");
    }

    // Build on the current configuration, and skip setting the size
    // of the input and output buffers with SetupComm.

    fSuccess = GetCommState(serialChannel, &dcb);

    if (!fSuccess)
    {
        // Handle the error.
        printf("GetCommState failed with error %d.\n", GetLastError());
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

    default:
        dcb.BaudRate = CBR_19200;
        break;
    }
    // Fill in rest of DCB:  8 data bits, no parity, and 1 stop bit.

    dcb.ByteSize = 8; // data size, xmit, and rcv
    dcb.Parity = NOPARITY; // no parity bit
    dcb.StopBits = ONESTOPBIT; // one stop bit

    fSuccess = SetCommState(serialChannel, &dcb);

    if (!fSuccess)
    {
        // Handle the error.
        printf("SetCommState failed with error %d.\n", GetLastError());
    }

    // we are connected now
    connected = true;
}

fob::~fob()
{
#ifdef VERBOSE
    fprintf(stderr, "fob::~fob\n");
#endif
    CloseHandle(serialChannel);
    serverRunning = false;
    *terminate = true;
}

int
fob::sendSer(unsigned char *bfr, int num)
{
    DWORD numWrite = 0;
    BOOL bResult;
    bResult = WriteFile(serialChannel, bfr, num, &numWrite, NULL);
    if (numWrite != num)
    {
        perror("fob::sendSer write");
        fprintf(stderr, "incomplete write: only %d of %d bytes written\n",
                numWrite, num);
        return (0);
    }
    else
    {
        // we have to wait here because the bird is not the fastest
        //sleep( 1 );
        return (1);
    }
}

int
fob::receiveSer(char *bfr, int num)
{
    DWORD bytesReceived;
    BOOL bResult;
    int bytesToRead;
    long startTime, endTime;
    ////Die serielle Schnittstelle
    ////zum Flock muss geschlossen werden, genau dann, wenn
    ////keine Zeichen mehr auf der seriellen
    ////Schnittstelle liegen.
    ////Daher schliesst der Signalhandler
    ////auch die serielle Schnittstelle nicht,
    ////sondern setzt ein Flag( _stopping ), so dass
    ////die serielle Schnittstelle genau nach dem letzten Lesen
    ////d.h. in dieser Routine geschlossen wird.
    ////Fuer die Windowsversion ist es wichtig zu wissen,
    ////dass es einen wichtigen Unterschied im Lesen der seriellen Schnittstelle
    ////zwischen Unix und Windows gibt:
    ////Unix: Werden mehr Zeichen angefordert, als von der seriellen Schnittstelle
    ////   zur Verfuegung gestellt, so gibt es unter Unix irgendwann einen
    ////   Timeout; auf diese Weise kann leicht festgestellt werden, dass
    ////   keine Zeichen mehr kommen.
    ////Windows: Werden mehr Zeichen angefordert, als von der seriellen Schnittstelle
    ////   zur Verfuegung gestellt, so wartet Windows bis die geforderten Zeichen kommen.
    ////   egal, ob sie ueberhaupt noch kommen. Festzustellen, dass _keine_ Zeichen
    ////   mehr kommen, ist unter Windows nur unter erschwerten Bedingungen moeglich
    ////Dies sicherzustellen ist aber fue den sauberen Betrieb des Flock of bird unerlaesslich.
    ////Genauergesagt, erwartet der Flock of birds, dass bei der letzten Benutzung
    ////alle seine Ausgaben gelesen wurden.
    ////Daher setzt der Signalhandler ein Flag, so dass der Flock an geeigenter Stelle
    ////heruntergefahren weden kann, naemlich nach dem Lesen der letzten Koordinaten.
    ////also am besten hier in dieser Methode.
    ////STREAM STOP  ( Commando 0x3f ) bzw. RESET ( Commando 0x2f ) sind
    ////halten den Flock of Birds im Uebrigen icht davon ab, weitere
    ////Daten auf die serielle Schnittstelle zu schreiben.
    ////Dies erreichen wir, indem wir in den POINT Modus schalten
    ////(Commando 0x42 )
    bytesToRead = num;
    bytesReceived = 0;
    startTime = (long)time(NULL);
    endTime = startTime + 3;
    while (bytesToRead)
    {
        bResult = ReadFile(serialChannel, (bfr + (num - bytesToRead)), 1, &bytesReceived, NULL);
        if (bytesReceived)
        {
            bytesToRead -= 1;
        }
        else
        {
            bytesToRead = 0;
        }
    }
    int retVal = 0;
    if (bytesToRead > 0)
    {
        fprintf(stderr, "SERVER: lost bytes (%d read from %d, %d lost)\n",
                num - bytesToRead, num, bytesToRead);
        retVal = 0;
    }
    else
        retVal = 1;
    if (_stopping)
    {
        sendGroupMode();
        //Send a stream stop command
        sendSer((unsigned char *)"B", 1);
        CloseHandle(serialChannel);
        exit(0);
    }
    else
    {
        return retVal;
    }
}
