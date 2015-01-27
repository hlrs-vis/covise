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

#endif
#include <signal.h>
#ifdef __linux__
#define sigset signal
#endif

#include "fob.h"

#define SHMKEY ((key_t)325)
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
    serialChannel = -1;
    serverRunning = false;

#ifdef VISENSO
    allocSharedMemoryData();
#else
    receivers = new birdReceiver[maxNumReceivers];
    terminate = new bool;

    *terminate = false;
    memset(receivers, 0, maxNumReceivers * sizeof(birdReceiver));
#endif

    termio termPar;
    int n; // number of flushed bytes

    // open serial port
    serialChannel = open(portname, O_RDWR);

    if (serialChannel == -1)
    {
        fprintf(stderr, "client: can't open serial port\n");
        connected = false;
        return;
    }
    // configure the serial port
    // first get current configuration
    ioctl(serialChannel, TCGETA, &termPar);

#if !defined(_OLD_TERMIOS)
    // fprintf(stderr,"CONFIGURE for IRIX 6.4 SPEED=%d\n", baudrate);
    // IRIX 6.4 and later
    termPar.c_iflag = 0;
    termPar.c_oflag = 0;
    termPar.c_ospeed = baudrate;
    termPar.c_ispeed = baudrate;
    termPar.c_cflag = CS8 | CREAD;
    termPar.c_lflag = 0;
    termPar.c_line = 0;
    termPar.c_cc[VMIN] = 0;
    termPar.c_cc[VTIME] = 20;
#else
    int serialSpeed = 0;

    // for IRIX 6.2
    switch (baudrate)
    {
    case 1200:
        serialSpeed |= B1200;
        break;
    case 2400:
        serialSpeed |= B2400;
        break;
    case 4800:
        serialSpeed |= B4800;
        break;
    case 9600:
        serialSpeed |= B9600;
        break;
    case 19200:
        serialSpeed |= B19200;
        break;
    case 38400:
        serialSpeed |= B38400;
        break;
    case 57600:
        serialSpeed |= B57600;
        break;
    case 115200:
        serialSpeed |= B115200;
        break;
    default:
        serialSpeed |= B9600;
        break;
    }
    termPar.c_oflag &= ~OPOST;
    termPar.c_cc[VMIN] = 0;
    termPar.c_cc[VTIME] = 20;
    termPar.c_lflag = 0;
    termPar.c_oflag = 0;
    termPar.c_iflag = 0;
    termPar.c_cflag = (serialSpeed | CS8 | CREAD | CLOCAL) & (~CSTOPB) & (~PARENB);
#endif

    ioctl(serialChannel, TCSETA, &termPar);
    ioctl(serialChannel, FIONREAD, &n);
    fprintf(stderr, "%d bytes in the queue\n", n);

    mode = fmode;
    // we are connected now
    connected = true;
}

fob::~fob()
{
#ifdef VERBOSE
    fprintf(stderr, "fob::~fob\n");
#endif

    if (serialChannel > 0)
    {
        // shut-down the flock
        //stopStreaming();
        //flush();
        //autoconfig();
        //flush();
        //sendGroupMode();
        //flush();
        serverRunning = false;
        *terminate = true;
        //sleep(6);
        //kill(childID, SIGKILL);

        // and close the connection
        //close (serialChannel);
    }
}

int
fob::sendSer(unsigned char *bfr, int num)
{
    int numWrite = write(serialChannel, bfr, num);

    if (numWrite != num)
    {
        perror("fob::sendSer write");
        fprintf(stderr, "incomplete write: only %d of %d bytes written\n",
                numWrite, num);
        return (0);
    }
    else
    {
//ioctl(serialChannel, TCFLUSH, 2);
#if !defined(__hpux)
        if (tcdrain(serialChannel) < 0)
            perror("fob::sendSer tcdrain");
#endif
        // we have to wait here because the bird is not the fastest
        //sleep( 1 );
        return (1);
    }
}

int
fob::receiveSer(char *bfr, int num)
{
    int bytesReceived;
    int bytesToRead;
    long startTime, endTime;
    bytesToRead = num;
    bytesReceived = 0;
    startTime = (long)time(NULL);
    endTime = startTime + 3;
    while (bytesToRead && time(NULL) < endTime)
    {
        bytesReceived = read(serialChannel, (bfr + (num - bytesToRead)), bytesToRead);
        if (bytesReceived)
            bytesToRead -= bytesReceived;
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
        close(serialChannel);
        exit(0);
    }
    else
    {
        return retVal;
    }
}

#ifdef VISENSO
void
fob::startServerProcess()
{
    if (serverRunning)
        return;

#ifdef VERBOSE
    fprintf(stderr, "fob::startServerProcess\n");
#endif

    childID = fork();

    if (childID == -1)
    {
        fprintf(stderr, "CLIENT: fork failed\n");
    }

    if (childID == 0) // child process
    {
#ifdef VERBOSE
        fprintf(stderr, "CLIENT: server process forked\n");
#endif

#if !defined(__linux__) && !defined(__hpux)
        prctl(PR_TERMCHILD); // Exit when parent does
#endif

        sigset(SIGHUP, SIG_DFL); // Exit when sent SIGHUP by TERMCHILD
        while (1)
        {
            processSerialStream();
        }
    }
    else // master: make serial line unusable
    {
        serverRunning = true;
        //serialChannel = -1;
    }
}

void
fob::allocSharedMemoryData()
{
#ifdef VERBOSE
    fprintf(stderr, "fob::allocSharedMemoryData\n");
#endif
    int shmid;
    key_t shmkey = SHMKEY;
    while ((shmid = shmget(shmkey, maxNumReceivers * sizeof(birdReceiver) + 1,
                           PERMS | IPC_CREAT)) < 0)
    {
        fprintf(stderr, "shmget of %d failed... trying another one\n", shmkey);
        shmkey++;
        fprintf(stderr, "shmget of %d succesful\n", shmkey);
    }
    receivers = (birdReceiver *)shmat(shmid, (char *)0, 0);
    bzero((char *)receivers, maxNumReceivers * sizeof(birdReceiver));
    shmkey++;
    while ((shmid = shmget(shmkey, sizeof(terminate),
                           PERMS | IPC_CREAT)) < 0)
    {
        fprintf(stderr, "shmget of %d failed... trying another one\n", shmkey);
        shmkey++;
        fprintf(stderr, "shmget of %d succesful\n", shmkey);
    }
    terminate = (bool *)shmat(shmid, (char *)0, 0);

    *terminate = false;
    memset(receivers, 0, maxNumReceivers * sizeof(birdReceiver));
}
#endif /* VISENSO */
