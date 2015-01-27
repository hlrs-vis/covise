/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef STANDALONE
#endif

#include <util/common.h>

#include "VRTracker.h"
#ifndef _WIN32
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <strings.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#define SHMKEY ((key_t)324)
#define PERMS 0666
#else
#include <windows.h>
#include <process.h>
#include <util/unixcompat.h>
#endif

#define SYSTEM_RUNNING 128
#define SYSTEM_ERROR 64
#define SYSTEM_FBB_ERROR 32
#define SYSTEM_LOCAL_ERROR 16
#define SYSTEM_LOCAL_POWER 8
#define SYSTEM_MASTER 4
#define SYSTEM_CRTSYNC_TYPE 2
#define SYSTEM_CRTSYNC 1

#define DEVICE_IS_ACCESSIBLE 128
#define DEVICE_IS_RUNNING 64
#define DEVICE_IS_RECEIVER 32
#define DEVICE_IS_ERC 16
#define DEVICE_IS_SRT 1

#define FLOCK_IS_RECEIVER 4
#define FLOCK_HAS_BUTTONS 8

#define INCHES_IN_MM 25.4

#include "birdTracker.h"
#include "birdPacket.h"
#include "birdReceiver.h"

int debugOutput = 0;
int debugOutputAll = 0;

birdTracker::birdTracker(const char *ipAddr,
                         int buttonNumberArg,
                         const char *numRecvArg,
                         const char *biosVersionArg,
                         bool /*debugOutput*/, bool /*debugOutputAll*/)
{
    // we're not yet connected and we've got no receivers
    connected = 0;
    receivers = NULL;
    birdPort = 6000;
    buttonSystem = 0;
    dualTrans = 0;

    buttonNumber = buttonNumberArg;
    bios_version = biosVersionArg;
    numReceiversStr = numRecvArg;

    // we start in singleShot mode
    transfer_mode = 0;

    // packets start with sequence 0
    sequence = 0;
    //debugOutput = coCoviseConfig::isOn("MotionstarConfig.Debug");
    //debugOutputAll = coCoviseConfig::isOn("MotionstarConfig.DebugAll");

    if (!ipAddr)
    {
        fprintf(stderr, "MotionStar driver: no IP address given");
    }

    // open socket
    if ((sockId = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        fprintf(stderr, "MotionStar driver: can't open stream socket");
        return;
    }

    // try to establish connection to motion-star
    memset(&server, 0, sizeof(server));
    server.sin_family = AF_INET;
    server.sin_port = htons(birdPort);
    server.sin_addr.s_addr = inet_addr(ipAddr);
    fprintf(stderr, "connecting to MotionStar at %s\n", ipAddr);
#ifdef __sgi
    if (connect(sockId, (struct SOCKADDR *)&server, sizeof(server)))
#else
    if (connect(sockId, (sockaddr *)&server, sizeof(server)))
#endif
    {
        fprintf(stderr, "can't connect to MotionStar\n");
        return;
    }

    // we are connected now
    connected = 1;
    allocSharedMemoryData();

    // done
    return;
}

birdTracker::~birdTracker()
{
    // shut-down the motion-star
    shutDown();
#ifdef _WIN32
    closesocket(sockId);
#else
    // and close the connection
    close(sockId);
#endif

    // clean up
    //delete[] receivers;

    // done
    return;
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

void
birdTracker::allocSharedMemoryData()
{

#ifdef _WIN32
    receivers = new birdReceiver[40];
#else
    // get shared memory segment for tracker output data ds1, ds2. ds3, ds4

    int shmid;
    key_t shmkey = SHMKEY;

    while ((shmid = shmget(shmkey, 40 * sizeof(birdReceiver) + 1, PERMS | IPC_CREAT)) < 0)
    {
        cout << "shmget failed" << endl;
        shmkey++;
    }
#ifdef DEBUG
    cout << "INFO: shmid: " << shmid << " shmkey: " << shmkey << endl;
#endif

    receivers = (birdReceiver *)shmat(shmid, (char *)0, 0);
#ifdef DEBUG
    printf("INFO: shm_start_addr: %x\n", shm_start_addr);
#endif

    memset((char *)receivers, '\0', 40 * sizeof(birdReceiver));
#endif
}

int birdTracker::init()
{
    if (wakeUp() < 0)
    {
        fprintf(stderr, "birdTracker: ERROR - wakeUp failed\a\n");
        return (-1);
    }

    if (getSystemStatus() < 0)
    {
        fprintf(stderr, "birdTracker: ERROR - getSystemStatus failed\a\n");
        return (-1);
    }

    return (0);
}

int birdTracker::setup(hemisphere hemi, dataformat df, unsigned int rate)
{
    unsigned int i, u;
    unsigned char *data;
    unsigned char ds = 0;
    char rate_string[7];

    switch (df)
    {
    case birdTracker::FLOCK_POSITION:
        ds = 0x30;
        break;
    case birdTracker::FLOCK_ANGLES:
        ds = 0x30;
        break;
    case birdTracker::FLOCK_MATRIX:
        ds = 0x90;
        break;
    case birdTracker::FLOCK_POSITIONANGLES:
        ds = 0x60;
        break;
    case birdTracker::FLOCK_POSITIONMATRIX:
        ds = 0xC0;
        break;
    case birdTracker::FLOCK_QUATERNION:
        ds = 0x40;
        break;
    case birdTracker::FLOCK_POSITIONQUATERNION:
        ds = 0x70;
        break;
    default:
        cerr << "oops birdTracker::Setup" << endl;
        break;
    }

    sprintf(rate_string, "%06d", rate * 1000);

    // request a system-status
    packet.setType(birdPacket::MSG_GET_STATUS, 0);
    packet.setDataSize(0);
    send();

    // wait for the answer
    if (receive(birdPacket::RSP_GET_STATUS) < 0)
        return (-1);

    // change the measurementRate
    data = (unsigned char *)packet.getData();
    for (u = 0; u < 6; u++)
        data[5 + u] = rate_string[u];

    // and send the new setup to the motion-star
    packet.setType(birdPacket::MSG_SEND_SETUP, 0);
    send();

    // wait for the answer
    if (receive(birdPacket::RSP_SEND_SETUP) < 0)
        return (-1);

    // now continue with the devices
    for (i = 0; i < numReceivers; i++)
    {
        // first we have to get the status of the device
        // so send a status-request
        packet.setType(birdPacket::MSG_GET_STATUS, receivers[i].address);
        packet.setDataSize(0);
        send();

        // wait for the answer
        if (receive(birdPacket::RSP_GET_STATUS) < 0)
            return (-1);

        // now perform the setup
        data = (unsigned char *)packet.getData();

        /* IAO:
       if (SensorState[Sensor].HasButton==TRUE) {
          NetDataBuffer.buffer[5]=0x0d;   // dc on ac, notc on, ac wide off, sudden off and buttons
       }
         else {
            NetDataBuffer.buffer[5]=0x05;   // dc on ac, notc on, ac wide off, sudden off
              }
              */
        // set dataformat, size and hemisphere
        if (hasButtons(i))
            data[5] = 0x0d; // dc on ac, notc on, ac wide off, sudden off and buttons
        else
            data[5] = 0x05; // dc on ac, notc on, ac wide off, sudden off
        data[6] = ((unsigned char)df) | ds;
        data[7] = 0x01; //reportrate
        data[10] = (unsigned char)hemi;
        data[11] = (unsigned char)receivers[i].address;

#ifdef BYTESWAP
        // get range and scale
        unsigned short Lsb, Msb;
        short int Word;
        Msb = data[8];
        Lsb = data[9];
        Word = ((Msb << 8) & 0x0F00) | (Lsb);
        receivers[i].range = Word;
        /// SPECIAL IAO HACK!!!!!!!
        receivers[i + 1].range = Word;
        receivers[i + 2].range = Word;
#else
        // get range and scale
        receivers[i].range = (*(signed short int *)(data + 8)) & 0x0FFF;
        /// SPECIAL IAO HACK!!!!!!!
        receivers[i + 1].range = (*(signed short int *)(data + 8)) & 0x0FFF;
        receivers[i + 2].range = (*(signed short int *)(data + 8)) & 0x0FFF;
#endif
        // send a setup-message
        packet.setType(birdPacket::MSG_SEND_SETUP, receivers[i].address);
        send();

        // wait for the answer
        if (receive(birdPacket::RSP_SEND_SETUP) < 0)
            return (-1);

        // on to the next device
    }

    // done
    return (0);
}

int birdTracker::setFilter(int nr, int buttons, int ac_nn, int ac_wn, int dc)
{
    unsigned char *data;
    dataformat df;
    unsigned char ds = 0;

    // request status for given device
    packet.setType(birdPacket::MSG_GET_STATUS, receivers[nr].address);
    packet.setDataSize(0);
    send();

    // wait for the answer
    if (receive(birdPacket::RSP_GET_STATUS) < 0)
        return (-1);

    // now get the data
    data = (unsigned char *)packet.getData();

    // set bits depending on parameters
    receivers[nr].add_button_data = buttons;
    // was #ifdef OLD_MOTIONSTAR_BIOS
    if (bios_version)
    {
        if (strcasecmp(bios_version, "OLD") == 0)
        {

            receivers[nr].add_button_data = 0;
        }
    }
    // was #endif

    // get current dataformat
    df = (dataformat)(data[6] & 0x0F);

    switch (df)
    {
    case birdTracker::FLOCK_POSITION:
        ds = 0x30;
        break;
    case birdTracker::FLOCK_ANGLES:
        ds = 0x30;
        break;
    case birdTracker::FLOCK_MATRIX:
        ds = 0x90;
        break;
    case birdTracker::FLOCK_POSITIONANGLES:
        ds = 0x60;
        break;
    case birdTracker::FLOCK_POSITIONMATRIX:
        ds = 0xC0;
        break;
    case birdTracker::FLOCK_QUATERNION:
        ds = 0x40;
        break;
    case birdTracker::FLOCK_POSITIONQUATERNION:
        ds = 0x70;
        break;
    default:
        cerr << "oops birdTracker::Setup" << endl;
        break;
    }

    //   if( buttons )
    //      ds += 16;

    // set dataformat and size
    data[6] = ((unsigned char)df) | ds;

    // clear / set flags
    if (buttons)
        setBit(data + 5, 3);
    else
        clearBit(data + 5, 3);

    if (ac_nn)
        setBit(data + 5, 2);
    else
        clearBit(data + 5, 2);

    if (ac_wn)
        setBit(data + 5, 1);
    else
        clearBit(data + 5, 1);

    if (dc)
        setBit(data + 5, 0);
    else
        clearBit(data + 5, 0);

    // and send the setup to the device
    packet.setType(birdPacket::MSG_SEND_SETUP, receivers[nr].address);
    send();

    // and wait for the answer
    return (receive(birdPacket::RSP_SEND_SETUP));
}

int birdTracker::isConnected()
{
    return (connected);
}

int birdTracker::getNumReceivers()
{
    return (numReceivers);
}

int birdTracker::hasButtons(int nr)
{
    return (receivers[nr].buttons_flag);
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

int birdTracker::singleShot()
{
    // send a singleShot-request
    packet.setType(birdPacket::MSG_SINGLE_SHOT, 0);
    packet.setDataSize(0);
    send();

    // wait for the answer
    return (receive());
}

void birdTracker::continuousThread(void *data)
{
    birdTracker *bt = (birdTracker *)data;
    while (1)
    {
        bt->getContinuousPacket();
        bt->processPacket();
    }
}

int birdTracker::runContinuous()
{
    // start continuous mode
    packet.setType(birdPacket::MSG_RUN_CONTINUOUS, 0);
    packet.setDataSize(0);
    send();

    transfer_mode = 1;

    // we should get an ack for that one ?!?
    //cerr << "set Continuous" << endl;
    receive(birdPacket::RSP_RUN_CONTINUOUS);
//cerr << "received first packet" << endl;

#ifdef _WIN32
    // Not used: "uintptr_t thread = "
    _beginthread(continuousThread, 0, this);
#else
    int ret;

    ret = fork();
    if (ret == -1)
    {
        //cout << "fork failed" << endl;
    }

    else if (ret == 0) // child process
    {
        // child code
        // read serial port and write data to shared memory

        //		cerr << "INFO: server forked" << endl;

        while (1)
        {
#ifdef DBGPRINT1
            fprintf(stderr, "bird server process is running ...\n");
#endif

            getContinuousPacket();
            processPacket();

            if (getppid() == 1)
            {
                //packet.setType( birdPacket::MSG_SLEEP );
                //packet.setDataSize(0);
                //send();

                //receive();

                //init();
                //shutDown();
                //fprintf(stderr, "SERVER: exit\n");
                exit(1);
                return (0);
            }
        }
    }
#endif
    //else
    //cerr << "INFO: client is running" << endl;

    return (0);
}

int birdTracker::getContinuousPacket()
{
    // just receive one packet
    return (receive());
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

int birdTracker::processPacket()
{
    unsigned char t;
    unsigned char *data;
    int total_size, current_size;
    int nr;
    int offs;

    float ratio;

    unsigned int fbb_addr, button_flag;
    unsigned int data_size, data_format;
    //dataformat df;

    // get packet-type
    t = ((unsigned char *)packet.getPtr())[8];

    // we only support DATA_PACKET_MULTI here
    if ((birdPacket::command)t != birdPacket::DATA_PACKET_MULTI)
    {
        cerr << "Wrong dataType, expected DATA_PACKET_MULTI but got " << (int)t << endl;
        return (-1);
    }

    // get the data-field
    data = (unsigned char *)packet.getData();
    total_size = packet.getDataSize();
    current_size = 0;
    data_size = 0;

    if (debugOutputAll)
        packet.dump();

    // now work through the entire data-field
    while (current_size + 2 < total_size)
    {
        if (debugOutput)
            cerr << "1current_size" << current_size << " data_size " << data_size << "total_size " << total_size << endl;
        // get the fbb-address (lower 7 bits)
        fbb_addr = ((unsigned int)(data[0] & 0x7F));

        // is there button-information available ? (highest bit)
        button_flag = ((unsigned int)(data[0] & 0x80)) >> 7;

        // get number of words (2bytes) in this record
        data_size = ((unsigned int)(data[1] & 0x0F));

        // and get the format of the data
        data_format = ((unsigned int)(data[1] & 0x0F0)) >> 4;
        //df = (dataformat)data_format;

        // in order to store the data we require the nr instead of the address
        nr = fbb2nr[fbb_addr];
        ratio = ((float)receivers[nr].range) * ((float)INCHES_IN_MM);
        if (dualTrans)
            ratio = ratio * 0.5f;
        if (debugOutput)
            cerr << "number:" << nr << endl;
        if (debugOutput)
            cerr << "dataformat:" << data_format << " data_size " << data_size << endl;

        // now finally process the data
        if (data_format == birdTracker::FLOCK_FEEDTHROUGH_DATA)
        {
            // feedthrough data (reserved for future use)
            //fprintf(stderr,"got FLOCK_FEEDTHROUGH_DATA size: %d, nr.: %d , fpp_addr %d\n",data_size,nr,fbb_addr);
            //fprintf(stderr,"data : %x %x\n",data[data_size*2],data[data_size*2]&(~0x80));
            /*	int i;
            for(i=0;i<16;i++)
            {
                if(idat&1<<i)
               fprintf(stderr,"1");
               else
               fprintf(stderr,"0");
            }
               fprintf(stderr,"\n");*/

            //int tmp = idat;
            //fprintf(stderr,"RawButtonData: %d, &80: %d\n",tmp, tmp&(~0x80));
            if (buttonSystem == B_MIKE)
            {
                /*int dn = *(data+(data_size)*2+3);
            unsigned char buttonState = *(data+1+dn);
            unsigned short int idat = *(unsigned short int *)(data+1+dn);
            #ifdef BYTESWAP
            receivers[buttonNumber].buttons = idat&(~0x80);
            #else
            receivers[buttonNumber].buttons = (idat>>8)&(~0x80);
            #endif

            alles krampf
            */
                receivers[buttonNumber].buttons = data[data_size * 2] & (~0x80);
            }
            if (buttonSystem == B_HORNET)
            {
                receivers[buttonNumber].buttons = data[2];
            }
            current_size += data_size * 2 + 2;
            data += data_size * 2 + 2;
            if (button_flag)
            {
                //fprintf(stderr, "buttons %d\n", nr);

                // this isn't implemented into the fucked-up protocoll, but
                // we keep it here for propable future-use
                current_size += 2;
                data += 2;
            }
        }
        else
        {
            // don't forgett to skip the 2byte header !
            offs = 2;

            /*
                  if( receivers[nr].add_button_data )
                  {
                     receivers[nr].buttons = *(unsigned short int *)(data+offs);
                     offs += 2;
                  }
                  else
                     receivers[nr].buttons = 0;
                  */

            switch (data_format)
            {
            case birdTracker::FLOCK_POSITION:

                receivers[nr].x = getFloatPosition(data + offs) * ratio;
                receivers[nr].y = getFloatPosition(data + offs + 2) * ratio;
                receivers[nr].z = getFloatPosition(data + offs + 4) * ratio;
                break;

            case birdTracker::FLOCK_POSITIONQUATERNION:

                receivers[nr].x = getFloatPosition(data + offs) * ratio;
                receivers[nr].y = getFloatPosition(data + offs + 2) * ratio;
                receivers[nr].z = getFloatPosition(data + offs + 4) * ratio;
                receivers[nr].u = getFloatPosition(data + offs + 6);
                receivers[nr].v = getFloatPosition(data + offs + 8);
                receivers[nr].w = getFloatPosition(data + offs + 10);
                receivers[nr].a = getFloatPosition(data + offs + 12);
                break;

            case birdTracker::FLOCK_POSITIONANGLES:

                receivers[nr].x = getFloatPosition(data + offs) * ratio;
                receivers[nr].y = getFloatPosition(data + offs + 2) * ratio;
                receivers[nr].z = getFloatPosition(data + offs + 4) * ratio;
                receivers[nr].h = getFloatPosition(data + offs + 6) * 180;
                receivers[nr].r = getFloatPosition(data + offs + 8) * 180;
                receivers[nr].p = getFloatPosition(data + offs + 10) * 180;
                break;

            case birdTracker::FLOCK_POSITIONMATRIX:

                receivers[nr].x = getFloatPosition(data + offs) * ratio;
                receivers[nr].y = getFloatPosition(data + offs + 2) * ratio;
                receivers[nr].z = getFloatPosition(data + offs + 4) * ratio;
                receivers[nr].m[0][0] = getFloatPosition(data + offs + 6);
                receivers[nr].m[1][0] = getFloatPosition(data + offs + 8);
                receivers[nr].m[2][0] = getFloatPosition(data + offs + 10);
                receivers[nr].m[0][1] = getFloatPosition(data + offs + 12);
                receivers[nr].m[1][1] = getFloatPosition(data + offs + 14);
                receivers[nr].m[2][1] = getFloatPosition(data + offs + 16);
                receivers[nr].m[0][2] = getFloatPosition(data + offs + 18);
                receivers[nr].m[1][2] = getFloatPosition(data + offs + 20);
                receivers[nr].m[2][2] = getFloatPosition(data + offs + 22);
                //fprintf(stderr,"posx: %f\n posy: %f\n posz: %f\n",receivers[nr].x,receivers[nr].y,receivers[nr].z);
                break;

            default:
                fprintf(stderr, "birdTracker::processPacket: undefined data_format\n");
            }

            // skip the just processed data
            current_size += data_size * 2 + 2;
            data += data_size * 2 + 2;

            // if button-data is available, then get it

            if (button_flag)
            {
                //fprintf(stderr, "buttons %d\n", nr);

                // this isn't implemented into the fucked-up protocoll, but
                // we keep it here for propable future-use
                if (buttonSystem == 0)
                {
                    receivers[nr].buttons = *(unsigned short int *)data;
                }
                current_size += 2;
                data += 2;
            }
            else
            {
                if (buttonSystem == 0)
                {
                    receivers[nr].buttons = 0;
                }
            }
        }
        if (debugOutput)
            cerr << "current_size" << current_size << " data_size " << data_size << "total_size " << total_size << endl;
    }

    // done
    return (0);
}

float birdTracker::getFloatPosition(unsigned char *ptr)
{
#ifdef BYTESWAP
    unsigned short Lsb, Msb;
    short int Word;
    Msb = ptr[0];
    Lsb = ptr[1];
    Word = (Msb << 8) | (Lsb);
    float value = ((float)Word) / (float)0x8000;
    //fprintf(stderr,"Wert: %f\n" ,value);
    return value;
#else
    float r;
    float ratio;

    signed short int is;
    //int r_num;

    unsigned char high_byte = ptr[0];
    unsigned char low_byte = ptr[1];
    is = ((signed short int)low_byte) & 0x00FF;
    is |= ((((signed short int)high_byte) << 8) & 0xFF00);

    r = (float)is;

    is = (signed short int)0x8000;
    ratio = -1.0 / ((float)is);

    //ratio = 0.0043945312f;

    return (r * ratio);
#endif
}

float birdTracker::getFloatValue(unsigned char *ptr)
{
#ifdef BYTESWAP
    unsigned short Lsb, Msb;
    short int Word;
    Msb = ptr[0];
    Lsb = ptr[1];
    Word = ((Msb << 8) & 0xFF00) | (Lsb);
    return (float)Word;

#else
    float r;
    //float ratio;

    signed short int is;
    //int r_num;

    unsigned char low_byte, high_byte;
    high_byte = ptr[0];
    low_byte = ptr[1];

    is = ((signed short int)low_byte) & 0x00FF;
    is |= ((((signed short int)high_byte) << 8) & 0xFF00);

    r = (float)is;
    return (r);
#endif
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

int birdTracker::send()
{
    // set sequence
    packet.setSequence(sequence++);

    // no timestamp used
    packet.setTimeStamp(0, 0);

// debugging - information
//   char *t;
//   fprintf(stderr, "send: %s\n", (t=packet.getType()));
//   delete[] t;

#ifndef __sgi
    return (sendto(sockId, (const char *)packet.getPtr(), packet.getSize(), 0, (sockaddr *)&server, sizeof(server)));
#else
    return (sendto(sockId, packet.getPtr(), packet.getSize(), 0, &server, sizeof(server)));
#endif
}

int birdTracker::receive(birdPacket::command c)
{
    int ret;
    unsigned char t;
    do
    {
        ret = receive();
        if (ret < 0)
            return ret;
        // get packet-type
        t = ((unsigned char *)packet.getPtr())[8];
        if ((birdPacket::command)t != c)
        {
            cerr << "expected " << c << " but received " << (int)t << endl;
        }
    } while ((birdPacket::command)t != c);
    return ret;
}

int birdTracker::receive()
{
    void *buf = packet.getPtr();
    int headerBytesReceived = 0;
    int bytesReceived;
    int toReceive = 16;

    // receive the entire packet (starting with the 16byte header)
    while (headerBytesReceived != toReceive)
    {
        if (debugOutput)
        {
            cerr << "toReceive" << toReceive << endl;
        }
        bytesReceived = recvfrom(sockId, (char *)buf, toReceive - headerBytesReceived, 0, NULL, NULL);

        if (bytesReceived < 0)
        {
            cerr << "recvfrom returned " << bytesReceived << endl;
            return -1;
        }

        headerBytesReceived += bytesReceived;

        if (headerBytesReceived == 16)
        {
            // header is here, so get actual packet-size
            toReceive = packet.getSize();
        }

        // add received data to packet
        buf = (void *)((char *)buf + bytesReceived);
    }

    // debugging - information
    if (debugOutput)
    {
        char *t;
        fprintf(stderr, "recv: %s\n", (t = packet.getType()));
        delete[] t;
    }

    return (headerBytesReceived);
}

int birdTracker::sendAck()
{
    // set sequence
    packet.setSequence(sequence);

// debugging - information
//   char *t;
//   fprintf(stderr, "send-ack: %s\n", (t=packet.getType()));
//   delete[] t;

#ifndef __sgi
    return (sendto(sockId, (const char *)packet.getPtr(), packet.getSize(), 0, (sockaddr *)&server, sizeof(server)));
#else
    return (sendto(sockId, packet.getPtr(), packet.getSize(), 0, &server, sizeof(server)));
#endif
}

void birdTracker::setBit(unsigned char *b, int nr)
{
    *b |= (unsigned char)(1 << nr);
    return;
}

void birdTracker::clearBit(unsigned char *b, int nr)
{
    *b &= (unsigned char)(~(1 << nr));
    return;
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

int birdTracker::wakeUp()
{
    if (!connected)
        return (-1);

    packet.setType(birdPacket::MSG_WAKE_UP);
    packet.setDataSize(0);
    send();

    return (receive(birdPacket::RSP_WAKE_UP));
}

int birdTracker::shutDown()
{
    if (!connected)
        return (0);

    if (transfer_mode)
    {
        packet.setType(birdPacket::MSG_STOP_DATA);
        packet.setDataSize(0);
        send();

        receive(birdPacket::RSP_STOP_DATA);
    }

    packet.setType(birdPacket::MSG_SHUT_DOWN, 0);
    packet.setDataSize(0);
    send();

    return (0); //receive() );
}

int birdTracker::getSystemStatus()
{
    unsigned char *data;

    unsigned int status, error;
    unsigned int i;

    // send a status-request to the motion-star
    packet.setType(birdPacket::MSG_GET_STATUS, 0);
    packet.setDataSize(0);
    send();

    // wait for the answer
    if (receive(birdPacket::RSP_GET_STATUS) < 0)
        return (-1);

    // general system status
    data = (unsigned char *)packet.getData();
    status = data[0];
    error = data[1];

    // check status for any error
    if (status & (SYSTEM_ERROR | SYSTEM_FBB_ERROR | SYSTEM_LOCAL_ERROR | SYSTEM_LOCAL_POWER))
    {
        fprintf(stderr, "birdTracker: ERROR - system error (status %d, error %d)\a\n", status, error);
        return (-1);
    }

    // see if the system is up
    if (!(status & SYSTEM_RUNNING))
    {
        fprintf(stderr, "birdTracker: ERROR - system not running (status %d, error %d)\a\n", status, error);
        return (-1);
    }

    // now get number of connected devices
    numReceivers = 0;
    for (i = 0; i < 120; i++)
        if ((data[16 + i] & (DEVICE_IS_ACCESSIBLE | DEVICE_IS_RECEIVER | DEVICE_IS_RUNNING)) == (DEVICE_IS_ACCESSIBLE | DEVICE_IS_RECEIVER | DEVICE_IS_RUNNING))
            numReceivers++;

    // alloc
    //receivers = new birdReceiver[numReceivers];

    // retreive addresses of the devices
    numReceivers = 0;
    for (i = 0; i < 120; i++)
    {
        if ((data[16 + i] & (DEVICE_IS_ACCESSIBLE | DEVICE_IS_RECEIVER | DEVICE_IS_RUNNING)) == (DEVICE_IS_ACCESSIBLE | DEVICE_IS_RECEIVER | DEVICE_IS_RUNNING))
        {
            receivers[numReceivers].address = i;
            fbb2nr[i] = numReceivers;
            numReceivers++;
        }
        else
            fbb2nr[i] = -1;
    }
    if (bios_version)
    {
        if (strcasecmp(bios_version, "OLD") == 0)
        {
            //if(numReceivers == 0)
            //{
            int numR = 5;
            numReceivers = 0;
            if (numReceiversStr)
            {
                if (sscanf(numReceiversStr, "%d", &numR) != 1)
                {
                    cerr << "birdTracker::getSystemStatus: sscanf1 failed" << endl;
                }
            }
            fprintf(stderr, "numR %d\n", numR);
            for (i = 2; i < (unsigned int)numR; i++)
            {
                receivers[numReceivers].address = i;
                fbb2nr[i] = i;
                numReceivers++;
            }
            //}
        }
    }
    // was #endif

    // get each receivers type
    for (i = 0; i < numReceivers; i++)
    {
        // send a status-request
        packet.setType(birdPacket::MSG_GET_STATUS, receivers[i].address);
        packet.setDataSize(0);
        send();

        // wait for the answer
        if (receive(birdPacket::RSP_GET_STATUS) < 0)
            return (-1);

        data = (unsigned char *)packet.getData();

        // we want to know if this receiver has buttons
        receivers[i].buttons_flag = data[0] & FLOCK_HAS_BUTTONS;
        receivers[i].add_button_data = data[5] & 8;
        //fprintf(stderr,"Receiver %d: buttons_flag: %d add_button_data: %d\n",i,receivers[i].buttons_flag,receivers[i].add_button_data);

        // on to the next one
    }

    return (0);
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

/*
int birdTracker::updatePosition()
{
   unsigned char *data;
   int total_size, current_size;
   int nr;

   unsigned int fbb_addr, button_flag;
   unsigned int data_size, data_format;
   unsigned int d;

receive();

unsigned char t;

t = ((unsigned char *)packet.getPtr())[8];
fprintf(stderr, "packet_type : %d\n", t);

data = (unsigned char *)packet.getData();
total_size = packet.getDataSize();
current_size = 0;

while( current_size+2 < total_size )
{
fbb_addr = ((unsigned int)(data[0]&0x0F));
button_flag = ((unsigned int)(data[0]&0x0F0))>>4;
data_size = ((unsigned int)(data[1]&0x0F));
data_format = ((unsigned int)(data[1]&0x0F0))>>4;

nr = fbb2nr[fbb_addr];

fprintf( stderr, "data_size: %d   data_format: %d    nr: %d   button_flag: %d\n", data_size, data_format, nr, button_flag );

if( data_format==14 )
{
// feedthrough data (reserved for future use)
current_size += data_size*2 + 2;
data += data_size*2 + 2;

}
else
{
switch( data_format )
{
case birdTracker::FLOCK_POSITIONQUATERNION:
d = *(short int *)data;
receivers[nr].x = ((float) d) / 100.0;
d = *(short int *)(data+2);
receivers[nr].y = ((float) d) / 100.0;
d = *(short int *)(data+4);
receivers[nr].z = ((float) d) / 100.0;
d = *(short int *)(data+6);
receivers[nr].u = ((float) d) / 100.0;
d = *(short int *)(data+8);
receivers[nr].v = ((float) d) / 100.0;
d = *(short int *)(data+10);
receivers[nr].w = ((float) d) / 100.0;
d = *(short int *)(data+12);
receivers[nr].a = ((float) d) / 100.0;
break;
case birdTracker::FLOCK_POSITIONANGLES:
d = *(short int *)data;
receivers[nr].x = ((float) d) / 100.0;
d = *(short int *)(data+2);
receivers[nr].y = ((float) d) / 100.0;
d = *(short int *)(data+4);
receivers[nr].z = ((float) d) / 100.0;

// keiner weiss ob's stimmt

d = *(short int *)(data+6);
receivers[nr].h = ((float) d) / 100.0;
d = *(short int *)(data+8);
receivers[nr].p = ((float) d) / 100.0;
d = *(short int *)(data+10);
receivers[nr].r = ((float) d) / 100.0;

break;

default:
fprintf(stderr, "birdTracker::singleShot - undefined data_format\n");
}

current_size += data_size*2 + 2;
data += data_size*2 + 2;

if( button_flag )
{
receivers[nr].buttons = *(unsigned short int *)data;
current_size += 2;
data += 2;
}

}
}

return( 0 );
}

*/

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

int birdTracker::getPositionQuaternion(int nr, float *x, float *y, float *z,
                                       float *u, float *v, float *w, float *a)
{
    // get data
    *x = receivers[nr].x;
    *y = receivers[nr].y;
    *z = receivers[nr].z;
    *u = receivers[nr].u;
    *v = receivers[nr].v;
    *w = receivers[nr].w;
    *a = receivers[nr].a;

    // done
    return (0);
}

int birdTracker::getPositionEuler(int nr, float *x, float *y, float *z,
                                  float *h, float *p, float *r)
{
    // get data
    *x = receivers[nr].x;
    *y = receivers[nr].y;
    *z = receivers[nr].z;
    *h = receivers[nr].h;
    *p = receivers[nr].p;
    *r = receivers[nr].r;

    // done
    return (0);
}

int birdTracker::getPositionMatrix(int nr, float *x, float *y, float *z,
                                   float *m00, float *m01, float *m02, float *m10, float *m11, float *m12, float *m20, float *m21, float *m22)
{
    // get data
    *x = receivers[nr].x;
    *y = receivers[nr].y;
    *z = receivers[nr].z;
    *m00 = receivers[nr].m[0][0];
    *m01 = receivers[nr].m[0][1];
    *m02 = receivers[nr].m[0][2];
    *m10 = receivers[nr].m[1][0];
    *m11 = receivers[nr].m[1][1];
    *m12 = receivers[nr].m[1][2];
    *m20 = receivers[nr].m[2][0];
    *m21 = receivers[nr].m[2][1];
    *m22 = receivers[nr].m[2][2];

    // done
    return (0);
}

void birdTracker::setButtonSystem(int bs)
{
    buttonSystem = bs;
}

int birdTracker::getButtons(int nr, unsigned int *buttons)
{
    //fprintf(stderr,"nr: %d receivers[nr].buttons: %d ",nr,receivers[nr].buttons);
    // set the button-state
    if (buttonSystem == B_MIKE)
    {
        *buttons = receivers[nr].buttons & (~0x80);
    }
    else
    {
#ifdef BYTESWAP
        if (receivers[nr].buttons == 4096)
            *buttons = 1;
        else if (receivers[nr].buttons == 12288)
            *buttons = 2;
        else if (receivers[nr].buttons == 28672)
            *buttons = 4;
        else
            *buttons = 0;
#else
        if (receivers[nr].buttons == 16)
            *buttons = 1;
        else if (receivers[nr].buttons == 48)
            *buttons = 2;
        else if (receivers[nr].buttons == 112)
            *buttons = 4;
        else
            *buttons = 0;
#endif
    }

    // done
    return (0);
}
