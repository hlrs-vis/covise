/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__BIRDTRACKER_H)
#define __BIRDTRACKER_H

#ifdef _WIN32
#include <windows.h>
#else
#include <netinet/in.h>
#include <arpa/inet.h>
#endif

#include "birdReceiver.h"
#include "birdPacket.h"

class INPUT_LEGACY_EXPORT birdTracker
{
protected:
    // tcp/ip - stuff
    int sockId;
    int connected;
    struct sockaddr_in server;
    int birdPort;
    int buttonSystem;
    int buttonNumber;

    // string from covise.config, NULL allowed
    const char *bios_version;
    const char *numReceiversStr;

    static void continuousThread(void *data);
    // packet used for communication with the motion-star
    birdPacket packet;

    // fill-up the header of the packet then send it
    int send();

    // receive one packet from the motion-star
    int receive();

    // wait for one packet of type c from the motion-star
    int receive(birdPacket::command c);

    // just send the packet without modifying the header in any way
    // (especially do not change sequence)
    int sendAck();

    // we need to keep this up-to-date
    unsigned short int sequence;

    // number of receivers connected to the motion-star
    unsigned int numReceivers;

    // array of receivers, index ranges from 0 to numReceivers-1
    birdReceiver *receivers;
    // this array may be used for converting fbb-address to receivers-array-entry-number
    int fbb2nr[120];

    // ==0 if in singleShot mode, otherwise !0
    int transfer_mode;

    // get the position
    float getFloatPosition(unsigned char *ptr);
    // get float values
    float getFloatValue(unsigned char *ptr);

    // some tools
    void setBit(unsigned char *b, int nr);
    void clearBit(unsigned char *b, int nr);

    void allocSharedMemoryData();

    // messages sent to the motion-star

    // wake-up message
    int wakeUp();
    // shut-down
    int shutDown();
    // get status of the motion-star and setup the receivers-array
    int getSystemStatus();

    char *shm_start_addr;
    int dualTrans;

public:
    enum hemisphere
    {
        FRONT_HEMISPHERE = 0,
        REAR_HEMISPHERE = 1,
        UPPER_HEMISPHERE = 2,
        LOWER_HEMISPHERE = 3,
        LEFT_HEMISPHERE = 4,
        RIGHT_HEMISPHERE = 5
    };

    enum dataformat
    {
        FLOCK_NOBIRDDATA = 0,
        FLOCK_POSITION = 1,
        FLOCK_ANGLES = 2,
        FLOCK_MATRIX = 3,
        FLOCK_POSITIONANGLES = 4,
        FLOCK_POSITIONMATRIX = 5,
        FLOCK_QUATERNION = 7,
        FLOCK_POSITIONQUATERNION = 8,
        FLOCK_FEEDTHROUGH_DATA = 14,
        FLOCK_ERROR = 15
    };

    int *t_data[40];

    birdTracker(const char *ipAddr,
                int buttonNumberArg, // TrackerConfig.HAND_ADDR
                const char *numRecvArg, // MotionstarConfig.numReceivers
                const char *biosVersionArg, // MotionstarConfig.BIOS
                bool debugOutput, // isOn(MotionstarConfig.Debug,false)
                bool debugOutputAll); // isOn(MotionstarConfig.DebugAll,false)
    ~birdTracker();

    void DualTransmitter(int on)
    {
        dualTrans = on;
    };
    // wake-up the motion-star and retreive required receivers-information
    int init();

    // setup all receivers to use the given hemisphere and supply the given dataformat
    int setup(hemisphere hemi, dataformat df, unsigned int rate);

    // enable/disable the usage of the buttons and the individual filters
    //   (nr ranges from 0 to numReceivers-1)
    int setFilter(int nr, int buttons, int ac_nn, int ac_wn, int dc);

    // are we connected to the motion-star ?
    int isConnected();

    // get number of receivers attached to the motion-star
    int getNumReceivers();

    // poll for one single packet
    int singleShot();

    // start the motion-stars continuous-mode
    int runContinuous();

    // get a packet while in continuous-mode
    int getContinuousPacket();

    // has the receiver buttons ?
    int hasButtons(int nr);

    // process the received packet
    int processPacket();

    // setthe button system attached to a wireless motionstar, default is 0, no buttons through auxdata
    void setButtonSystem(int buttonSystem);

    // return the data retreived during the processPacket()-function
    //   for the specified receivers-array-index receiver
    //   (nr ranges from 0 to numReceivers-1)

    // return position and quaternions
    int getPositionQuaternion(int nr, float *x, float *y, float *z,
                              float *u, float *v, float *w, float *a);
    // return position and quaternions
    int getPositionEuler(int nr, float *x, float *y, float *z,
                         float *h, float *p, float *r);
    // return position and quaternions
    int getPositionMatrix(int nr, float *x, float *y, float *z,
                          float *m00, float *m01, float *m02, float *m10, float *m11, float *m12, float *m20, float *m21, float *m22);
    // return button-bytes
    int getButtons(int nr, unsigned int *buttons);
};
#endif
