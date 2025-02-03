/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_DTRACK_H_
#define _CO_DTRACK_H_
/************************************************************************
 *									*
 *          								*
 *                            (C) 2001					*
 *              Computer Centre University of Stuttgart			*
 *                         Allmandring 30				*
 *                       D-70550 Stuttgart				*
 *                            Germany					*
 *									*
 *									*
 *	File			DTrack.cpp 				*
 *									*
 *	Description		DTrack optical tracking system interface class				*
 *									*
 *	Author			DUwe Woessner				*
 *									*
 *	Date			July 2001				*
 *									*
 *	Status			in dev					*
 *									*
 ************************************************************************/
#define MAXSENSORS 22
#define MAXBODYS 10
#define MAXFLYSTICKS 10
#define MAXBYTES 4000

#include <OpenThreads/Thread>
#include <sys/types.h>
#include <util/UDP_Sender.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <util/coTypes.h>
#define DTRACK_MAX_BUTTONS 10
#define DTRACK_MAX_VALUATORS 20

/************************************************************************
 *									*
 *	Description		class DTrack				*
 *									*
 *									*
 *	Author			D. Rainer				*
 *									*
 ************************************************************************/

class INPUT_LEGACY_EXPORT DTrack : public OpenThreads::Thread
{

public:
    typedef struct
    {
        float x, y, z; // cartesion coordinates of position
        float matrix[9]; // orientation matrix
        float ro;
        float lo;
        float alphaom;
        float lm;
        float alphami;
        float li;
    } FingerData;

    typedef struct
    {
        float quality;
        float x, y, z; // cartesion coordinates of position
        float az, el, roll; // euler orientation angles
        float matrix[9]; // orientation matrix
        int button[DTRACK_MAX_BUTTONS];
        float valuators[DTRACK_MAX_VALUATORS];
        DTrack::FingerData finger[5];
        int lr;
        int nf;
    } DTrackOutputData;

    DTrack(int portnumber, const char *sendStr);
    ~DTrack();
    void getPositionMatrix(int station, float *x, float *y, float *z, float *m00, float *m01, float *m02, float *m10, float *m11, float *m12, float *m20, float *m21, float *m22);
    void getButtons(int station, unsigned int *status); // get the bitmask of the first 32 buttons
    const FingerData *getFingerData(int station) const; /// Get a five component array of the finger tracking data from station.
    bool getButton(int station, int buttonNumber); // get any button status
    bool gotData()
    {
        return dataArrived;
    }; // returns true, if data ever arrived from the ART
    void sendStart();
    void sendStop();
    pid_t getSlavePID();

private:
    int port;
    int sock;
    char *c;
    int numbytes;
    bool dataArrived;
    char rawdata[MAXBYTES];

    covise::UDP_Sender *startStop;

    DTrackOutputData *stationData;

    void allocSharedMemoryData();
    bool openUDPPort();
    void mainLoop();
    void receiveData();
    int nextLine();
    int nextBlock();
    pid_t slavePid_;
    virtual void run();
    volatile bool running;
};

#endif
