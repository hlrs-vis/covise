/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *									*
 *          								*
 *                            (C) 1996					*
 *              Computer Centre University of Stuttgart			*
 *                         Allmandring 30				*
 *                       D-70550 Stuttgart				*
 *                            Germany					*
 *									*
 *									*
 *	File			polhemusdrvr.h 				*
 *									*
 *	Description		polhemus driver class			*
 *				extra process for reading the		*
 *				serial port				*
 *									*
 *	Author			D. Rainer				*
 *									*
 *	Date			January 5th '97				*
 *									*
 *	Status			in dev					*
 *									*
 ************************************************************************/

#include <util/coTypes.h>
typedef struct
{
    float x, y, z; // cartesion coordinates of position
    float dx, dy, dz; // relative movement
    float az, el, roll; // euler orientation angles
    float xdircos[3]; // x direction cosines
    float ydircos[3]; // y direction cosines
    float zdircos[3]; // z direction cosines
    float w, q1, q2, q3; // orientation quaternion
    int button; // stylus switch status
} stationOutputData;

/************************************************************************
 *									*
 *	Description		class fastrak				*
 *									*
 *									*
 *	Author			D. Rainer				*
 *									*
 ************************************************************************/

class INPUT_LEGACY_EXPORT fastrak
{
private:
    int s1, s2, s3, s4; // receiver input numbers (stations)

    float hx, hy, hz;
    int numActiveStations;
    char serialPortName[10]; // string for the serial port name "/dev/ttyd3"
#ifdef WIN32
    HANDLE desc;
#else
    int desc; // file desciptor for the serial port
#endif
    int baudrate;
    int buttonDevice;
    int bufLen;

    static void continuousThread(void *data);
    stationOutputData *ds1, *ds2, *ds3, *ds4;
    int *resetFlag;
    void allocSharedMemoryData();
    void reinitialize();
    int openSerialPort();
    int openSerialPort_irix_6_4();
    void resetReferenceFrame(int station);
    void setReferenceFrame(int station, float Ox, float Oy, float Oz,
                           float Xx, float Xy, float Xz, float Yx, float Yy, float Yz);
    void setAsciiFormat();
    void disableContinousOutput();
    void enableContinousOutput();
    void setUnitCentimeters();
    void setUnitInches();
    void setBoresight(int station);
    void unsetBoresight(int station);
    void sendFastrakCmd(char *cmd_buf);
    void setOutput(int station);
    void calibrateStation(int station);
    void setStationActive(int station);
    void setStationPassive(int station);
    void serverDummyRead();

public:
    enum
    {
        BUTTONDEVICE_STYLUS = 0,
        BUTTONDEVICE_WAND = 1
    };

    fastrak(const char *portname, int baudrate, int numStations, int buttonDevice);
    ~fastrak();

    int readActiveStations();
    // testing the connection to serial port
    int testConnection();

    // reset tracker
    void reset();

    // user configuration
    void setHemisphere(int station, float p1, float p2, float p3);
    void setPositionFilter(float f, float flow,
                           float fhigh, float factor);
    void setAttitudeFilter(float f, float flow,
                           float fhigh, float factor);

    // calibration
    void calibrateStation(int station, float Ox, float Oy, float Oz,
                          float Xx, float xy, float Xz, float Yx, float Yy, float Yz);

    // set and unset a station
    void setStation(int station);
    void unsetStation(int station);

    void setStylusMouseMode();

    // fork extra process
    void start();

    // get output data
    void getAbsPositions(int station, float *x, float *y, float *z);
    void printAbsPositions(int station);
    void getRelPositions(int station, float *dx, float *dy, float *dz);
    void getEulerAngles(int station, float *az, float *el, float *roll);
    void getQuaternions(int station, float *w, float *q1, float *q2,
                        float *q3);
    void getXDirCosines(int station, float xdircos[3]);
    void getYDirCosines(int station, float ydircos[3]);
    void getZDirCosines(int station, float zdircos[3]);
    void getStylusSwitchStatus(int station, unsigned int *status);
};
