/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++
// CLASS CellToVert
//
// Initial version: 2001-07-02 we
// +++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++
// Changes:

#include <util/common.h>

#include <osg/Matrix>

/// class implementing a UDP receiver for tracking
class INPUT_LEGACY_EXPORT VRCTracker
{
public:
    // max. number of stations
    enum
    {
        MAX_STATIONS = 32
    };

    static const char *MAGIC;

    typedef struct
    {
        unsigned int stationID;
        float x, y, z; // cartesian coordinates of position
        float mat[9]; // orientation matrix
        float analog[2]; // up to 2 analog values
        unsigned int buttonVal; // up to 32 buttons
    } StationOutput;

    float unit;

private:
    /// UDP socket for data receive
    int d_socket;

    /// UDP socket for data receive
    int d_debugLevel;

    /// dump file for raw dump
    FILE *d_rawDump;

    /// pointer to SHM segment with data
    StationOutput *d_stationData;

    /// number of stations - in shared memory
    int *d_numStations;

    /// get the SHM segment
    StationOutput *allocSharedMemoryData();

    /// open UDP port receiving from anyhwere, return FD or -1 on error
    int openUDPPort(int portnumber);

    /// scaling factor to convert from sent data to cm
    float d_scale;

    // read the position of a certain station
    void getPositionMatrix(int station, float &x, float &y, float &z, float &m00, float &m01, float &m02, float &m10, float &m11, float &m12, float &m20, float &m21, float &m22);

    // debug outputs
    bool d_debugTracking, d_debugButtons; // enables raw debug tracking
    int d_debugStation;

public:
    /** Start the Daemon: Debug levels 0-3, no options yet defined
       * @param  portnumber  Port to receive packets from
       * @param  debugLevel  0-3
       * @param  options     unused so far
       */
    VRCTracker(int portnumber, int debugLevel, float scale, const char *options);

    /// D'tor
    ~VRCTracker();

    /// receive one package of data and store it into SHM
    void receiveData();

    // read matrix of a certain station
    void getMatrix(int station, osg::Matrix &mat);

    // read button states
    unsigned int getButton(int station);

    // read button states
    unsigned int getButtons(int station);

    // read button states
    void getAnalog(int station, float &d1, float &d2);

    // check whether initialisation worked
    bool isOk()
    {
        return (d_socket > 0) && (d_stationData != NULL);
    }

    /// this forks one process for receiving
    void mainLoop();
};
