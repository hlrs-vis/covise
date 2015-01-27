/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __ACTION_DEFINED
#define __ACTION_DEFINED

#include "attribute.h"
typedef enum
{
    RAW,
    APP
} specifier;

typedef enum
{
    SERIAL_PORT,
    HEMISPHERE,
    HAND,
    HEAD,
    POLHEMUS_STYLUS,
    POLHEMUS_SENSOR,
    NUM_STATIONS,
    BAUD_RATE
} PolhemusItem;

typedef enum
{
    DEBUG_STATION,
    DEBUG_TRACKING,
    HANDSENSOR_OFFSET,
    HANDSENSOR_ORIENTATION,
    HAND_ADDR,
    HEADSENSOR_OFFSET,
    HEADSENSOR_ORIENTATION,
    HEAD_ADDR,
    TRANSMITTER_OFFSET,
    TRANSMITTER_ORIENTATION,
    INTERPOLATION_FILE,
    INTERPOLATION_FILE_X,
    INTERPOLATION_FILE_Y,
    INTERPOLATION_FILE_Z,
    LINEAR_MAGNETIC_FIELD_CORRECTION,
    NUM_SENSORS,
    ORIENT_INTERPOLATION,
    WRITE_CALIBRATION
} TrackerItem;

class actionClass
{
public:
    virtual void Tracker(TrackerItem item, const char *itemString, int x, int y, int z) = 0;
    virtual void Tracker(TrackerItem item, const char *itemString, int number) = 0;
    virtual void Tracker(TrackerItem item, const char *itemString, const char *str) = 0;
    virtual void Tracker(TrackerItem item, const char *itemString, specifier, const char *specAsString) = 0;

    virtual void Polhemus(PolhemusItem item, const char *itemString, int x, int y, int z) = 0;
    virtual void Polhemus(PolhemusItem item, const char *itemString, int number) = 0;
    virtual void Polhemus(PolhemusItem item, const char *itemString, const char *str) = 0;
    virtual void Polhemus(PolhemusItem item, const char *itemString, specifier, const char *specAsString) = 0;
};
#endif
