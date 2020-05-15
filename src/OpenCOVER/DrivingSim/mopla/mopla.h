/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef MOPLA_INCLUDE
#define MOPLA_INCLUDE
#include <list>
#include "ValidateMotionPlatform.h"
#include "XenomaiTask.h"
#include "GasPedal.h"
#include "KI.h"
#include "KLSM.h"
#include "Klima.h"
#include "Beckhoff.h"
#include "IgnitionLock.h"
#include <sys/time.h>

using namespace vehicleUtil;

namespace opencover
{
class mopla: public XenomaiTask
{
public:
    mopla(const char *filename);
    ~mopla();
    
    KI *p_kombi;
    KLSM *p_klsm;
    Klima *p_klima;
    VehicleUtil *vehicleUtil;
    Beckhoff *p_beckhoff;
    GasPedal *p_gaspedal;
    IgnitionLock *p_ignitionLock;
    ValidateMotionPlatform *motPlat;
    static const RTIME period = 1000000;
    unsigned long overruns;


    fasiUpdateManager *fum;

    void run();
    
    static mopla *myFasi;
    static mopla *instance()
    {
        return myFasi;
    };
    
};
}
#endif
