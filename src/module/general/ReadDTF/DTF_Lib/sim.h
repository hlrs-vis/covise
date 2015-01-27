/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/sim.h
 * @brief contains definition of class DTF_Lib::Sim
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 * @date 5.11.2003
 * moved static member classInfo to private scope
 */

/** @class DTF_Lib::ClassInfo_DTFLibSim
 * @brief used to register class DTF_Lib::Sim at class manager
 *
 * \b Description:
 *
 * Class defined by macro CLASSINFO(). This class is friend of
 * DTF_Lib::Sim and is therefore the only class allowed to create new
 * objects. Class manager is the only class which is allowed to tell
 * ClassInfo_DTFLibSim to create new objects of type DTF_Lib::Sim.
 */

/** @class DTF_Lib::Sim
 * @brief contains access functions for simulation related data.
 *
 * \b Description:
 *
 * Simulations store simulation data and zones.
 */

/** @fn DTF_Lib::Sim::Sim();
 * @brief default constructor
 *
 * \b Description:
 *
 * Calls constructor of DTF_Lib::LibObject for the actual initializations.
 */

/** @fn DTF_Lib::Sim::Sim( int objectID );
 * @brief constructor which initializes new objects with given object ID.
 *
 * @param objectID - unique identifier for the created object
 *
 * \b Description:
 *
 * The class manager is used to create new objects of this class. The object ID
 * is needed by the class manager to identify the object.
 */

/** @fn virtual DTF_Lib::Sim::~Sim();
 * @brief default destructor.
 *
 * \b Description:
 *
 * Called when objects are destroyed.
 */

/** @fn bool DTF_Lib::Sim::queryNumSDs(int simNum,
int& numSDs);
 * @brief get number of simulation data arrays in simulation
 *
 * @param simNum - simulation number
 *
 * @param numSDs - number of simulation data for given simulation number
 (output)
 *
 * @return \c false on error, \c true on success.
 *
* \b Description:
*
* wraps calls to \c dtf_query_nsds().
*/

/** @fn bool DTF_Lib::Sim::queryNumZones(int simNum,
            int& numZones );
 * @brief get number of zones in simulation
 *
 * @param simNum - simulation number
 *
 * @param numZones - number of zones in simulation (output)
 *
 * @return \c false on error, \c true on success.
 *
 * \b Description:
*
 * wraps calls to \c dtf_query_nzones().
 */

/** @fn bool DTF_Lib::Sim::queryMinMax(int simNum,
          vector<double>& minMax );
 * @brief not implemented
 *
 * @param simNum - simulation number
 *
 * @param minMax - vector with xyz range of grid (6 values). (output)
 *
 * @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
* \b Description:
 *
 * wraps calls to \c dtf_query_minmax_sim_d() and \c dtf_query_minmax_sim_s().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Sim::querySimDescr(int simNum,
            string& description );
 * @brief get simulation description
 *
 * @param simNum - simulation number
 *
 * @param description - string describing the simulation
 *
 * @return \c false on error, \c true on success.
 *
 * \b Description:
*
 * wraps calls to \c dtf_query_simdescr().
 */

/** EOD */

/** BOC */

#ifndef __DTF_LIB_SIM_H_
#define __DTF_LIB_SIM_H_

#include "baseinc.h"

namespace DTF_Lib
{
class ClassInfo_DTFLibSim;

class Sim : public LibObject
{
    friend class ClassInfo_DTFLibSim;

private:
    Sim();
    Sim(string className, int objectID);

    static ClassInfo_DTFLibSim classInfo;

public:
    virtual ~Sim();

    virtual bool setFileHandle(int handle);
    virtual Tools::BaseObject *operator()(string className);

    bool queryNumSDs(int simnum,
                     int &numSDs);

    bool queryNumZones(int simNum,
                       int &numZones);

    bool queryMinMax(int simNum,
                     vector<double> &minMax);

    bool querySimDescr(int simNum,
                       string &description);
    virtual bool init();
};

CLASSINFO(ClassInfo_DTFLibSim, Sim);
};
#endif

/** EOC */
