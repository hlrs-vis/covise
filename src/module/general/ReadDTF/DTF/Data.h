/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file DTF/Data.h
 * @brief contains definition of class DTF::Data
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 28.10.2003
 * created
 * @date 5.11.2003
 * moved static member classInfo to private scope.
 */

/** @class DTF::ClassInfo_DTFData
 * @brief used to register class DTF::Data at class manager
 *
 * \b Description:
 *
 * Class defined by macro CLASSINFO(). This class is friend of
 * DTF::Data and is therefore the only class allowed to create new
 * objects. Class manager is the only class which is allowed to tell
 * ClassInfo_DTFData to create new objects of type DTF::Data.
 */

/** @class DTF::Data;
 * @brief contains DTF data extracted from a CFD-DTF file.
 *
 * This class contains parts of the data normally contained inside a CFD-DTF
 * file.
 *
 * The data is ordered hierarchically - this means that one has to use
 * following commands to get a specific cell from DTF::Data object:
 *
 * @code
 * Tools::ClassManager* cm = Tools::ClassManager::getInstance();
 * DTF::Data* data = (DTF::Data*) cm->getObject("DTF::Data");
 * if ( data->read(FILENAME) )
 * {
 *   // ... initialization and checking...
 *   // to get cell 1 in zone 1 of simulation 1
 *   DTF::Cell* cell = data->getSim(1)->getZone(1)->getCell(1);
 * }
 * @endcode
 *
 * @note The class manager creates only 1 object of class DTF::Data. Subsequent
 * calls to class manager result in retrieving the same object again and again. This behaviour
 * can be changed through editing DTF/Data.cpp and setting the maximum object
 * count to the desired value.
 */

/** @fn DTF::Data::Data();
 * @brief default constructor
 *
 * Adds object statistic to statistic manager.
 */

/** @fn DTF::Data::Data(string className, int objectID);
 * @brief creates new objects with given class name and object ID
 *
 * @param className - name of the class
 * @param objectID - unique identifier for the object
 *
 * The class manager creates a new object of class DTF::Data and assigns it
 * an unique identifier by which he refers to it.
 *
 * Initializes member \a file to \c NULL.
 *
 * Adds object statistic to statistic manager.
 */

/** @fn virtual DTF::Data::~Data();
 * @brief destructor
 *
 * Clears list of simulation objects of type DTF::Sim*. The contained
 * objects are deleted later by the class manager when his destructor is
 * called.
 *
 * Tells statistic manager that object has been deleted.
 */

/** @fn bool DTF::Data::read(string fileName);
 * @brief reads DTF data contained in CFD-DTF file.
 *
 * @param fileName - path to DTF file
 *
 * @return \c true if data could be read. \c false if not.
 *
 * First \a clear() is called to remove data which has been readed in a subsequent
 * call. After that step the given file name is sent to the DTF lib interface.
 * If it points to a valid DTF file then the DTF data is extracted simulation
 * per simulation. The actual data is read in by the created simulation objects
 * in a hierarchical way. This means:
 * - data reads simulations
 * - simulation reads zones
 * - zone reads cells and nodes
 *
 * Read simulations are stored in member \a sims.
 */

/** @fn File* DTF::Data::getFile();
 * @brief gets file info.
 *
 * @return DTF::File object containing general DTF file information. NULL if
 * empty.
 */

/** @fn Sim* DTF::Data::getSim(int simNum);
 * @brief gets simulation with given number
 *
 * @param simNum - number of the simulation
 *
 * @return DTF::Sim object related to given simulation number \c simNum.
 *
 * This function searches in simulation list for the simulation with given
 * number. If it is found then the associated object is returned. If it isn't
 * found then \c NULL is returned.
 */

/** @fn int DTF::Data::getNumSims();
 * @brief gets number of simulations
 *
 * @return number of simulations in DTF file. \c 0 if there are no simulations.
 */

/** @fn virtual void DTF::Data::clear();
 * @brief clears data from memory
 *
 * This function removes the DTF data contained in object of class DTF::Data.
 * It iterates through simulation list and calls clean-up function \c clear() of
 * each contained DTF::Sim object. Then the class manager is instructed to
 * remove the object.
 *
 * Member \a file is also deleted by the class manager.
 *
 * @note Call this function before reading DTF data to erase old data from
 * memory.
 */

/** @fn virtual bool DTF::Data::init();
 * @brief basic initializations done by class manager
 *
 * @return always \c true.
 *
 * Clears the simulation list to be sure that it is empty. Probably not
 * necessary but nevermind...
 */

/** @fn virtual void DTF::Data::print();
 * @brief print data to stdout
 *
 * This function is used for debugging purposes. It iterates through the
 * simulation list and calls \c print() operation of each simulation object.
 */

/** @var File* DTF::Data::file;
 * @brief contains general information about a parsed DTF file.
 *
 * @see DTF::File for a list of the informations which are available about a
 * DTF file.
 */

/** @var map<int, Sim*> DTF::Data::sims;
 * @brief list of simulations in DTF file.
 *
 * This map contains all simulations found in a former call to
 * \a DTF::Data::read(). Key is the simulation number which points to a
 * DTF::Sim object.
 */

/** @var static ClassInfo_DTFData DTF::Data::classInfo;
 * @brief used to register class DTF::Data.
 *
 * This static object is used to register class DTF::Cell at the class manager.
 * The class manager is responsible for creation and deletion of new objects
 * of class DTF::Data.
 */

#ifndef __DTF_DATA_H_
#define __DTF_DATA_H_

using namespace std;

#include "File.h"
#include "Sim.h"

#include "../DTF_Lib/file.h"

namespace DTF
{
class ClassInfo_DTFData;

class Data : public Tools::BaseObject
{
    /** This class is the only class allowed to create new objects of DTF::Data
       */
    friend class ClassInfo_DTFData;

private:
    File *file;
    map<int, Sim *> sims;

    Data();
    Data(string className, int objectID);
    static ClassInfo_DTFData classInfo;

public:
    virtual ~Data();

    bool read(string fileName);

    File *getFile();
    Sim *getSim(int simNum);
    int getNumSims();

    virtual void clear();
    virtual bool init();
    virtual void print();
};

CLASSINFO(ClassInfo_DTFData, Data);
};
#endif
