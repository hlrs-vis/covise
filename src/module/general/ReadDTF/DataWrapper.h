/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file DataWrapper.h
 * @brief contains declaration of class DataWrapper
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 09.12.2003
 * created
 */

/** @class ClassInfo_DataWrapper
 * @brief used to register class DataWrapper at class manager.
 *
 * \b Description:
 *
 * Class defined by macro CLASSINFO(). This class is friend of
 * DataWrapper and is therefore the only class allowed to create new
 * objects. Class manager is the only class which is allowed to tell
 * ClassInfo_DataWrapper to create new objects of type DataWrapper.
 */

/** @class ::DataWrapper
 * @brief wraps reading, processing, and preparing of DTF data.
 *
 * This class is used by ReadDTF to read, process and prepare geometry and
 * CFD data available through DTF files. CFD-DTF is a data exchange format
 * designed to store CFD (computational fluid dynamics) data.
 *
 * DataWrapper uses the classes in packages DTF and DTF_Lib to read the CFD data.
 * Readed DTF data is processed and converted inside the wrapper object. That
 * data is used later to create coDoSet objects needed by the Covise module
 * ReadDTF to fill its output ports.
 *
 * DTF data can be read in either from a single DTF file or from multiple files
 * specified by a special index file.
 *
 * @note The class manager creates only 1 object of class DTF::Cell. Subsequent
 * calls to class manager result in retrieving the same object again and again.
 * That behaviour can be changed through editing DTF/Cell.cpp and setting the
 * maximum object count to the desired value.
 *
 * @see ReadDTF::compute(const char *port) for code which demonstrates the usage of this
 * wrapper class.
 */

/** @fn DataWrapper::DataWrapper();
 * @brief default constructor.
 *
 * Does basically nothing except of adding object statistic to StatisticManager
 */

/** @fn DataWrapper::DataWrapper(string className,
 int objectID);
 * @brief initializes new objects with className and objectID.
 *
 * @param className - name of the class for the object
 * @param objectID - ID assigned by class manager
 *
 * Adds object statistic to StatisticManager.
 */

/** @fn DataWrapper::~DataWrapper()
 * @brief destructor.
 *
 * Called when objects of type DataWrapper are deleted by ClassManager.
 *
 * Removes object statistic from StatisticManager.
 */

/** @fn bool DataWrapper::fillData(DTF::Data* tsData,
 string fileName);
 * @brief reads DTF data contained in file \c fileName.
 *
 * @param tsData   - data contained in file \c fileName is stored in this object
 * @param fileName - path to DTF file
 *
 * @return \c true if data could be read entirely. \c false on error.
 *
 * Reads the entire geometry and all zone data arrays from DTF file. All
 * zones are concatenated into one virtual zone to ensure that connectivities
* and associated zone data are correct. This also speeds up data extraction.
 *
 * If data extraction is successfull then \c tsData is added to
 * \a DataWrapper::data for later processing.
 *
 * If data extraction fails then the read data is deleted from memory.
 *
 * @note only simulation 1 is read. All other simulations are ignored.
 */

/** @fn bool DataWrapper::fillTimeSteps();
 * @brief fills time steps for later processing.
 *
 * @return \c true if operation was successfull. \c false on error.
 *
 * Processes DTF data read by earlier call of \a fillData() and prepares
 * it as time steps for Covise module ReadDTF.
 *
 * Each time step contains
 * - geometry
 * - zone data arrays
 *
 * which can be accessed later through calls to \a getGridSet() and
 * \a getDataSet().
 */

/** @fn coDistributedObject* DataWrapper::getGrid(int timeStepNr);
 * @brief get grid object from timestep \c timeStepNr.
 *
 * @param timeStepNr - timestep from which grid should be extracted
 *
 * @return pointer to object of type coDoUnstructuredGrid if grid is available
 * inside the given timestep. NULL if not available.
 *
 * Gets grid object from timeStep \c timeStepNr - if it exists.
 */

/** @fn coDistributedObject* DataWrapper::getData(int timeStepNr, string name);
 * @brief get data object from timestep \c timeStepNr
 *
 * @param timeStepNr - timestep from which zone data array should be extracted
 * @param name - name of the zone data array
 *
 * @return pointer to object of type coDoFloat if data \c name
 * is available inside the given timestep. NULL if not available.
 */

/** @fn virtual bool DataWrapper::init();
 * @brief makes some basic object initializations
 *
 * @return \c true if initialization was successfull. \c false on error.
 *
 * Called by ClassManager after new object was created.
 *
 * Initializes DTF::TimeSteps object \a timeSteps to NULL.
 */

/** @fn virtual void DataWrapper::clear();
 * @brief removes contained DTF data from memory
 *
 * This function is used to clear the memory occupied by DTF specific data.
 * It also removes all timestep related data from memory.
 */

/** @fn void DataWrapper::value2Name(int type, string& name);
 * @brief converts \c type to \c name
 *
 * @param type - int indicating the desired index in type list
 * @param name - data name needed by DTF library to find the data (output)
 *
 * Zone data is accessed inside DTF lib by providing a data name. Covise
 * choice parameters provide their values as index of currently selected
 * entry in select box.
 *
 * This function maps the select box entry to the corresponding DTF zone
 * data name.
 *
 * Default is second entry in type list since first entry means "None".
 */

/** @fn bool DataWrapper::readData(string fileName,
 map<int, string> typeList, bool isIndex);
 * @brief reads data from DTF file and prepares it for later processing
 *
 * @param fileName - path to DTF file
 * @param typeList - list of possible data types in DTF file
 * @param isIndex  - specifies if given fileName is an DTF index file.
 *
 * @return \c true if data could be read, extracted and processed. \c false
 * on error.
 *
* This function reads DTF data contained in file \c fileName. The found
 * data is extracted and saved into object of type DTF::TimeSteps. That class
 * is used later to create the geometry and data sets needed by the Covise
 * module ReadDTF to fill its output ports with timestep data.
 *
 * Raw DTF data contained in \a data is deleted after data extraction
 * to lessen memory occupation of DataWrapper.
 *
 * \a isIndex specifies if the given fileName is a simple DTF file or a text file
 * which contains the names of multiple DTF files. Such an index file is used
 * to read a simulation with time steps. Each file contains one time step and
 * must be contained in the index file in the following format:
 *
 * @code
 * /relative/path/to/index/file/FILE.DTF
 * @endcode
 *
 * This means that if index file is in \c /tmp then the entry is relative to
 * that path. e.g. \c files/FILE.DTF if \c FILE.DTF is in \c /tmp/files.
 */

/** @fn coDoSet* DataWrapper::getGridSet(string name, int begin, int end);
 * @brief get grid set containing geometry for given timestep range
 *
 * @param name  - name for the created grid set
 * @param begin - first frame in time step range
 * @param end   - last frame in time step range
 *
 * @return pointer to coDoSet object containing the geometry. NULL if
 * geometry extraction failed.
 *
 * This function returns a grid set containing the geometry for the given time
 * step range in DTF simulation. If \c begin is smaller than 0, then first time
 * step is at time 0. If \c end is greater than the number of time steps
 * available then end of time step range is last available time step.
 *
 * @note The created coDoSet object is never deleted since Covise crashes when
 * data assigned to a port is deleted.
 */

/** @fn coDoSet* DataWrapper::getDataSet(string name,
 string dataName,
 int begin,
 int end);
 * @brief get data set named \c name for data \c dataName for given time step
 * range.
 *
 * @param name - name of the created coDoSet object
 * @param dataName - name of the zone data array to store into coDoSet object
 * @param begin - starting frame for time step range
 * @param end - last frame for time step range
*
* @return pointer to coDoSet object containing all timesteps for given data
* \c dataName. NULL if data couldn't be found.
 *
 * Creates new coDoSet containing the given time step range for given data
 * \c dataName in a Covise readable format. Each timestep contains the
 * specified zone data array.
 *
 * @note The created coDoSet pointer is never deleted since Covise crashes
 * when data assigned to a port is deleted.
 */

/** @fn bool DataWrapper::hasData();
 * @brief checks if DataWrapper contains any zone data arrays.
 *
 * @return \c true if wrapper contains any data (geometry + additional zone
 * data). \c false if there is no data available.
 *
 * If number of time steps is <= 0 then false is returned.
 */

/** @fn bool DataWrapper::readSingle(string fileName);
 * @brief called to read in one single DTF file.
 *
 * @param fileName - path to DTF file
 *
 * @return \c true if reading was successfull. \c false if reading fails.
 *
 * The DTF data contained in \c fileName is read in, processed, and stored into
 * DataWrapper::timeSteps for later extraction.
 */

/** @fn bool DataWrapper::readMultiple(string indexFile);
 * @brief reads in DTF files described in a index file.
 *
 * @param indexFile - path to index file containing the names of DTF files
 *
 * @return \c true if successfull. \c false on error.
 *
 * The index file contains the relative path to DTF files which should be read
 * in. Each file is treated as one time step of the simulation.
 *
 * This function reads in the files described in the index file. Result is
 * stored in DataWrapper::timeSteps.
 */

/** @fn string DataWrapper::extractPath(string fileName);
 * @brief parses \c fileName and extracts the contained path name.
 *
 * @param fileName - name of the file from which the path has to be extracted
 *
 * @return path extracted from \c fileName. The returned string is empty if
 * extraction failed or path is empty.
 *
 * The given string is tokenized at delimiter "/". After all tokens are extracted
 * the path is put together - omitting the last token which is the name of the
 * file itself. The created path is returned.
 */

/** @fn bool DataWrapper::checkForFile(string fileName);
 * @brief checks if file could be opened.
 *
 * @param fileName - path to file which is to be checked
 *
 * @return \c true if file could be opened. \c false if opening failed.
 */

/** @var DTF::Data* DataWrapper::data;
 * @brief contains data read from DTF file(s)
 *
 * The contained data is cleared as soon as the time steps were extracted or
 * before \a readData() returns. Each vector entry contains geometry and data
 * for one time step.
 */

/** @var DTF::TimeSteps* DataWrapper::timeSteps;
 * @brief contains all timesteps of DTF simulation
 *
 * A DTF simulation may contain several timesteps. Each of this time steps
 * could contain geometry and zone data arrays. All of this data is stored
 * inside this object.
 */

/** @var map<int, string> DataWrapper::dataTypes;
 * @brief maps from "index in coChoiceParam" to "data name"
 */

/** @var static ClassInfo_DataWrapper DataWrapper::classInfo;
 * @brief used to register class DataWrapper at class manager.
 *
 * This object is used later by the class manager to create objects of type
 * DataWrapper.
 */

#ifndef __DATAWRAPPER_H_
#define __DATAWRAPPER_H_

#include "covise.h"
#include "DTF/timesteps.h"

#ifdef __sgi
using namespace std;
#endif

class ClassInfo_DataWrapper;

class DataWrapper : public Tools::BaseObject
{
    friend class ClassInfo_DataWrapper;

private:
    vector<DTF::Data *> data;
    map<int, string> dataTypes;
    DTF::TimeSteps *timeSteps;

    static ClassInfo_DataWrapper classInfo;

    DataWrapper();
    DataWrapper(string className,
                int objectID);

    bool checkForFile(string fileName);
    string extractPath(string fileName);

    bool fillData(DTF::Data *tsData,
                  string fileName);
    bool fillTimeSteps();

    coDistributedObject *getGrid(int timeStepNr);
    coDistributedObject *getData(int timeStepNr,
                                 string name);

    bool readSingle(string fileName);
    bool readMultiple(string indexFile);

public:
    virtual ~DataWrapper();

    coDoSet *getGridSet(string name,
                        int begin,
                        int end);

    coDoSet *getDataSet(string name,
                        string dataName,
                        int begin,
                        int end);

    bool hasData();

    bool readData(string fileName,
                  map<int, string> typeList,
                  bool isIndex);

    void value2Name(int type,
                    string &name);

    virtual void clear();
    virtual bool init();
};

CLASSINFO(ClassInfo_DataWrapper, DataWrapper);
#endif
