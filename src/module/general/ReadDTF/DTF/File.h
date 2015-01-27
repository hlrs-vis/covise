/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file DTF/File.h
 * @brief contains definition of class DTF::File
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 28.10.2003
 * created
 * @date 5.11.2003
 * moved static member classInfo to private scope.
 */

/** @class DTF::ClassInfo_DTFFile
 * @brief used to register class DTF::File at class manager
 *
 * \b Description:
 *
 * Class defined by macro CLASSINFO(). This class is friend of
 * DTF::File and is therefore the only class allowed to create new
 * objects. Class manager is the only class which is allowed to tell
 * ClassInfo_DTFFile to create new objects of type DTF::File.
 */

/** @class DTF::File;
 * @brief contains general informations about a CFD-DTF file.
 *
 * The contained informations are:
 * - creation time of file
 * - DTF lib version which created the file
 * - last modification time of file
 * - file origin
 * - scaling factor
 * - file title
 *
 * Use following code to acquire file informations:
 *
 * @code
 * Tools::ClassManager* cm = Tools::ClassManager::getInstance();
 * DTF::Data* data = (DTF::Data*) cm->getObject("DTF::Data");
 *
 * if ( data->read("filename.DTF") )
 * {
 *   DTF::File* file = data->getFile();
 *
 *   if ( file != NULL )
 *   {
 *     cout << "title: " << file->getTitle() << endl;
 *     // proceed for further file informations
 *   }
 * }
 * @endcode
 *
 * @note The class manager creates only 1 object of class DTF::File.
 * Subsequent calls to class manager result in retrieving the same object
 * again and again. This behaviour can be changed through editing DTF/File.cpp
 * and setting the maximum object count to the desired value.
 */

/** @fn DTF::File::File();
 * @brief default constructor.
 *
 * Adds object statistic to statistic manager.
 */

/** @fn DTF::File::File(string className, int objectID);
 * @brief creates new objects with given class name and object ID.
 *
 * @param className - name of the class
 * @param objectID - unique identifier for created object
 *
 * The class manager creates a new object of class DTF::File and assigns it a unique identifier
 * by which he refers to it.
 *
 * Initializes member variables to following values:
 * - title :     "none"
 * - origin:     "none"
 * - dtfVersion: ""
 * - scaling:    1.0
 *
 * Adds object statistic to statistic manager.
 */

/** @fn DTF::File::~File();
 * @brief destructor
 *
 * Tells statistic manager that object has been removed.
 */

/** @fn bool DTF::File::read(string fileName);
 */

/** @fn string DTF::File::getTitle();
 */

/** @fn string DTF::File::getOrigin();
 */

/** @fn bool DTF::File::getCreationTime(time_t& creationTime);
 */

/** @fn bool DTF::File::getModTime(time_t& modTime);
 */

/** @fn string DTF::File::getDTFVersion();
 */

/** @fn double DTF::File::getScaling();
 */

/** @fn virtual void DTF::File::print();
 */

/** @var string DTF::File::title;
 */

/** @var string DTF::File::origin;
 */

/** @var time_t DTF::File::created;
 */

/** @var time_t DTF::File::modified;
 */

/** @var string DTF::File::dtfVersion;
 */

/** @var double DTF::File::scaling;
 */

/** @var static ClassInfo_DTFFile DTF::File::classInfo;
 */

#ifndef __DTF_FILE_H_
#define __DTF_FILE_H_

#include <ctime>
#include "../DTF_Lib/libif.h"
#include "../DTF_Lib/info.h"

using namespace std;

namespace DTF
{
class ClassInfo_DTFFile;

class File : public Tools::BaseObject
{
    friend class ClassInfo_DTFFile;

private:
    string title;
    string origin;
    time_t created;
    time_t modified;
    string dtfVersion;
    double scaling;

    File();
    File(string className, int objectID);

    static ClassInfo_DTFFile classInfo;

public:
    virtual ~File();

    bool read(string fileName);

    bool getCreationTime(time_t &creationTime);
    string getDTFVersion();
    bool getModTime(time_t &modTime);
    string getOrigin();
    double getScaling();
    string getTitle();

    virtual void print();
};

CLASSINFO(ClassInfo_DTFFile, File);
};
#endif
