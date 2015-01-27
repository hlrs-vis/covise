/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file classmanager.h
 * @brief Contains definition of class ClassManager.
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 12.10.2003
 * created
 */

/** @class Tools::ClassManager
 * @brief creates, manages, and destroys objects.
 *
 * \b Description:
 *
 * Singleton class which manages class objects.
 *
 * Features:
 * - register classes
 * - create class objects
 * - delete created class objects (automatically when application exits)
 *
 * Get class manager instance:
 *
 * @code
 * ClassManager* cm = ClassManager::getInstance();
 * @endcode
 *
 * Create object:
 *
 * @code
 * BaseObject* bObj = cm->createObject("A"); // classname
 * @endcode
 *
 * Use object:
 *
 * @code
 * bObj->printFunc();
 * @endcode
 *
 * Clean up memory:
 *
 * Done automagically by class manager.
 */

/** @fn Tools::ClassManager::ClassManager();
 * @brief Default constructor.
 *
 * \b Description:
 *
 * Initializes map classInfos and vector objects for later use.
 */

/** @fn virtual Tools::ClassManager::~ClassManager();
 * @brief destructor.
 *
 * Called when class manager object is destroyed. Clears object lists when
 * application is finished.
 */

/** @fn bool Tools::ClassManager::addClass(string className, ClassInfo* classInfo);
 * @brief registers class at class manager
 *
 * @param className - name of the class to add
 * @param classInfo - classInfo object used to create new objects of class \c className
 *
 * @return \c true on success, \c false on error. \c false means that class is
 * already registered.
 */

/** @fn Tools::BaseObject* Tools::ClassManager::createObject(string className);
 * @brief creates new object of type \c className.
 *
 * @param className - type of class object to create.
 *
 * @return new object of type \c className. NULL if the given class name couldn't
 * be found.
 *
 * \b Description:
 *
 * This function searches in classInfos for class named \c className and
 * creates a new object of that type when that class is found.
 */

/** @fn bool Tools::ClassManager::deleteObject(int objectID);
 * @brief removes the object identified by objectID
 *
 * @param objectID - unique identifier of the object
 *
 * @return true if object could be located and deleted. otherwise false.
 *
 * \b Description:
 *
 * This function searches in objects list if there is an object with given ID.
 * If an object is found, then it is removed from objects list.
 */

/** @var map<string, Tools::ClassInfo*> Tools::ClassManager::classInfos;
 * @brief contains ClassInfo objects needed to create new objects of registered
 classes
 */

/** @var map<int, Tools::BaseObject*> Tools::ClassManager::objects;
 * @brief contains created objects.
 *
 * \b Description:
 *
 * Contained objects are destroyed when destructor of class ClassManager is
 * called.
 */

#ifndef __TOOLS_CLASSMANAGER_H_
#define __TOOLS_CLASSMANAGER_H_

#include "classinfo.h"
#include "OutputManagerBase.h"
#include "statisticmanagerbase.h"
#include "Helper.h"

class ClassInfo;

using namespace std;

namespace Tools
{
class ClassManager : public Singleton<ClassManager>
{
    friend class Singleton<ClassManager>::InstanceHolder;

private:
    map<string, ClassInfo *> classInfos;
    map<int, BaseObject *> objects;
    map<string, BaseObject *> singles;
    map<int, string> singleIDs;

    OutputManagerBase *outMan;
    StatisticManagerBase *statMan;
    int nextObjID;

    ClassManager();

    bool addSingle(string className, int objectID, BaseObject *object);
    bool removeSingle(int objectID);
    BaseObject *getSingle(string className);

    void clear();

public:
    virtual ~ClassManager();

    bool addClass(string className,
                  ClassInfo *classInfo);

    BaseObject *getObject(string className);
    bool deleteObject(int objectID);
    OutputManagerBase *getOutputMan();

    void selfTest();
    StatisticManagerBase *getStatMan();
};
};
#endif
