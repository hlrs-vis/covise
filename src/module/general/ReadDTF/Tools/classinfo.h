/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file classinfo.h
 * @brief contains definition of class Tools::ClassInfo.
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 12.10.2003
 * created
 */

/** @class Tools::ClassInfo
 * @brief Used to create new objects of classes.
 *
 * \b Description:
 *
 * To enhance classes for use by the class manager use the following define:
 *
 * \c CLASSINFO(ClassInfo_Derived,Derived);
 *
 * This creates a new class named \c ClassInfo_Derived, which can be used to create
 * new objects of type \c Derived.
 *
 * The class is registered by calling: (in .cpp)
 *
 * \c CLASSINFO_OBJ(ClassInfo_Derived,Derived,"Derived", MAX_OBJ);
 */

/** @var string Tools::ClassInfo::className;
 * @brief name of the class to create.
 *
 * This value is used when registering class 'className' at class manager.
 */

/** @fn Tools::ClassInfo::ClassInfo(string className, int maxObj);
 * @brief registers class \c className at class manager.
 *
 * @param className - name of the class to register
 * @param maxObj - maximum number of objects. 0 for blocked, 1 for single
 object, INT_MAX for unrestricted.
 */

/** @fn virtual Tools::ClassInfo::~ClassInfo();
 * @brief Destructor.
 *
 * Called when objects are destroyed.
 */

/** @fn virtual Tools::BaseObject* Tools::ClassInfo::New(int objectID);
 * @brief creates new objects of type \c className with given objectID.
 *
 * @param objectID - unique identifier for the object
 *
 * @return new object of base type BaseObject (actually of type className).
 *
 * \b Description:
 *
 * The new object contains an object ID assigned by the class manager. This
 * object ID is used as argument for the constructor of new objects of type
 * Tools::BaseObject* and its derived classes.
 */

/** @fn virtual bool Tools::ClassInfo::maxObjReached();
 * @brief checks if the maximum object count is reached.
 *
 * @return true if maximum object is reached, else false.
 *
 * \b Description:
 *
 * The class manager is not allowed to serve more objects than the maximum
 * object count.
 */

/** @fn virtual int Tools::ClassInfo::getMaxObj();
 * @brief get maximum object count
 *
 * @return The maximum number of objects the class manager is allowed to
 * create from the served class.
 *
 * \b Description:
 *
 * Returns the maximum object count for the served class.
 */

/** @var int Tools::ClassInfo::maxObj;
 * @brief maximum number of objects
 *
 * \b Description:
 *
 * The maximum number of objects the class manager is allowed to create from
 * the served class.
 */

#ifndef __CLASSINFO_H_
#define __CLASSINFO_H_

#include "baseobject.h"

using namespace std;

/** @brief prototype for classes derived from ClassInfo.
 *
 * \b Description:
 *
 * These derived classes are used to register classes derived from BaseObject
 * at the class manager.
 *
 * Now follows the description of the contained functions:
 *
 * -# x(string className, int maxObj): adds class info object to class manager.
 *sets maximum object count
 * -# virtual ~x(): does nothing
 * -# Tools::BaseObject* New(int objectID): creates new objects of type y
 * -# virtual int getMaxObj(): gets maximum objects count
 * -# virtual bool maxObjReached(): checks if maximum object count is reached
 */
#define CLASSINFO(x, y)                                                    \
    class x : public Tools::ClassInfo                                      \
    {                                                                      \
        friend class Tools::ClassManager;                                  \
                                                                           \
    public:                                                                \
        x(string className, int maxObj) : ClassInfo(className, maxObj)     \
        {                                                                  \
            Tools::ClassManager::getInstance()->addClass(className, this); \
            this->maxObj = maxObj;                                         \
        }                                                                  \
        virtual ~x(){};                                                    \
                                                                           \
    private:                                                               \
        Tools::BaseObject *New(int objectID)                               \
        {                                                                  \
            return new y(getClassName(), objectID);                        \
        }                                                                  \
        virtual int getMaxObj()                                            \
        {                                                                  \
            return this->maxObj;                                           \
        }                                                                  \
        virtual bool maxObjReached()                                       \
        {                                                                  \
            if (y::getNumObj() >= maxObj)                                  \
                return true;                                               \
            return false;                                                  \
        }                                                                  \
        virtual int getNumObj()                                            \
        {                                                                  \
            return y::getNumObj();                                         \
        }                                                                  \
    };

#define INC_OBJ_COUNT(className) Tools::ClassManager::getInstance()->getStatMan()->createdObj(className);

#define DEC_OBJ_COUNT(className) Tools::ClassManager::getInstance()->getStatMan()->deletedObj(className);

/** @brief prototype initialization for class objects of type ClassInfo_x
 *
 * \b Description:
 *
 * Provides macro for static initialization of the classInfo object used
 * to register a class at the class manager.
 *
 * Given arguments have the following meaning:
 *
 * - clsInfo: type name of the ClassInfo_x class
 * - type: type name of the class derived from Tools::BaseObject which is to
 * be registered at the class manager.
 * - className: string describing the class which is to be registered at class
 * manager
 * - maxObj: number of maximum allowed objects of this class. 0 means blocked,
 * 1 means single objects. Use INT_MAX if you don't want to restrict the class
 * to a few objects.
 */
#define CLASSINFO_OBJ(clsInfo, type, className, maxObj) clsInfo type::classInfo(className, maxObj);

namespace Tools
{
class ClassInfo
{
    friend class ClassManager;

    string className;

protected:
    int maxObj;

    ClassInfo(string className, int maxObj);
    virtual ~ClassInfo();

    virtual BaseObject *New(int objectID);
    virtual bool maxObjReached();
    virtual int getMaxObj();
    virtual int getNumObj();

    string getClassName();
};
};
#endif
