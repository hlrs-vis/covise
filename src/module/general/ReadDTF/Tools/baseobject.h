/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file baseobject.h
 * @brief contains definition of class Tools::BaseObject
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 12.10.2003
 * created
 */

/** @class Tools::BaseObject
 * @brief Base class for all managed objects.
 *
 * \b Description:
 *
 * The Class Manager stores objects of type Tools::BaseObject. All objects
 * which should be managed through the class manager MUST be from a class
 * derived from Tools::BaseObject.
 *
 * To make use of non-virtual functions inherited from Tools::BaseObject
 * the objects created by the class manager must be casted to the correct
 * pointer types.
 *
 * e.g.
 * @code
 * DTF::LibIF* libIF = (DTF::LibIF*) cm->getObject("DTF::LibIF");
 * @endcode
 *
 * If the objects are casted to an incorrect type then a segmentation fault
 * is the most probable result.
 */

/** @fn Tools::BaseObject::BaseObject();
 * @brief default constructor
 *
 * \b Description:
 *
 * Called when new objects of class BaseObject are created. Actually, this
 * function isn't called because there are no instances of BaseObject.
 *
 * The object counter is incremented when this function is called.
 */

/** @fn Tools::BaseObject::BaseObject( int objectID );
 * @brief constructor which initializes new objects with given object ID.
 *
 * @param objectID - unique identifier for the created object
 *
 * \b Description:
 *
 * The class manager is used to create new objects of this class. The object ID
 * is needed by the class manager to identify the object.
 *
 * The object counter is incremented when this function is called.
 */

/** @fn Tools::BaseObject::BaseObject(string className, int objectID);#
 * @brief constructor for new objects managed by class manager
 *
 * @param className - name of the class for the object. Needed by statistic
 * manager for object statistics.
 * @param objectID - unique identifier for the object needed by the class
 * manager to identify and delete the created object.
 *
 * \b Description:
 *
 * The class manager contains base pointers to objects from classes derived
 * from BaseObject. These objects are created in their classes through an
 * overloaded constructor which calls this constructor for base
 * initializations.
 *
 * The object counter is incremented when this constructor is called.
 */

/** @fn virtual Tools::BaseObject::~BaseObject();
 * @brief destructor
 *
 * \b Description:
 *
 * Called when object is destroyed. Frees memory occupied by the object by
 * calling \a clear().
 *
 * The object counter is decremented when this destructor is called.
 */

/** @fn int Tools::BaseObject::getID();
 * @brief get ID of the object
 *
 * @return object ID. -1 if the ID wasn't set.
 *
 * \b Description:
 *
 * Returns the ID of the object assigned by the class manager when object was
 * created.
 */

/** @fn virtual bool Tools::BaseObject::init();
 * @brief initializes new objects
 *
 * @return true if initialization was successfull. false on error.
 *
 * \b Description:
 *
 * Avoids several problems encountered at design time when the class manager
 * was created. Some objects have child functions. This means: they have
 * objects with specialized functionality related to a subset of the containing
 * class. These objects are created also by the class manager which lead to the
 * problem that two objects got the same object ID, which leads to another
 * problem: they can't be managed any more through the class manager.
 *
 * Therefore objects are created first and then initialized through the init()
 * function.
 */

/** @fn virtual void Tools::BaseObject::clear();
 * @brief clean-up memory occupied by object
 *
 * \b Description:
 *
 * Called every time when member variables of the object and occupied memory
 * should be cleaned up. Main use is in destructor of the class.
 */

/** @fn static int Tools::BaseObject::getNumObj();
 * @brief get number of objects of this class
 *
 * @return number of objects currently existing of this type
 *
 * \b Description:
 *
 * Returns the number of objects which are currently in use and managed by the
 * class manager. If new objects are created then the counter is incremented.
 * It is decremented when the destructor is called.
 */

/** @fn bool Tools::BaseObject::addChildFunc(string shortName,
 BaseObject* object);
 * @brief add new child function to map of child functions.
 *
 * @param shortName - key under which child func should be accessed in map
 * @param object - the object which is to be stored as value in map
 *
 * @return \c true if object could be added to list of child functions.
 * \c false on error.
 *
 * \b Description:
*
 * Adds given object (if not already contained in map) to map under given
 * short name as key.
 */

/** @fn Tools::BaseObject* Tools::BaseObject::getChildFunc(string shortName);
 * @brief get object from list of child functions.
 *
 * @param shortName - name by which the object is accessed in map of
 * child functions.
 *
 * @return pointer to the object which contains the desired child functions.
 * \c NULL if object could not be found.
 *
 * \b Description:
 *
 * The returned pointer must be casted to the correct type to be of any use
 * to the caller.
 *
 * e.g.
 * @code
 * DTF_Lib::Zone* zone = (DTF_Lib::Zone*) (*libif).getChildFunc("Zone");
 * @endcode
 */

/** @fn virtual string Tools::BaseObject::getClassName();
 * @brief get name of class
 *
 * @return string with class name
 *
 * This function retrieves the class name set during object creation.
 */

/** @var map<string, BaseObject*> Tools::BaseObject::childFuncs;
 * @brief child functions of the object.
 *
 * \b Description:
 *
 * Some objects have child functions which are objects containing specific
 * subsets of access functions. Such child functions are contained in this map.
 * To actually use these child functions, they must be casted to their correct
 * type.
 *
 * The pointers are created, managed, and deleted by the class manager.
 */

/** @var int Tools::BaseObject::objectID;
 * @brief ID of the object
 *
 * \b Description:
 *
 * This ID is assigned by the class manager when an object is created. It can
 * be used to tell the class manager to delete an object.
 *
 * Objects inside the class manager are indexed and accessed through their
 * object ID.
 *
 * @attention EACH OBJECT MUST HAVE ITS OWN UNIQUE ID!!
 */

/** @var static int Tools::BaseObject::numObj;
 * @brief number of currently existing objects of this type in memory
 *
 * \b Description:
 *
 * Each created object of a type derived from BaseObject increments this
 * counter. Each call of a destructor of a type derived from BaseObject
 * decrements this counter.
 *
 * The class manager restricts some classes to a specific number of maximum
 * objects created from them. e.g. DTF_Lib::LibIF is a single object.
 */

/** @var string Tools::BaseObject::className;
 * @brief name of the class
 *
 * \b Description:
 *
 * This is the name of the class. It is needed by the statistic manager to
 * hold statistics about creation and deletion of objects of this class.
 */

#ifndef __BASEOBJECT_H_
#define __BASEOBJECT_H_

#include "baseinc.h"

using namespace std;

namespace Tools
{
class BaseObject
{
    // normal members
private:
    int objectID;
    string className;

protected:
    map<string, BaseObject *> childFuncs;

    bool addChildFunc(string shortName, BaseObject *object);
    BaseObject *getChildFunc(string shortName);

public:
    BaseObject();
    BaseObject(string className, int objectID);

    virtual ~BaseObject();

    int getID();
    virtual bool init();

    virtual void clear();

    virtual string getClassName();

private:
    static int numObj;

public:
    static int getNumObj();
};
};
#endif
