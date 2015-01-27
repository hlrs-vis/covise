/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file Singleton.h
 * @brief contains definition of template class Singleton
 * @author Alexander Martinez <kubus3561@gmx.de>
 *
 * This file demonstrates an example implementation of the singleton pattern
 * for C++.
 *
 */

/** @class Tools::Singleton
 * @brief providing implementation of singleton pattern to all derived classes.
 *
 * A singleton is a class of which only one object can be created. This is
 * achieved by the following steps:
 * - declare all constructors private (-> no objects can be created directly)
 * - add a static member variable which is retrieved as singleton object
 * - provide a static access function which returns the singleton object
 *
 * However singleton classes are a pain-in-the-ass when it comes to
 * inheritance: a lot of trouble accessing class objects with getInstance...
 *
 * ::Singleton provides a workaround for this problem. Inherited classes have to
 * declare this class as their friend and singleton functionality should work
 * for them, too.
 */

/** @fn Tools::Singleton::Singleton()
 * @brief Default constructor.
 *
 * Does absolutely nothing.
 */

/** @fn virtual Tools::Singleton::~Singleton()
 * @brief Default destructor.
 *
 * Does absolutely nothing. Called when objects are deleted.
 */

/** @fn T& Tools::Singleton<T>::getInstance()
 * @brief returns singleton object
 *
 * This function provides the access point for retrieving the only available
 * object of this class. It is actually provided by calling class Singleton::InstanceHolder
 *
 * @return Reference to singleton object
 */

/** @class Tools::Singleton::InstanceHolder
 * @brief inner class needed for singleton template to work.
 *
 * Contains the actual singleton object.
 */

/** @fn Tools::Singleton::InstanceHolder::InstanceHolder()
 * @brief default constructor
 *
 * Calls constructor of T to create the singleton object
 */

/** @fn Tools::Singleton::InstanceHolder::~InstanceHolder()
 * @brief default destructor
 *
 * deletes singleton object
 */

/** @var Tools::Singleton::InstanceHolder::m_MyClass
 * @brief singleton object
 *
 * This is the singleton object stored and provided by ::Singleton.
 */

#ifndef __TOOLS_SINGLETON_H_
#define __TOOLS_SINGLETON_H_

#include "baseinc.h"

/** @namespace Tools
 * @brief contains some general helper classes used by several classes
 */
namespace Tools
{

template <class T>
class Singleton
{
protected:
    Singleton(){};

public:
    virtual ~Singleton(){};

    class InstanceHolder
    {
    public:
        InstanceHolder()
            : m_MyClass(new T())
        {
        }
        ~InstanceHolder()
        {
            delete m_MyClass;
            m_MyClass = NULL;
        }
        T *m_MyClass;
    };

    static T *getInstance()
    {
        static InstanceHolder Instance;

        return Instance.m_MyClass;
    }

    friend class InstanceHolder;
};
};
#endif
