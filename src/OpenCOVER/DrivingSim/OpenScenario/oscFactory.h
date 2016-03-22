/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_FACTORY_H
#define OSC_FACTORY_H

#include "oscExport.h"

#include <map>


namespace OpenScenario
{

/// \class Factory which creates OpenScenario Objects or Members
/// can be subclassed to create your own derived classes from the plane OpenScenario classes
template <typename T, typename TType>
class oscFactory
{

public:
    virtual ~oscFactory() {} ///< destructor
    
    template <typename TDerived>
    void registerType(TType &name)
    {
        static_assert(std::is_base_of<T, TDerived>::value, "oscObjectFactory::registerType doesn't accept this type because doesn't derive from base class");
        _createFuncs[name] = &createFunc<TDerived>;
    }
    template <typename TDerived>
    void registerType(TType name)
    {
        static_assert(std::is_base_of<T, TDerived>::value, "oscObjectFactory::registerType doesn't accept this type because doesn't derive from base class");
        _createFuncs[name] = &createFunc<TDerived>;
    }
    virtual T* create(TType &name)
    {
        typename std::map<TType,PCreateFunc>::const_iterator it = _createFuncs.find(name);
        if (it != _createFuncs.end())
        {
            return it->second();
        }
        return nullptr;
    }
    
private:
    template <typename TDerived>
    static T* createFunc()
    {
        return new TDerived();
    }
    
    typedef T* (*PCreateFunc)();
    std::map<TType,PCreateFunc> _createFuncs;
};

}

#endif // OSC_FACTORY_H
