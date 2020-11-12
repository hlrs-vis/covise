/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#ifndef VrbRegistry_h 
#define VrbRegistry_h

#include "RegistryClass.h"
#include <util/coExport.h>
namespace vrb
{
class VRBEXPORT VrbRegistry
{
public:
	const regClass* getClass(const std::string& name) const;
	regClass* getClass(const std::string& name);

    virtual int getID() = 0;
    virtual std::shared_ptr<regClass> createClass(const std::string &name, int id) = 0;
	void deserialize(covise::TokenBuffer& tb);
	void serialize(covise::TokenBuffer& tb) const;

    typedef std::vector<std::shared_ptr<regClass>> ContainerType;
    ContainerType::iterator begin();
    ContainerType::const_iterator begin() const;
    ContainerType::iterator end();
    ContainerType::const_iterator end() const;

    virtual ~VrbRegistry() = default;
protected:
    ContainerType m_classes;
    ContainerType::iterator findClass(const std::string &className);
    void clearRegistry()
    {
        //delete all entries and inform their observers
    }
};

}
#endif // !VrbRegistry_h 
