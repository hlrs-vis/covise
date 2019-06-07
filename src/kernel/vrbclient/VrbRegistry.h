/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#ifndef VrbRegistry_h 
#define VrbRegistry_h

#include "regClass.h"
#include <boost/filesystem.hpp>
#include <fstream>
#include <ctime> 

namespace vrb
{
class VRBEXPORT VrbRegistry
{
protected:
    std::map<const std::string, std::shared_ptr<regClass>> myClasses;

    ///changes name to the read name and return the char which contains the classes variables
	void readClass(std::ifstream& file);

   
    void clearRegistry() {
        //delete all entries and inform their observers
    }

public:
	regClass* getClass(const std::string& name) const;
    virtual int getID() = 0;
    virtual std::shared_ptr<regClass> createClass(const std::string &name, int id) = 0;
	void loadRegistry(std::ifstream& inFile);
	void saveRegistry(std::ofstream& outFile) const;
};
}
#endif // !VrbRegistry_h 
