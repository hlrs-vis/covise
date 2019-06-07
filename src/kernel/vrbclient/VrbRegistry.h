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
template <class ClassType, class VarType>
class VrbRegistry
{
protected:
    std::map<const std::string, std::shared_ptr<ClassType>> myClasses;

    ///changes name to the read name and return the char which contains the classes variables
    void readClass(std::ifstream &file)
    {
        std::string className, delimiter, space;
        file >> className;
        file >> space; 
        std::shared_ptr<ClassType> cl = createClass(className, -1); // -1 = nobodies client ID
        myClasses[className] = cl;
        cl->readVar(file);
    }
   
    void clearRegistry() {
        //delete all entries and inform their observers
    }

public:

    virtual int getID() = 0;
    virtual std::shared_ptr<ClassType> createClass(const std::string &name, int id) = 0;
    void loadRegistry(std::ifstream &inFile) {

        while (!inFile.eof())
        {
            readClass(inFile);
        }

    }

    void saveRegistry(std::ofstream &outFile) const {
        for (const auto &cl : myClasses)
        {
            outFile << "\n";
            cl.second->writeClass(outFile);
        }
    }
};
}
#endif // !VrbRegistry_h 
