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
    void loadFile(const std::string &filename) {
        clearRegistry();
        if (filename.find(".vrbreg") == std::string::npos)
        {
        std::cerr << "can not load file: wrong format" << std::endl;
            return;
        }
        std::ifstream inFile;
        inFile.open(filename, std::ios_base::binary);
        if (inFile.fail())
        {
            std::cerr << "can not load file: file does not exist" << std::endl;
            return;
        }
        std::string line;
        inFile >> line; //skip date
        while (!inFile.eof())
        {
            readClass(inFile);
        }

    }

    void saveFile(const std::string &path) const {
        //openFile
        std::ofstream outFile;
        std::string fullPath = path +"/" +getTime() + ".vrbreg";
        if (boost::filesystem::create_directory(path))
        {
            std::cerr << "Directory Created: " << path.c_str() << std::endl;
        }
        outFile.open(fullPath, std::ios_base::binary);
        outFile << getTime();
        for (const auto &cl : myClasses)
        {
            outFile << "\n";
            cl.second->writeClass(outFile);
        }
        outFile.close();
    }
private:
    std::string getTime() const {
        time_t rawtime;
        time(&rawtime);
        struct tm *timeinfo = localtime(&rawtime);

        char buffer[80];
        strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H-%M-%S", timeinfo);
        std::string str(buffer);
        return str;
    }
};
}
#endif // !VrbRegistry_h 
