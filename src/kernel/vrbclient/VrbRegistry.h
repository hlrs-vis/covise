/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#ifndef VrbRegistry_h 
#define VrbRegistry_h

#include "regClass.h"

#include <filesystem>
#include <fstream>
#include <ctime> 

namespace vrb
{
template <class ClassType, class VarType>
class VrbRegistry {
protected:
    std::map<const std::string, std::shared_ptr<ClassType>> myClasses;


    
    ///changes name to the read name and return the char which contains the classes variables
    char *readClass(std::string &name);
    ///reads the name and value out of stream
    void readVar(char *stream, std::string &name, covise::TokenBuffer &value);

public:


    void saveFile(const std::string &path) const {
        //openFile
        std::ofstream outFile;
        outFile.open(path);
        outFile << getTime();
        for (const auto &cl : myClasses)
        {
            outFile << std::endl;
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
        strftime(buffer, sizeof(buffer), "%d-%m-%Y %H:%M:%S", timeinfo);
        std::string str(buffer);
        return str;
    }
};
}
#endif // !VrbRegistry_h 
