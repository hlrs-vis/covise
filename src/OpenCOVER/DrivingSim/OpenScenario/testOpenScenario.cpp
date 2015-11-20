/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include <iostream>
#include <OpenScenarioBase.h>
#include <oscHeader.h>

using namespace OpenScenario;

int main(int argc, char **argv)
{
    OpenScenarioBase *osdb = new OpenScenarioBase();
    std::string fileName = "testScenario.xosc";
    if(argc > 1)
    {
        fileName = argv[1];
    }
    std::cerr << "trying to load " << fileName << std::endl;
    if(osdb->loadFile(fileName)== false)
    {
        std::cerr << "failed to load OpenScenarioBase from file " << fileName << std::endl;
        return -1;
    }
    oscHeader * h = osdb->header.getObject();
    if (h != NULL)
    {
        std::cerr << "revMajor:" << h->revMajor.getValue() << std::endl;
        std::cerr << "revMinor:" << osdb->header->revMinor.getValue() << std::endl;
        std::cerr << "Author:" << osdb->header->author.getValue() << std::endl;
    }
    if(argc > 2)
    {
        fileName = argv[2];
        std::cerr << "trying to save to " << fileName << std::endl;
        if(osdb->saveFile(fileName,true)== false)
        {
            std::cerr << "failed to save OpenScenarioBase to file " << fileName << std::endl;
            delete osdb;
            return -1;
        }
    }
    delete osdb;
    return 0;
}
