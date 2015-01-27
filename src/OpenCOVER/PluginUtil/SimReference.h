/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SIM_REFERENCE
#define SIM_REFERENCE

#include <util/coExport.h>
#include <osg/Referenced>
#include <string>
#include <vector>

namespace opencover
{
class PLUGIN_UTILEXPORT SimReference : public osg::Referenced
{
public:
    SimReference(const char *SimuPath, const char *SimuName = "undef");
    virtual ~SimReference()
    {
    }

    std::vector<std::string> getSimPath();
    std::vector<std::string> getSimName();
    void addSim(const char *SimuPath, const char *SimuName = "undef");

private:
    std::vector<std::string> SimPath;
    std::vector<std::string> SimName;
};
}
#endif
