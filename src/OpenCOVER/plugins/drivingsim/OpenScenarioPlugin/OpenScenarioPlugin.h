/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OPENSCENARIO_PLUGIN_H
#define OPENSCENARIO_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: OpenScenario                           					  **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                **
 **                                                                          **
 ** History:  								                                **
 ** Nov-01  v1	    				       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include <cover/coVRPlugin.h>

namespace OpenScenario
{
class OpenScenarioBase;
}

class OpenScenarioPlugin : public opencover::coVRPlugin
{
public:
    OpenScenarioPlugin();
    ~OpenScenarioPlugin();
	
	static int loadOSC(const char *filename, osg::Group *loadParent, const char *key);
	int loadOSCFile(const char *filename, osg::Group *loadParent, const char *key);

	static OpenScenarioPlugin *plugin;

	bool init();

    // this will be called in PreFrame
    void preFrame();

private:
	OpenScenario::OpenScenarioBase *osdb;
};

#endif //OPENSCENARIO_PLUGIN_H