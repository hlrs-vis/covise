/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: ExampleTracker OpenCOVER Plugin (is polite)                          **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                  **
 **                                                                          **
 ** History:  								                                         **
 ** June 2008  v1	    				       		                                **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "ExampleTracker.h"
#include <cover/coVRPluginSupport.h>
using namespace opencover;
ExampleTracker::ExampleTracker()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "Starting ExampleTracker\n");
}

unsigned int ExampleTracker::button(int station)
{
    static bool toggle = true;
    if (station == 1) // return some random button presses for station 1
    {
        toggle = !toggle;
        if (toggle)
        {
            return 1; // first button pressed, second button released
        }
        return 2; // second button pressed, first button released
    }
    return 0;
}

void ExampleTracker::getMatrix(int station, osg::Matrix &mat)
{
    double time = cover->frameTime();
    double angh = (time - (int)time) * 2.0 * M_PI;
    double angv = ((time - (((int)(time / 2.0)) * 2)) / 2.0) * 2.0 * M_PI;
    // return a moving head position ( 2m away from the origin)
    if (station == 0)
    {
        mat.makeTranslate(sin(angv) * 200, -2000, cos(angv) * 100);
    }
    else // and a moving hand position
    {
        mat.makeTranslate(sin(angh) * 200, 0, cos(angh * 2) * 100);
    }
}

// this is called if the plugin is removed at runtime
ExampleTracker::~ExampleTracker()
{
    fprintf(stderr, "Goodbye\n");
}

COVERPLUGIN(ExampleTracker)
