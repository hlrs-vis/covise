/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ERROR_PLUGIN_H
#define ERROR_PLUGIN_H


#include <cover/coVRPlugin.h>

using namespace opencover;

// use HUD to show COVISE errors
class ErrorPlugin : public coVRPlugin
{
public:
    ErrorPlugin();
    virtual ~ErrorPlugin();

    void coviseError(const char *msg);
};
#endif
