/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ERROR_PLUGIN_H
#define ERROR_PLUGIN_H

/****************************************************************************\ 
 **                                                            (C)2008 ZAIK  **
 **                                                                          **
 ** Description: Error Plugin (use HUD to show COVISE errors)                **
 **                                                                          **
 **                                                                          **
 ** Author: Martin Aumueller <aumueller@uni-koeln.de>                        **
 **                                                                          **
\****************************************************************************/

#include <cover/coVRPlugin.h>

using namespace opencover;
class ErrorPlugin : public coVRPlugin
{
public:
    ErrorPlugin();
    virtual ~ErrorPlugin();

    void coviseError(const char *msg);
};
#endif
