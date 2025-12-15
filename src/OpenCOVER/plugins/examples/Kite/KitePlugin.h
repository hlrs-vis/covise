/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef KITE_PLUGIN_H
#define KITE_PLUGIN_H

#include <cover/coVRPlugin.h>

class KitePlugin : public opencover::coVRPlugin
{
public:
    KitePlugin();
    ~KitePlugin() override;

    bool init() override;
};

#endif // KITE_PLUGIN_H
