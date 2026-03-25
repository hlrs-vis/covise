/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _AURAL_REALITY_PLUGIN_H
#define _AURAL_REALITY_PLUGIN_H

#include <cover/coVRPlugin.h>
#include <cover/ui/Owner.h>

class AuralRealityPlugin : public opencover::coVRPlugin, public opencover::ui::Owner
{
public:
    AuralRealityPlugin();
    ~AuralRealityPlugin();
    static AuralRealityPlugin *instance() { return plugin; };

private:
    static AuralRealityPlugin *plugin;
    bool update();
};

#endif
