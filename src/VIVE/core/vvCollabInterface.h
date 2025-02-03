/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#pragma once

#include <OpenVRUI/sginterface/vruiCollabInterface.h>

namespace vive
{
class vvPlugin;
/// collaborative interface manager
class VVCORE_EXPORT coCOIM : public vrui::vruiCOIM
{
public:
    coCOIM(vvPlugin *);
    virtual ~coCOIM();
    vvPlugin *getPlugin();
    void setPlugin(vvPlugin *);

private:
    vvPlugin *myPlugin; ///< pointer to the current plugin
};
}
