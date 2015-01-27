/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*! \file
 \brief  OpenVRUI interface to collaborative interface manager

 \author Andreas Kopecki <kopecki@hlrs.de>
 \author (C) 2004
         High Performance Computing Center Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date   2004
 */

#ifndef CO_COLLAB_INTERFACE_H
#define CO_COLLAB_INTERFACE_H

#include <OpenVRUI/sginterface/vruiCollabInterface.h>

namespace opencover
{
class coVRPlugin;
/// collaborative interface manager
class COVEREXPORT coCOIM : public vrui::vruiCOIM
{
public:
    coCOIM(coVRPlugin *);
    virtual ~coCOIM();
    coVRPlugin *getPlugin();
    void setPlugin(coVRPlugin *);

private:
    coVRPlugin *myPlugin; ///< pointer to the current plugin
};
}
#endif
