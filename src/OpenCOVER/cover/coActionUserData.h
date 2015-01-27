/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_ACTION_USER_DATA_H
#define CO_ACTION_USER_DATA_H

/*! \file
 \brief  OpenVRUI interface to action user data

 \author Andreas Kopecki <kopecki@hlrs.de>
 \author (C) 2004
         High Performance Computing Center Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date  2004
 */

#include <OpenVRUI/sginterface/vruiActionUserData.h>

namespace opencover
{

/// Userdata that can be attached to Nodes in the scenegraph
class COVEREXPORT coActionUserData : public vrui::vruiActionUserData
{
public:
    coActionUserData(vrui::coAction *a); ///< Constructor
    virtual ~coActionUserData();
};
}
#endif
