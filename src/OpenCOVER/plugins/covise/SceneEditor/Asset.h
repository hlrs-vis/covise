/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ASSET_H
#define ASSET_H

#include "SceneObject.h"

class Asset : public SceneObject
{
public:
    Asset();
    virtual ~Asset();

    // virtual int receiveEvent(Event & e);
};

#endif
