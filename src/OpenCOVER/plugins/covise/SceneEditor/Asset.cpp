/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Asset.h"

Asset::Asset()
{
    _name = "";
    _type = SceneObjectTypes::ASSET;
}

Asset::~Asset()
{
}

/*
int Asset::receiveEvent(Event & e)
{
   // do my custom pre event processing here

   // dont forget to inform all my behaviors
   SceneObject::receiveEvent(e);

   // do my custom post event processing here
}
*/
