/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ASSET_CREATOR
#define ASSET_CREATOR

#include "SceneObject.h"
#include "SceneObjectCreator.h"
#include "Asset.h"

#include <QDomElement>

class AssetCreator : public SceneObjectCreator
{
public:
    AssetCreator();
    virtual ~AssetCreator();

    virtual SceneObject *createFromXML(QDomElement *root);

protected:
    virtual bool buildFromXML(SceneObject *so, QDomElement *root);

private:
    bool buildGeometryFromXML(Asset *asset, QDomElement *root);
};

#endif
