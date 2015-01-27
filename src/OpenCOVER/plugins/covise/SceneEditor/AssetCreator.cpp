/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "AssetCreator.h"

#include "SceneUtils.h"

#include <iostream>

AssetCreator::AssetCreator()
{
}

AssetCreator::~AssetCreator()
{
}

SceneObject *AssetCreator::createFromXML(QDomElement *root)
{
    Asset *asset = new Asset();
    if (!buildFromXML(asset, root))
    {
        delete asset;
        return NULL;
    }
    return asset;
}

bool AssetCreator::buildFromXML(SceneObject *so, QDomElement *root)
{
    if (!buildGeometryFromXML((Asset *)so, root))
    {
        return false;
    }
    return SceneObjectCreator::buildFromXML(so, root);
}

bool AssetCreator::buildGeometryFromXML(Asset *asset, QDomElement *root)
{
    osg::ref_ptr<osg::Node> node = SceneUtils::createGeometryFromXML(root);
    if (node == NULL)
    {
        return false;
    }
    asset->setGeometryNode(node);

    return true;
}
