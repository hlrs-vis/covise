/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coPanelGeometry.h>

#include <OpenVRUI/sginterface/vruiPanelGeometryProvider.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/sginterface/vruiTransformNode.h>

using std::string;

namespace vrui
{

coPanelGeometry::coPanelGeometry(const string &name)
{
    this->textureName = name;
    provider = 0;
}

coPanelGeometry::coPanelGeometry()
{
    textureName = "";
    provider = 0;
}

coPanelGeometry::~coPanelGeometry()
{
    delete provider;
}

void coPanelGeometry::attachGeode(vruiTransformNode *node)
{
    if (!provider)
        provider = vruiRendererInterface::the()->createPanelGeometryProvider(this);

    provider->attachGeode(node);
}

float coPanelGeometry::getWidth()
{
    if (!provider)
        provider = vruiRendererInterface::the()->createPanelGeometryProvider(this);

    return provider->getWidth();
}

float coPanelGeometry::getHeight()
{
    if (!provider)
        provider = vruiRendererInterface::the()->createPanelGeometryProvider(this);

    return provider->getHeight();
}

float coPanelGeometry::getDepth()
{
    if (!provider)
        provider = vruiRendererInterface::the()->createPanelGeometryProvider(this);

    return provider->getDepth();
}

const char *coPanelGeometry::getTextureName() const
{
    return textureName.c_str();
}

const char *coPanelGeometry::getClassName() const
{
    return "coPanelGeometry";
}

bool coPanelGeometry::isOfClassName(const char *classname) const
{
    // paranoia makes us mistrust the string library and check for NULL.
    if (classname && getClassName())
    {
        // check for identity
        if (!strcmp(classname, getClassName()))
        { // we are the one
            return true;
        }
        else
        { // we are not the wanted one. Branch up to parent class
            return false;
            // coPanelGeometry already is the root class. Else:
            // return co::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}
}
