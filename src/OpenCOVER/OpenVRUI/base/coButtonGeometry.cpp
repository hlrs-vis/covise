/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coButtonGeometry.h>

#include <OpenVRUI/sginterface/vruiNode.h>
#include <OpenVRUI/sginterface/vruiButtonProvider.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>

namespace vrui
{

coButtonGeometry::coButtonGeometry(const std::string &name)
    : texture(name)
    , buttonGeometryProvider(0)
{
}

coButtonGeometry::~coButtonGeometry()
{
    if (buttonGeometryProvider)
        delete buttonGeometryProvider;
}

void coButtonGeometry::switchGeometry(coButtonGeometry::ActiveGeometry active)
{

    createGeometry();
    buttonGeometryProvider->switchGeometry(active);
}

float coButtonGeometry::getWidth() const
{
    if (buttonGeometryProvider)
        return buttonGeometryProvider->getWidth();
    else
        return 0.0f;
}

float coButtonGeometry::getHeight() const
{
    if (buttonGeometryProvider)
        return buttonGeometryProvider->getHeight();
    else
        return 0.0f;
}

void coButtonGeometry::createGeometry()
{

    if (!buttonGeometryProvider)
    {
        //VRUILOG("coDefaultButtonGeometry::createGeometry info: creating buttonGeometryProvider for "
        //        << getClassName())
        buttonGeometryProvider = vruiRendererInterface::the()->createButtonProvider(this);
        buttonGeometryProvider->switchGeometry(coButtonGeometry::NORMAL);
    }
}

const char *coButtonGeometry::getClassName() const
{
    return "coButtonGeometry";
}

bool coButtonGeometry::isOfClassName(const char *classname) const
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
            // coButtonGeometry already is the root class. Else:
            // return co::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}

vruiTransformNode *coButtonGeometry::getDCS()
{

    createGeometry();

    return buttonGeometryProvider->getDCS();
}

void coButtonGeometry::resizeGeometry()
{
    createGeometry();
    buttonGeometryProvider->resizeGeometry();
}
}
