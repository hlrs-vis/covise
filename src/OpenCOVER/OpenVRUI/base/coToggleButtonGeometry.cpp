/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coToggleButtonGeometry.h>

#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/sginterface/vruiButtonProvider.h>

#include <string.h>

#include <OpenVRUI/util/vruiLog.h>

namespace vrui
{

coToggleButtonGeometry::coToggleButtonGeometry(const std::string &name)
    : coButtonGeometry(name)
{
}

coToggleButtonGeometry::~coToggleButtonGeometry()
{
}

float coToggleButtonGeometry::getWidth() const
{
    return buttonGeometryProvider->getWidth();
}

float coToggleButtonGeometry::getHeight() const
{
    return buttonGeometryProvider->getHeight();
}

void coToggleButtonGeometry::createGeometry()
{

    if (!buttonGeometryProvider)
    {
        //     VRUILOG("coToggleButtonGeometry::createGeometry info: creating buttonGeometryProvider for "
        //          << getClassName())
        buttonGeometryProvider = vruiRendererInterface::the()->createButtonProvider(this);
        buttonGeometryProvider->switchGeometry(coButtonGeometry::NORMAL);
        //    VRUILOG("coToggleButtonGeometry::createGeometry info: buttonGeometryProvider created")
    }
}

void coToggleButtonGeometry::resizeGeometry()
{
    createGeometry();
    buttonGeometryProvider->resizeGeometry();
}

const char *coToggleButtonGeometry::getClassName() const
{
    return "coToggleButtonGeometry";
}

bool coToggleButtonGeometry::isOfClassName(const char *classname) const
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
            return coButtonGeometry::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}
}
