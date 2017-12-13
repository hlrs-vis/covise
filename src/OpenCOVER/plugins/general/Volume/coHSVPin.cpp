/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coHSVPin.h"
#include <virvo/vvtoolshed.h>
#include <osg/Geode>
#include <osg/Drawable>

using namespace osg;

coHSVPin::coHSVPin(Group *root, float Height, float Width, vvTFColor *myPin)
    : coPin(root, Height, Width, myPin, false)
{
}

coHSVPin::~coHSVPin()
{
}

void coHSVPin::setColor(float h, float s, float v)
{

    float r, g, b;

    vvToolshed::HSBtoRGB(h, s, v, &r, &g, &b);
    (*color)[0].set(r, g, b, 1.0);
	color->dirty();
    for (unsigned int i = 0; i < geode->getNumDrawables(); i++)
    {
        geode->getDrawable(i)->dirtyDisplayList();
    }
}
