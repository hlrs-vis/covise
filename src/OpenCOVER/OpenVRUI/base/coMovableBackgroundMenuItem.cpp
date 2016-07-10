/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <coMovableBackgroundMenuItem.h>

#include <math.h>
#include <util/unixcompat.h>

namespace vrui
{

coMovableBackgroundMenuItem::coMovableBackgroundMenuItem(const char *name, float aspect, float size)
    : coMenuItem(name)
{
    //fprintf(stderr, "coMovableBackgroundMenuItem::coMovableBackgroundMenuItem(%s, %f, %f)\n", name, aspect, size);

    background = new coTexturedBackground(name, name, name);
    background->setMinWidth(60.0);
    background->setMinHeight(60.0);
    float s;
    if (size == -1)
        s = 500.0;
    else
        s = size;
    if (aspect == 0 || std::isnan(aspect))
        aspect = 1;

    aspect_ = aspect;
    vsize_ = size;

    background->setSize(s * aspect, s, 1);
    background->setWidth(s * aspect);
    background->setHeight(s);
    background->setTexSize(s * aspect, s);
    background->setRepeat(false);
}

coMovableBackgroundMenuItem::coMovableBackgroundMenuItem(const char *name, uint *normalImage, int comp, int ns, int nt, int nr, float aspect, float size)
    : coMenuItem(name)
{
    //fprintf(stderr, "coMovableBackgroundMenuItem::coMovableBackgroundMenuItem 2(%s, %f, %f)\n", name, aspect, size);

    (void)normalImage;
    (void)comp;
    (void)ns;
    (void)nt;
    (void)nr;

    background = new coTexturedBackground(name, name, name); //(normalImage, normalImage, normalImage, comp, ns, nt, nr, aspect, size);
    background->setMinWidth(60.0);
    background->setMinHeight(60.0);
    float s;
    if (size == -1)
        s = 500.0;
    else
        s = size;
    if (aspect == 0 || std::isnan(aspect))
        aspect = 1;

    aspect_ = aspect;
    vsize_ = size;

    background->setSize(s * aspect, s, 1);
    background->setWidth(s * aspect);
    background->setHeight(s);
    background->setTexSize(s * aspect, s);
    background->setRepeat(false);
}

coMovableBackgroundMenuItem::~coMovableBackgroundMenuItem()
{
}

coUIElement *coMovableBackgroundMenuItem::getUIElement()
{
    return background;
}

const char *coMovableBackgroundMenuItem::getClassName() const
{
    return "coMovableBackgroundMenuItem";
}

bool coMovableBackgroundMenuItem::isOfClassName(const char *classname) const
{
    // paranoia makes us mistrust the string library and check for NULL.
    if (classname && getClassName())
    { // check for identity
        if (!strcmp(classname, getClassName()))
        { // we are the one
            return true;
        }
        else
        { // we are not the wanted one. Branch up to parent class
            return coMenuItem::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}

// return scalefactor of movableBackground;
float coMovableBackgroundMenuItem::getScale()
{
    return background->getScale();
}

float coMovableBackgroundMenuItem::getAspect()
{
    return aspect_;
}

float coMovableBackgroundMenuItem::getVSize()
{
    return vsize_;
}

// set scalefactor of movable background
void coMovableBackgroundMenuItem::setScale(float s)
{
    //fprintf(stderr, "coMovableBackgroundMenuItem::setScale %f\n", s);
    background->setScale(s);
}

// reset scale and position of movable background
void coMovableBackgroundMenuItem::reset()
{
    background->setScale(1.0);
}

// set size
void coMovableBackgroundMenuItem::setSize(float hsize, float vsize)
{
    vsize_ = vsize;
    aspect_ = hsize / vsize;
    background->setSize(hsize, vsize, 1);
}
}
