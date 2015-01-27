/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "HeightFieldPickBox.h"

// C++:
#include <assert.h>
#include <iostream>
#include <osg/Geode>

using namespace cui;
using namespace osg;
using namespace std;

HeightFieldPickBox::HeightFieldPickBox(Interaction *interaction,
                                       const osg::Vec4 &c1, const osg::Vec4 &c2, const osg::Vec4 &c3)
    : PickBox(interaction, osg::Vec3(0, 0, 0), osg::Vec3(1, 1, 1), c1, c2, c3)
{
    _geode = new osg::Geode();
    _scale->addChild(_geode);

    setShowWireframe(true);
    setMovable(true);
}

HeightFieldPickBox::~HeightFieldPickBox()
{
}

void HeightFieldPickBox::createHeightField(int width, int height, int chan, unsigned char *data)
{
    int x, y, num, index1, index2;
    float scaleX, scaleY, scaleZ;
    unsigned char tmp1, tmp2;
    osg::Geometry *heightField;
    osg::Vec3Array *vertices;
    osg::Vec4Array *color;

    std::cerr << "createHeightField" << endl;

    _geode->removeDrawables(0, _geode->getNumDrawables());

    heightField = new osg::Geometry();
    vertices = new osg::Vec3Array((2 * height - 2) * width);
    heightField->setVertexArray(vertices);
    color = new osg::Vec4Array(1);
    heightField->setColorArray(color);

    StateSet *state = heightField->getOrCreateStateSet();
    state->setMode(GL_LIGHTING, StateAttribute::OFF);

    switch (chan)
    {
    case 0:
        (*color)[0].set(1.0, 0.0, 0.0, 1.0);
        break;
    case 1:
        (*color)[0].set(0.0, 1.0, 0.0, 1.0);
        break;
    case 2:
        (*color)[0].set(0.0, 0.0, 1.0, 1.0);
        break;
    case 3:
        (*color)[0].set(0.0, 0.0, 0.0, 1.0);
        break;
    default:
        assert(0);
        break;
    }

    cerr << "Boxsize: " << getBoxSize()[0] << " x " << getBoxSize()[1] << " x " << getBoxSize()[2] << " x " << endl;

    scaleX = getBoxSize()[0] / (float)width;
    scaleY = getBoxSize()[1] / (float)height;
    scaleZ = getBoxSize()[2] / 256.0;

    std::cerr << "scaleX: " << scaleX << endl;
    std::cerr << "scaleY: " << scaleY << endl;
    std::cerr << "scaleZ: " << scaleZ << endl;

    for (y = 0; y < (height - 1); y++)
        for (x = 0; x < width; x++)
        {
            index1 = 2 * chan * (y * width + x);
            index2 = index1 + 1;

            tmp1 = data[y * width * chan + x];
            tmp2 = data[(y + 1) * width * chan + x];

            std::cerr << "X1: " << x *scaleX;
            std::cerr << " Y1: " << y *scaleY;
            std::cerr << " Z1: " << tmp1 *scaleZ << endl;

            std::cerr << "X2: " << x *scaleX;
            std::cerr << " Y2: " << (y + 1) * scaleY;
            std::cerr << " Z2: " << tmp2 *scaleZ << endl;

            (*vertices)[index1].set(x * scaleX, y * scaleY, tmp1 * scaleZ);
            (*vertices)[index2].set(x * scaleX, (y + 1) * scaleY, tmp2 * scaleZ);
        }

    for (y = 0; y < (height - 1); y++)
    {
        index1 = y * width;
        num = 2 * width;
        heightField->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUAD_STRIP, index1, num));
    }

    _geode->addDrawable(heightField);
}
