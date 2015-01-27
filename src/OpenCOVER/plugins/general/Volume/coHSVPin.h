/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-
#ifndef CO_HSV_PIN_H
#define CO_HSV_PIN_H
#include "coPin.h"

#include <osg/Group>

class coHSVPin : public coPin
{
public:
    coHSVPin(osg::Group *root, float Height, float Width, vvTFColor *myPin);
    virtual ~coHSVPin();

    void setColor(float h, float s, float v);
};
#endif
