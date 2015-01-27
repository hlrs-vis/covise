/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ANNOTATIONSENSOR_H
#define _ANNOTATIONSENSOR_H

#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

#include "AnnotationPlugin.h"
#include "Annotation.h"

class Annotation;

class AnnotationSensor : public coPickSensor
{
private:
    Annotation *myAnnotation;

public:
    AnnotationSensor(Annotation *a, osg::Node *n);
    ~AnnotationSensor();
    // this method is called if intersection just started
    // and should be overloaded
    virtual void activate();

    // should be overloaded, is called if intersection finishes
    virtual void disactivate();
};

#endif //_ANNOTATIONSENSOR_H
