/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "AnnotationSensor.h"

AnnotationSensor::AnnotationSensor(Annotation *a, osg::Node *n)
    : coPickSensor(n)
{
    myAnnotation = a;
    setThreshold(50);
    //threshold = 50*50;
}

AnnotationSensor::~AnnotationSensor()
{
    if (active)
        disactivate();
}
/*
* Called when the mouse is over the Annotation. No Click necessary!
*
*/
void AnnotationSensor::activate()
{
    AnnotationPlugin::plugin->setCurrentAnnotation(myAnnotation);
    active = 1;
    //myAnnotation->setIcon(1);
}

/*
* Called when the mouse is not anymore over the annotation
*
*/
void AnnotationSensor::disactivate()
{
    AnnotationPlugin::plugin->setCurrentAnnotation(NULL);
    active = 0;
    //myAnnotation->setIcon(0);
}
