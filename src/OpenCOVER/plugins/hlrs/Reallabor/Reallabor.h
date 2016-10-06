/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef REALLABOR_H
#define REALLABOR_H

#include <cover/coVRPlugin.h>

class Reallabor : public opencover::coVRPlugin
{
public:
    Reallabor();
    ~Reallabor();

    // this will be called in PreFrame
    void preFrame();

    // this will be called if an object with feedback arrives
    void newInteractor(opencover::RenderObject *container, opencover::coInteractor *i);

    // this will be called if a COVISE object arrives
    void addObject(opencover::RenderObject *container,
                   opencover::RenderObject *obj, opencover::RenderObject *normObj,
                   opencover::RenderObject *colorObj, opencover::RenderObject *texObj,
                   osg::Group *root,
                   int numCol, int colorBinding, int colorPacking,
                   float *r, float *g, float *b, int *packedCol,
                   int numNormals, int normalBinding,
                   float *xn, float *yn, float *zn, float transparency);

    // this will be called if a COVISE object has to be removed
    void removeObject(const char *objName, bool replace);

private:
};
#endif
