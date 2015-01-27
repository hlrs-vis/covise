/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_COMPLEX_MODULES_H
#define CO_COMPLEX_MODULES_H

#include <util/coExport.h>
#include <api/coModule.h>
#include "coColors.h"
#include "coVectField.h"

namespace covise
{

class ALGEXPORT ComplexModules
{
public:
    static coDistributedObject *DataTexture(const string &color_name,
                                            const coDistributedObject *dataOut,
                                            const coDistributedObject *colorMap,
                                            bool texture,
                                            int repeat = 1, float *min = NULL, float *max = NULL);

    static coDistributedObject *DataTextureLineCropped(const string &color_name,
                                                       coDistributedObject *dataOut,
                                                       coDistributedObject *lines,
                                                       const coDistributedObject *colorMap,
                                                       bool texture,
                                                       int repeat = 1, int croppedLength = 0,
                                                       float min = FLT_MAX, float max = -FLT_MAX);

    static coDistributedObject *
    MakeArrows(const char *name, const coDistributedObject *geo, const coDistributedObject *data,
               const char *nameColor, coDistributedObject **colorSurf,
               coDistributedObject **colorLines,
               float factor,
               const coDistributedObject *colorMap,
               bool ColorMapAttrib,
               const ScalarContainer *SCont,
               float scale, int lineChoice, int numsectors, int project_lines, int vect_option);
    static coDistributedObject *
    MakeArrows(const char *name, const coDistributedObject *geo, const coDistributedObject *data,
               const char *nameColor, coDistributedObject **colorSurf,
               coDistributedObject **colorLines,
               float factor,
               const coDistributedObject *colorMap,
               bool ColorMapAttrib,
               const ScalarContainer *SCont,
               float scale, int lineChoice, int numsectors, float arrow_factor, float angle,
               int project_lines, int vect_option);
    static coDistributedObject *
    Spheres(const char *name, const coDistributedObject *points, float radius,
            const char *name_norm, coDistributedObject **normals);

    static coDistributedObject *
    Bars(const char *name, coDistributedObject *points, float radius, coDistributedObject *tangents, const char *name_norm, coDistributedObject **normals);

    static coDistributedObject *
    Compass(const char *name, const coDistributedObject *points, float radius, const coDistributedObject *tangents, const char *name_norm, coDistributedObject **normals, coDistributedObject **colors = NULL);

    static coDistributedObject *
    BarMagnets(const char *name, const coDistributedObject *points, float radius, const coDistributedObject *tangents, const char *name_norm, coDistributedObject **normals, coDistributedObject **colors = NULL);

    static coDistributedObject *
    Tubelines(const char *name, const coDistributedObject *lines, const coDistributedObject *tangents, float tubeSize, float radius, int trailLength, const char *headType, coDistributedObject **colors = NULL);

    static coDistributedObject *
    ThickStreamlines(const char *name, const coDistributedObject *lines, const coDistributedObject *tangents, float tubeSize, int trailLength);

    static coDistributedObject *
    croppedLinesSet(const coDistributedObject *lines, int croppedLength);
};
}
#endif
