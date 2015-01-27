/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CHARGED_OBJECT_HANDLER_H
#define _CHARGED_OBJECT_HANDLER_H

#include "ChargedPoint.h"
#include "ChargedPlate.h"

#include <osg/Camera>

class ChargedObjectHandler
{
public:
    static ChargedObjectHandler *Instance();

    void preFrame();
    void guiToRenderMsg(const char *msg);

    ChargedPoint *addPoint();
    ChargedPlate *addPlate();
    void removeAllObjects();

    osg::Vec3 getFieldAt(osg::Vec3 point);
    float getPotentialAt(osg::Vec3 point);
    osg::Vec4 getFieldAndPotentialAt(osg::Vec3 point);

    int getActiveObjectsCount(unsigned int types);
    bool fieldIsValid();
    void dirtyField();

    void objectsActiveStateChanged();

    float *getFieldU()
    {
        return field_u;
    };
    float *getFieldV()
    {
        return field_v;
    };
    float *getFieldW()
    {
        return field_w;
    };
    float *getFieldPotential()
    {
        return field_potential;
    };

    float getGridMin()
    {
        return grid_min;
    };
    float getGridMax()
    {
        return grid_max;
    };
    int getGridSteps()
    {
        return grid_steps;
    };

    void setRadiusOfPlates(float radius);

private:
    ChargedObjectHandler(); // constructor
    static ChargedObjectHandler *instance_;

    void calculateField();
    int fieldIsDirty;

    std::vector<ChargedObject *> chargedObjects; // all our objects
    ChargedPlate *chargedPlates[2]; // direct access to the plates

    // calculated in calculateField and used in get*At (to avoid too much computation during get*At)
    std::vector<ChargedObject *> activeObjects; // holds a pointer to the active objects
    bool twoPlatesActive;

    float *field_u, *field_v, *field_w, *field_potential;
    float grid_min, grid_max;
    int grid_steps;

    osg::ref_ptr<osg::Camera> textCamera;
};

#endif
