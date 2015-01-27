/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_VR_3D_TRANSROT_INTERACTOR_H
#define _CO_VR_3D_TRANSROT_INTERACTOR_H

#include <cover/coVRIntersectionInteractor.h>

namespace opencover
{

class PLUGIN_UTILEXPORT coVR3DTransRotInteractor : public coVRIntersectionInteractor
{

private:
    osg::Matrix _interMat_o, _oldHandMat, _oldHandMat_o;
    osg::Matrix _invOldHandMat, _invOldHandMat_o;
    osg::Matrix _oldInteractorXformMat, _oldInteractorXformMat_o;

    osg::Geometry *createLine(osg::Vec3 pos1, osg::Vec3 pos2, osg::Vec4 c);
    osg::ref_ptr<osg::MatrixTransform> axisTransform; ///< all the Geometry
    osg::ref_ptr<osg::MatrixTransform> xrTransform;
    osg::ref_ptr<osg::MatrixTransform> yrTransform;
    osg::ref_ptr<osg::MatrixTransform> zrTransform;
    osg::ref_ptr<osg::MatrixTransform> xlTransform;
    osg::ref_ptr<osg::MatrixTransform> ylTransform;
    osg::ref_ptr<osg::MatrixTransform> zlTransform;
    osg::ref_ptr<osg::Geode> sphereGeode;

protected:
    virtual void createGeometry();

public:
    coVR3DTransRotInteractor(osg::Matrix m, float s, coInteraction::InteractionType type, const char *iconName, const char *interactorName, coInteraction::InteractionPriority priority);

    // delete scene graph
    virtual ~coVR3DTransRotInteractor();

    // start the interaction (grab pointer, set selected hl, store dcsmat)
    virtual void startInteraction();
    virtual void doInteraction();

    virtual void updateTransform(osg::Matrix m);

    osg::Matrix getMatrix()
    {
        return _interMat_o;
    };
};
}
#endif
