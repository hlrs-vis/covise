/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUSTOM_TRANSFORM_INTERACTOR_H
#define _CUSTOM_TRANSFORM_INTERACTOR_H

#include <OpenVRUI/coRelativeInputInteraction.h>
#include <cover/coVRIntersectionInteractor.h>
#include <net/tokenbuffer_serializer.h>
#include <cover/MatrixSerializer.h>
#include <net/tokenbuffer.h>

class PLUGIN_UTILEXPORT CustomTransformInteractor : public opencover::coVRIntersectionInteractor
{

private:
    bool m_rotateOnly = false;
    uint8_t m_translateOnlyAxis = 0;

    osg::Matrix m_startMatrix;
    osg::Matrix m_objectRotation;
    osg::Matrix m_handToTarget;
    osg::Matrix m_objectToTarget;

    osg::ref_ptr<osg::MatrixTransform> m_debugTarget;

    osg::Geometry *createLine(osg::Vec3 pos1, osg::Vec3 pos2, osg::Vec4 c);
    osg::ref_ptr<osg::MatrixTransform> axisTransform; ///< all the Geometry

    vrui::coRelativeInputInteraction interactionRelative;

protected:
    virtual void createGeometry() override;
    void updateSharedState() override;
    typedef vrb::SharedState<osg::Matrix> SharedMatrix;

public:
    CustomTransformInteractor(osg::Matrix m, float s, coInteraction::InteractionType type, const char *iconName, const char *interactorName, coInteraction::InteractionPriority priority);

    // delete scene graph
    virtual ~CustomTransformInteractor();

    // start the interaction (grab pointer, set selected hl, store dcsmat)
    virtual void startInteraction() override;
    virtual void doInteraction() override;
    virtual void stopInteraction() override;

    virtual void updateTransform(osg::Matrix m);

    void setShared(bool state) override;
};

#endif
