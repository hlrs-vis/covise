/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PICK_SPHERE_INTERACTOR_H
#define PICK_SPHERE_INTERACTOR_H

#include "SphereData.h"

#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/StateSet>

#include <util/coRestraint.h>

#include <OpenVRUI/coTrackerButtonInteraction.h>

class PickSphereInteractor : public vrui::coTrackerButtonInteraction
{
public:
    void boxSelect(osg::Vec3, osg::Vec3);

    // Interactor to pick spheres
    PickSphereInteractor(coInteraction::InteractionType type, const char *name, vrui::coInteraction::InteractionPriority priority);

    ~PickSphereInteractor();

    // ongoing interaction
    void doInteraction();

    // stop the interaction
    void stopInteraction();

    // start the interaction
    void startInteraction();

    // return if selection was changed
    bool selectionChanged();

    // sets multipleSelect flag
    void enableMultipleSelect(bool);

    // returns the multipleSelect flag
    bool getMultipleSelect();

    // updates the selection string
    void updateSelection(const char *);

    // updates the temporary copy of the spheres
    // on them the intersection test takes place
    void updateSpheres(const std::vector<SphereData *> *allSpheres)
    {
        this->m_spheres = allSpheres;
    };

    // test for intersection between sphere and ray
    // returns -1 on no intersection
    // returns the (smallest) distance on intersection
    double LineSphereIntersect(osg::Vec3 center, float radius, osg::Vec3 handPos, osg::Vec3 handDirection);

    // returns the string that is created by the new selection
    std::string getSelectedParticleString();

    // returns the count of the selected particles
    int getSelectedParticleCount();

    // set animation state and according to it the state of the Animation checkbox in Animation menu
    void enableAnimation(bool state);

    void setSelectedWithBox(bool selected)
    {
        m_selectedWithBox = selected;
    }
    bool selectedWithBox()
    {
        return m_selectedWithBox;
    }

    const covise::coRestraint &getSelection() const
    {
        return m_evaluateIndices;
    }

private:
    float m_size;
    int m_lastIndex;
    double m_lastIndexTime;

    // needed for interaction
    osg::Vec3 m_initHandPos, m_initHandDirection;

    const std::vector<SphereData *> *m_spheres;
    bool m_animationWasRunning, m_multipleSelect, m_selectionChanged;
    bool m_selectedWithBox;

    void addSelectedSphere(int);
    void swap(float &m, float &n);
    covise::coRestraint m_evaluateIndices;

    int hitSphere();
    void highlightSphere(int index);

    osg::ref_ptr<osg::MatrixTransform> highlight;
};
#endif //PICK_SPHERE_INTERACTOR_H
