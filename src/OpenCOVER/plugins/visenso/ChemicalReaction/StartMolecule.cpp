/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "StartMolecule.h"

#include "ReactionArea.h"

StartMolecule::StartMolecule(Design *_design, osg::Vec3 initPos)
    : Molecule(_design)
    , coVR2DTransInteractor(osg::Vec3(0.0f, 0.0f, 0.0f), osg::Vec3(0.0f, 1.0f, 0.0f), 1.0f, coInteraction::ButtonA, "StartMolecule", _design->name.c_str(), coInteraction::Medium)
    , initPosition(initPos)
{
    scaleTransform->setMatrix(osg::Matrix()); // don't scale the interactor!
    buildFromDesign();
    setPosition(initPosition);
    this->enableIntersection();
}

StartMolecule::~StartMolecule()
{
    for (std::vector<Atom *>::iterator it = atoms.begin(); it < atoms.end(); ++it)
    {
        scaleTransform->removeChild((*it)->getNode());
        delete (*it);
    }
}

void StartMolecule::buildFromDesign()
{
    scaleTransform->removeChild(0, scaleTransform->getNumChildren());

    for (std::vector<AtomConfig>::iterator it = design->config.begin(); it < design->config.end(); ++it)
    {
        Atom *atom = new Atom((*it), this);
        atoms.push_back(atom);
        scaleTransform->addChild(atom->getNode());
    }
}

osg::Vec3 StartMolecule::getPosition()
{
    return coVR2DTransInteractor::getPosition();
}

void StartMolecule::setPosition(osg::Vec3 pos)
{
    updateTransform(pos, osg::Vec3(0.0f, 1.0f, 0.0f)); // interactor
}

void StartMolecule::stopInteraction()
{
    coVR2DTransInteractor::stopInteraction();
    if (!isInReactionArea(getPosition()))
    {
        setPosition(initPosition);
    }
}
