/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _MOLECULE_HANDLER_H
#define _MOLECULE_HANDLER_H

#include "StartMolecule.h"
#include "EndMolecule.h"
#include "Equation.h"

#include <osg/Geode>
#include <osg/Vec3>

#include <iostream>

#define NUM_START_MOLECULES 6

enum State
{
    STATE_IDLE,
    STATE_DEFAULT,
    STATE_REACTION_RUNNING,
    STATE_REACTION_DONE
};

class MoleculeHandler
{
public:
    MoleculeHandler();

    void preFrame();

    void clear();
    void createStartMolecules(int slot, std::string design);

    void performReaction();
    void resetReaction();

    State getState()
    {
        return state;
    };
    bool isCorrect(std::string target);

private:
    void updateEquation();

    Equation *equation;

    std::vector<StartMolecule *> startMolecules;
    std::vector<EndMolecule *> endMolecules;

    State state;
    float animationTime;
};

#endif
