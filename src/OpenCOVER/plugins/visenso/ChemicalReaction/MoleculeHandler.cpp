/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "MoleculeHandler.h"

#include "ReactionArea.h"
#include "DesignLib.h"
#include "Elements.h"

#include <cover/VRSceneGraph.h>
#include <cover/coVRPluginSupport.h>

#include <map>

#define MIN(a, b) (a < b ? a : b)

MoleculeHandler::MoleculeHandler()
    : state(STATE_IDLE)
{
    equation = new Equation();
}

void MoleculeHandler::preFrame()
{
    if (state == STATE_REACTION_RUNNING)
    {
        animationTime += cover->frameDuration() * 0.35f;
        if (animationTime >= 1.0f)
        {
            animationTime = 1.0f;
            state = STATE_REACTION_DONE;
        }
        for (std::vector<EndMolecule *>::iterator it = endMolecules.begin(); it < endMolecules.end(); ++it)
        {
            (*it)->animateAtoms(animationTime);
            if (state == STATE_REACTION_DONE)
            {
                if ((*it)->atoms.size() > 1)
                    (*it)->showName();
            }
        }
    }
    else if (state == STATE_DEFAULT)
    {
        for (std::vector<StartMolecule *>::iterator it = startMolecules.begin(); it < startMolecules.end(); ++it)
        {
            if ((*it)->wasStopped())
            {
                updateEquation();
            }
        }
    }
}

void MoleculeHandler::clear()
{
    state = STATE_IDLE;
    // delete endMolecules
    for (std::vector<EndMolecule *>::iterator it = endMolecules.begin(); it < endMolecules.end(); ++it)
        delete (*it);
    endMolecules.clear();
    // delete startMolecules
    for (std::vector<StartMolecule *>::iterator it = startMolecules.begin(); it < startMolecules.end(); ++it)
        delete (*it);
    startMolecules.clear();
    equation->setVisible(false);
}

void MoleculeHandler::createStartMolecules(int slot, std::string design)
{
    state = STATE_DEFAULT;
    Design *des = DesignLib::Instance()->getDesign(design);
    if (des != NULL)
    {
        osg::Vec3 slotPos = osg::Vec3(-11.0f, 0.0f, 5.5f - float(slot) * 3.5f);
        for (int i = 0; i < NUM_START_MOLECULES; ++i)
        {
            StartMolecule *mol = new StartMolecule(des, slotPos + osg::Vec3(0.0f, float(i) * 0.001f, 0.0f));
            startMolecules.push_back(mol);
            if (i == 0)
                mol->showName();
        }
    }
    equation->setVisible(true);
    updateEquation();
}

void MoleculeHandler::performReaction()
{

    ///////////////////////////////
    // 1: Clear endMolecules
    for (std::vector<EndMolecule *>::iterator mol_it = endMolecules.begin(); mol_it < endMolecules.end(); ++mol_it)
        delete (*mol_it);
    endMolecules.clear();

    ///////////////////////////////
    // 2: Collect activeMolecules
    std::vector<StartMolecule *> activeMolecules;
    for (std::vector<StartMolecule *>::iterator mol_it = startMolecules.begin(); mol_it < startMolecules.end(); ++mol_it)
    {
        if (isInReactionArea((*mol_it)->getPosition()))
        {
            activeMolecules.push_back(*mol_it);
        }
    }

    ///////////////////////////////
    // 3: Collect atoms of all activeMolecules
    std::vector<Atom *> availableAtoms[1 + ELEMENT_MAX];
    int availableAtomsCount(0);
    for (std::vector<StartMolecule *>::iterator mol_it = activeMolecules.begin(); mol_it < activeMolecules.end(); ++mol_it)
    {
        availableAtomsCount += (*mol_it)->atoms.size();
        for (std::vector<Atom *>::iterator atom_it = (*mol_it)->atoms.begin(); atom_it < (*mol_it)->atoms.end(); ++atom_it)
        {
            availableAtoms[(*atom_it)->getElement()].push_back((*atom_it));
        }
    }

    ///////////////////////////////
    // 4: Construct a list of endDesigns
    std::vector<Design *> endDesigns;
    while (availableAtomsCount > 0)
    {
        // search possible designs
        for (std::vector<Design *>::iterator design_it = DesignLib::Instance()->designList.begin(); design_it < DesignLib::Instance()->designList.end(); ++design_it)
        {

            std::vector<Atom *> usedAtoms;

            // try to use the design
            bool design_ok(true);
            for (std::vector<AtomConfig>::iterator conf_it = (*design_it)->config.begin(); conf_it < (*design_it)->config.end(); ++conf_it)
            {
                // check if we find an atom
                bool atom_ok(false);
                for (std::vector<Atom *>::iterator avail_it = availableAtoms[(*conf_it).element].begin(); avail_it < availableAtoms[(*conf_it).element].end(); ++avail_it)
                {
                    if (std::find(usedAtoms.begin(), usedAtoms.end(), (*avail_it)) == usedAtoms.end())
                    {
                        // unused atom found
                        usedAtoms.push_back(*avail_it);
                        atom_ok = true;
                        break;
                    }
                }
                // if we didnt find one, abort
                if (!atom_ok)
                {
                    design_ok = false;
                    break;
                }
            }

            // if ok, delete used availableAtoms from list and start over (break) ; test next design otherwise
            if (design_ok)
            {
                endDesigns.push_back(*design_it);
                availableAtomsCount -= (*design_it)->config.size();
                for (int elem = 1; elem <= ELEMENT_MAX; ++elem)
                {
                    for (int avail_id = 0; avail_id < availableAtoms[elem].size(); ++avail_id)
                    {
                        if (std::find(usedAtoms.begin(), usedAtoms.end(), availableAtoms[elem].at(avail_id)) < usedAtoms.end())
                        {
                            availableAtoms[elem].erase(availableAtoms[elem].begin() + avail_id);
                            --avail_id;
                        }
                    }
                }
                break;
            }
        }
    }

    ///////////////////////////////
    // 5: Construct "identical" endMolecules
    //      - endDesigns will be reduced
    //      - tmpMolecules contains unprocessed StartMolecules afterwards
    std::vector<StartMolecule *> tmpMolecules = activeMolecules;
    for (int design_id = 0; design_id < endDesigns.size(); ++design_id)
    {
        for (std::vector<StartMolecule *>::iterator mol_it = tmpMolecules.begin(); mol_it < tmpMolecules.end(); ++mol_it)
        {
            if ((*mol_it)->design == endDesigns.at(design_id))
            {
                EndMolecule *endMolecule = new EndMolecule((*mol_it)->design);
                for (std::vector<Atom *>::iterator atom_it = (*mol_it)->atoms.begin(); atom_it < (*mol_it)->atoms.end(); ++atom_it)
                {
                    (*atom_it)->endConfig = (*atom_it)->startConfig;
                    (*atom_it)->endMolecule = endMolecule;
                    endMolecule->atoms.push_back(*atom_it);
                }
                endMolecules.push_back(endMolecule);
                tmpMolecules.erase(mol_it);
                endDesigns.erase(endDesigns.begin() + design_id);
                --design_id;
                break;
            }
        }
    }

    if (endDesigns.empty())
        return; // no new molecules created -> no reaction

    ///////////////////////////////
    // 6: Collect atoms of all tmpMolecules (unprocessed molecules)
    for (int elem = 1; elem <= ELEMENT_MAX; ++elem)
        availableAtoms[elem].clear();
    for (std::vector<StartMolecule *>::iterator mol_it = tmpMolecules.begin(); mol_it < tmpMolecules.end(); ++mol_it)
    {
        for (std::vector<Atom *>::iterator atom_it = (*mol_it)->atoms.begin(); atom_it < (*mol_it)->atoms.end(); ++atom_it)
        {
            availableAtoms[(*atom_it)->getElement()].push_back((*atom_it));
        }
    }

    ///////////////////////////////
    // 7: Construct "new" endMolecules
    for (std::vector<Design *>::iterator design_it = endDesigns.begin(); design_it < endDesigns.end(); ++design_it)
    {
        EndMolecule *endMolecule = new EndMolecule(*design_it);
        for (std::vector<AtomConfig>::iterator conf_it = (*design_it)->config.begin(); conf_it < (*design_it)->config.end(); ++conf_it)
        {
            Atom *atom = availableAtoms[(*conf_it).element][0];
            atom->endConfig = (*conf_it);
            atom->endMolecule = endMolecule;
            endMolecule->atoms.push_back(atom);
            availableAtoms[(*conf_it).element].erase(availableAtoms[(*conf_it).element].begin());
        }
        endMolecules.push_back(endMolecule);
    }

    ///////////////////////////////
    // 8: Minimize travel-distance of the atoms (quadratic) and maximize distance between endMolecules (inverse quadratic)
    for (std::vector<EndMolecule *>::iterator mol_it = endMolecules.begin(); mol_it < endMolecules.end(); ++mol_it)
    {
        float best(9999.0f), bestX, bestZ;
        for (float x = AREA_X_MIN + 1.5f; x <= AREA_X_MAX - 1.5f; x += 0.5f)
        {
            for (float z = AREA_Z_MIN + 1.5f; z <= AREA_Z_MAX - 1.5f; z += 0.5f)
            {
                float dist(0.0f);
                for (std::vector<Atom *>::iterator it = (*mol_it)->atoms.begin(); it < (*mol_it)->atoms.end(); ++it)
                {
                    osg::Vec3 p1 = (*it)->startMolecule->getPosition() + (*it)->startConfig.position;
                    osg::Vec3 p2 = osg::Vec3(x, 0.0f, z) + (*it)->endConfig.position;
                    float tmp = (p2 - p1).length2();
                    dist += tmp / float((*mol_it)->atoms.size());
                }
                for (std::vector<EndMolecule *>::iterator it = endMolecules.begin(); it < mol_it; ++it)
                {
                    osg::Vec3 p1 = (*it)->getPosition();
                    osg::Vec3 p2 = osg::Vec3(x, 0.0f, z);
                    float tmp = AREA_X_MAX - (p2 - p1).length();
                    dist += tmp * tmp;
                }
                if (dist < best)
                {
                    best = dist;
                    bestX = x;
                    bestZ = z;
                }
            }
        }
        (*mol_it)->setPosition(osg::Vec3(bestX, 0.0f, bestZ));
    }

    // disable intersection
    for (std::vector<StartMolecule *>::iterator mol_it = startMolecules.begin(); mol_it < startMolecules.end(); ++mol_it)
        (*mol_it)->disableIntersection();

    // start animation
    animationTime = 0.0f;
    state = STATE_REACTION_RUNNING;

    updateEquation();
}

bool MoleculeHandler::isCorrect(std::string target)
{
    if (state == STATE_DEFAULT)
        return false;
    if (state == STATE_IDLE)
        return true;
    std::string result;
    for (std::vector<EndMolecule *>::iterator it = endMolecules.begin(); it < endMolecules.end(); ++it)
    {
        if (result.length() > 0)
        {
            result += " ";
        }
        result += (*it)->design->symbol;
    }
    return (result.compare(target) == 0);
}

void MoleculeHandler::resetReaction()
{
    state = STATE_DEFAULT;
    for (std::vector<StartMolecule *>::iterator mol_it = startMolecules.begin(); mol_it < startMolecules.end(); ++mol_it)
    {
        (*mol_it)->resetAtoms();
        (*mol_it)->enableIntersection();
    }
    for (std::vector<EndMolecule *>::iterator mol_it = endMolecules.begin(); mol_it < endMolecules.end(); ++mol_it)
        delete (*mol_it);
    endMolecules.clear();
    updateEquation();
}

void MoleculeHandler::updateEquation()
{
    // collect start designs and count
    std::map<Design *, int> designCount;
    for (std::vector<StartMolecule *>::iterator mol_it = startMolecules.begin(); mol_it < startMolecules.end(); ++mol_it)
    {
        if (isInReactionArea((*mol_it)->getPosition()))
        {
            std::map<Design *, int>::iterator des_it = designCount.find((*mol_it)->design);
            if (des_it == designCount.end())
            {
                designCount.insert(std::pair<Design *, int>((*mol_it)->design, 1));
            }
            else
            {
                (*des_it).second = (*des_it).second + 1;
            }
        }
    }

    std::string e("");
    bool first = true;
    for (std::map<Design *, int>::iterator des_it = designCount.begin(); des_it != designCount.end(); ++des_it)
    {
        if (!first)
        {
            e += "+ ";
        }
        first = false;
        if ((*des_it).second > 1)
        {
            std::stringstream out;
            out << (*des_it).second;
            e += out.str();
        }
        e += (*des_it).first->symbol + " ";
    }

    if (state != STATE_DEFAULT)
    {
        e += " >  "; // this results in two spaces in front of '>' and two spaces after

        // collect end designs and count
        designCount.clear();
        for (std::vector<EndMolecule *>::iterator mol_it = endMolecules.begin(); mol_it < endMolecules.end(); ++mol_it)
        {
            std::map<Design *, int>::iterator des_it = designCount.find((*mol_it)->design);
            if (des_it == designCount.end())
            {
                designCount.insert(std::pair<Design *, int>((*mol_it)->design, 1));
            }
            else
            {
                ++((*des_it).second);
            }
        }

        first = true;
        for (std::map<Design *, int>::iterator des_it = designCount.begin(); des_it != designCount.end(); ++des_it)
        {
            if (!first)
            {
                e += "+ ";
            }
            first = false;
            if ((*des_it).second > 1)
            {
                std::stringstream out;
                out << (*des_it).second;
                e += out.str();
            }
            e += (*des_it).first->symbol + " ";
        }
    }

    equation->setEquation(e);
}
