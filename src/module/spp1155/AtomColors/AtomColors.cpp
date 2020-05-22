/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                             (C)2004-2006 ZAIK/RRZK  ++
// ++ Description: Atom Properties module                                 ++
// ++                                                                     ++
// ++ Authors:                                                            ++
// ++                                                                     ++
// ++                Thomas van Reimersdahl, Sebastian Breuers            ++
// ++               Institute for Computer Science (Prof. Lang)           ++
// ++                        University of Cologne                        ++
// ++                         Robert-Koch-Str. 10                         ++
// ++                             50931 Koeln                             ++
// ++                                                                     ++
// ++ Date:  26.12.2004                                                   ++
// ++**********************************************************************/

#include "AtomColors.h"
#include <config/CoviseConfig.h>
#include <do/coDoText.h>
#include <alg/coChemicalElement.h>

#include <float.h>
#include <limits.h>

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

AtomColors::AtomColors(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Map colors to atoms")
{
    coAtomInfo *ai = coAtomInfo::instance();
    m_atom.resize(ai->all.size());
    for (int i = 0; i < ai->all.size(); i++)
    {
        if (!ai->all[i].valid)
            continue;

        std::string symbol = ai->all[i].symbol;
        if (!isalpha(symbol[0]))
            symbol = "_" + symbol;
        std::string description = ai->all[i].name;
        description += " (" + std::to_string(i) + ")";
#ifdef ATOMRADII
        coFloatParam *param = addFloatParam(symbol.c_str(), description.c_str());
        param->setValue(ai->all[i].radius);
#else
        coColorParam *param = addColorParam(symbol.c_str(), description.c_str());
        param->setValue(ai->all[i].color[0], ai->all[i].color[1], ai->all[i].color[2], ai->all[i].color[3]);
#endif
        m_atom[i] = param;
    }

    // Input Ports
    m_portInData = addInputPort("Data", "Int|Float|Text", "Atom type");
    m_portInPoints = addInputPort("Points", "Points", "Atom position");
    m_portInPoints->setRequired(false);

// Output ports
#ifdef ATOMRADII
    m_portOutColor = addOutputPort("radii", "Float", "Atom radii");
#else
    m_portOutColor = addOutputPort("colors", "RGBA", "Data as colors");
#endif
}

void AtomColors::preHandleObjects(coInputPort **ports)
{
    coInputPort *leader = m_portInData;
    if (m_portInPoints->isConnected())
        leader = m_portInPoints;
    for (int i = 0; ports[i]; ++i)
    {
        if (ports[i] == leader)
        {
            portLeader = i;
            return;
        }
    }
    portLeader = 0;
}

AtomColors::coDoResult *AtomColors::getOutputData(const coDoSet *inData)
{
    int iPoints;
    const coDistributedObject *const *types = inData->getAllElements(&iPoints);
    AtomColors::coDoResult *pResult = new AtomColors::coDoResult(m_portOutColor->getObjName(), iPoints);
    char *type;
    int text_size;

#ifdef ATOMRADII
    float *result;
    pResult->getAddress(&result);
#endif

    for (int i = 0; i < iPoints; i++)
    {
        const coDoText *text = dynamic_cast<const coDoText *>(types[i]);
        if (!text)
        {
            std::cerr << "no text data, ignored" << std::endl;
            continue;
        }
        text->getAddress(&type);
        text_size = text->getTextLength();
        std::string atomType = type;
        int idx = -1;
        auto elem = coAtomInfo::instance()->idMap.find(atomType);
        if (elem == coAtomInfo::instance()->idMap.end())
        {
            bool onlyDigit = true;
            for (const auto &c: atomType)
            {
                if (!isdigit(c)) {
                    onlyDigit = false;
                    break;
                }
            }
            if (onlyDigit)
            {
                idx = atoi(type);
            }
        }
        else
        {
            idx = elem->second;
        }

        if (idx >= 0 && idx < int(coAtomInfo::instance()->all.size()))
        {
#ifdef ATOMRADII
            result[i] = coAtomInfo::instance()->all[idx].radius;
#else
            pResult->setFloatRGBA(i, coAtomInfo::instance()->all[idx].color[0], coAtomInfo::instance()->all[idx].color[1], coAtomInfo::instance()->all[idx].color[2], coAtomInfo::instance()->all[idx].color[3]);
#endif
        }
        else
        {

#ifdef ATOMRADII
            result[i] = 1.0;
#else
            pResult->setFloatRGBA(i, 1,1,1,1);
#endif
        }
    }

    return pResult;
}

AtomColors::coDoResult *AtomColors::getOutputData(const coDoInt *inData)
{
    int numPoints = inData->getNumPoints();
    AtomColors::coDoResult *pResult = new AtomColors::coDoResult(m_portOutColor->getObjName(), numPoints);
    int *types;
    inData->getAddress(&types);

#ifdef ATOMRADII
    float *result;
    pResult->getAddress(&result);
#endif

    for (int i = 0; i < numPoints; i++)
    {
        int type = types[i];
        if (type >= coAtomInfo::instance()->all.size())
        {
#ifdef ATOMRADII
            result[i] = 1.0;
#else
            pResult->setFloatRGBA(i, 1, 1, 1, 1);
#endif
        }
        else
        {
#ifdef ATOMRADII
            result[i] = coAtomInfo::instance()->all[type].radius;
#else
            pResult->setFloatRGBA(i, coAtomInfo::instance()->all[type].color[0], coAtomInfo::instance()->all[type].color[1], coAtomInfo::instance()->all[type].color[2], coAtomInfo::instance()->all[type].color[3]);
#endif
        }
    }

    return pResult;
}

AtomColors::coDoResult *AtomColors::getOutputData(const coDoFloat *inData)
{
    int numPoints = inData->getNumPoints();

    AtomColors::coDoResult *pResult = new AtomColors::coDoResult(m_portOutColor->getObjName(), numPoints);

    float *fType;
    inData->getAddress(&fType);

#ifdef ATOMRADII
    float *result;
    pResult->getAddress(&result);
#endif

    for (int i = 0; i < numPoints; i++)
    {
        int type = static_cast<int>(fType[i]) - 1;
        if (type >= coAtomInfo::instance()->all.size())
        {
#ifdef ATOMRADII
            result[i] = 1.0;
#else
            pResult->setFloatRGBA(i, 1, 1, 1, 1);
#endif
        }
        else
        {
#ifdef ATOMRADII
            result[i] = coAtomInfo::instance()->all[type].radius;
#else
            pResult->setFloatRGBA(i, coAtomInfo::instance()->all[type].color[0], coAtomInfo::instance()->all[type].color[1], coAtomInfo::instance()->all[type].color[2], coAtomInfo::instance()->all[type].color[3]);
#endif
        }
    }

    return pResult;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  compute() is called once for every EXECUTE message
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int AtomColors::compute(const char *port)
{
    (void)port;

    if (const coDoInt *inData = dynamic_cast<const coDoInt *>(m_portInData->getCurrentObject()))
    {
        m_portOutColor->setCurrentObject(getOutputData(inData));
    }
    else if (const coDoFloat *inData = dynamic_cast<const coDoFloat *>(m_portInData->getCurrentObject()))
    {
        m_portOutColor->setCurrentObject(getOutputData(inData));
    }
    else if (const coDoSet *inData = dynamic_cast<const coDoSet *>(m_portInData->getCurrentObject()))
    {
        m_portOutColor->setCurrentObject(getOutputData(inData));
    }

    return SUCCESS;
}

void AtomColors::param(const char *name, bool /*inMapLoading*/)
{
    auto elem = coAtomInfo::instance()->idMap.find(name);
    if (elem != coAtomInfo::instance()->idMap.end())
    {
#ifdef ATOMRADII
        coAtomInfo::instance()->all[elem->second].radius = m_atom[elem->second]->getValue();
#else
        for (int j = 0; j < 4; j++)
            coAtomInfo::instance()->all[elem->second].color[j] = m_atom[elem->second]->getValue(j);
#endif
    }
}

MODULE_MAIN(Color, AtomColors)
