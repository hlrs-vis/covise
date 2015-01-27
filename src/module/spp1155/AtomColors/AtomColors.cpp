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
    // try to add local atommapping.xml to current coviseconfig
    m_mapConfig = new coConfigGroup("Module.AtomColors");
    m_mapConfig->addConfig(coConfigDefaultPaths::getDefaultLocalConfigFilePath() + "atommapping.xml", "local", true);
    coConfig::getInstance()->addConfig(m_mapConfig);

    coCoviseConfig::ScopeEntries mappingEntries = coCoviseConfig::getScopeEntries("Module.AtomMapping");
    if (mappingEntries.getValue() == NULL)
    {
        // add global atommapping.xml to current coviseconfig
        m_mapConfig->addConfig(coConfigDefaultPaths::getDefaultGlobalConfigFilePath() + "atommapping.xml", "global", true);
        coConfig::getInstance()->addConfig(m_mapConfig);
        // retrieve the values of atommapping.xml and build the GUI
    }
    coCoviseConfig::ScopeEntries mappingEntries2 = coCoviseConfig::getScopeEntries("Module.AtomMapping");

    const char **mapEntry = mappingEntries2.getValue();
    if (mapEntry == NULL)
        std::cout << "AtomMapping is NULL" << std::endl;
    int iNrCurrent = 0;
    float radius;
    char cAtomName[256], cAtomNameTmp[256];

    if (mapEntry == NULL || *mapEntry == NULL)
        std::cout << "The scope Module.AtomMapping is not available in your covise.config file!" << std::endl;

    const char **curEntry = mapEntry;
    while (curEntry && *curEntry)
    {
        AtomColor ac;
        int iScanResult = sscanf(curEntry[1], "%3s %s %f %f %f %f %f", ac.type, cAtomName, &radius, &ac.color[0], &ac.color[1], &ac.color[2], &ac.color[3]);
        if (iScanResult == 7)
        {
            m_rgb.push_back(ac);
            if (radius < 0.)
                radius = 0.;
            m_radius.push_back(radius);

            sprintf(cAtomNameTmp, "%03d  %s", iNrCurrent + 1, cAtomName);
#ifdef ATOMRADII
            coFloatParam *param = addFloatParam(ac.type, cAtomNameTmp);
            param->setValue(radius);
#else
            coColorParam *param = addColorParam(ac.type, cAtomNameTmp);
            param->setValue(ac.color[0], ac.color[1], ac.color[2], ac.color[3]);
#endif
            m_atom.push_back(param);
            //fprintf(stderr, "%d: name=%s (%s)\n", iNrCurrent+1, cAtomName, ac.type);
            if (iNrCurrent + 1 != atoi(curEntry[0]))
            {
                std::cout << "Your atommapping.xml is garbled" << std::endl;
            }
        }
        iNrCurrent++;
        curEntry += 2;
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
        if (text_size >= AtomColor::TypeLength)
        {
            fprintf(stderr, "incoming Text is longer than expected");
            continue;
        }
        for (int j = 0; j < m_rgb.size(); j++)
        {
            if (memcmp(m_rgb.at(j).type, type, text_size) == 0 && m_rgb.at(j).type[text_size] == '\0')
            {
#ifdef ATOMRADII
                result[i] = m_radius[j];
#else
                pResult->setFloatRGBA(i, m_rgb[j].color[0], m_rgb[j].color[1], m_rgb[j].color[2], m_rgb[j].color[3]);
#endif
                break;
            }
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
        int type = types[i] - 1;
        if (m_rgb.size() <= type || m_radius.size() <= type)
            type = 138; // Methanol

        if (m_rgb.size() > type && m_radius.size() > type)
        {
//fprintf(stderr, "%d: %d -> %s\n", i, type+1, m_atom[type]->getName());
#ifdef ATOMRADII
            result[i] = m_radius[type];
#else
            pResult->setFloatRGBA(i, m_rgb[type].color[0], m_rgb[type].color[1], m_rgb[type].color[2], m_rgb[type].color[3]);
#endif
        }
        else
            std::cout << "m_rgb is too small: no entry for atom type " << type << std::endl;
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
        if (m_rgb.size() > type && m_radius.size() > type)
        {
#ifdef ATOMRADII
            result[i] = m_radius[type];
#else
            pResult->setFloatRGBA(i, m_rgb[type].color[0], m_rgb[type].color[1], m_rgb[type].color[2], 1.0f);
#endif
        }
        else
            std::cout << "m_rgb is too small" << std::endl;
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
    int foundIndex = -1;
    for (int i = 0; i < m_atom.size(); ++i)
    {
        if (m_atom[i] && !strcmp(m_atom[i]->getName(), name))
        {
            foundIndex = i;
            break;
        }
    }

    if (foundIndex >= 0)
    {
        for (int j = 0; j < 4; j++)
#ifdef ATOMRADII
            m_radius[foundIndex] = m_atom[foundIndex]->getValue();
#else
            m_rgb[foundIndex].color[j] = m_atom[foundIndex]->getValue(j);
#endif
    }
}

MODULE_MAIN(Color, AtomColors)
