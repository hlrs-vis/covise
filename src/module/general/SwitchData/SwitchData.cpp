/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <do/coDoSet.h>
#include <do/coDoData.h>
#include <api/coFeedback.h>
#include <config/CoviseConfig.h>
#include "SwitchData.h"

static int ninput = 4;

/// Constructor
SwitchData::SwitchData(int argc, char *argv[])
    : coModule(argc, argv, "Output a selectable input object")
{
    ninput = coCoviseConfig::getInt("Module.SwitchData.NumberOfInputs", ninput);

    m_dataIn = new coInputPort *[ninput];
    for (int i = 0; i < ninput; ++i)
    {
        char portname[100];
        snprintf(portname, sizeof(portname), "DataIn%d", i);
        char description[100];
        snprintf(description, sizeof(description), "data alternative %d", i);
        m_dataIn[i] = addInputPort(portname, "coDistributedObject", description);
        m_dataIn[i]->setRequired(false);
    }
    m_switchIn = addInputPort("ControlIn0", "Int", "input selection");
    m_switchIn->setRequired(false);

    m_switchParam = addChoiceParam("switch", "input selection");
    std::vector<const char *> switchValues;
    switchValues.push_back("No output");
    switchValues.push_back("All inputs");
    for (int i = 0; i < ninput; ++i)
    {
        const int maxlen = 100;
        char *val = new char[maxlen];
        snprintf(val, maxlen, "Input no. %d", i + 1);
        char key[1000];
        snprintf(key, sizeof(key), "Module.SwitchData.Label%d", i);
        std::string entry = coCoviseConfig::getEntry(key);
        if (!entry.empty())
        {
            strncpy(val, entry.c_str(), maxlen);
            val[maxlen - 1] = '\0';
        }
        switchValues.push_back(val);
    }
    m_switchParam->setValue(ninput + 2, &switchValues[0], 2);
    for (int i = 0; i < ninput; ++i)
    {
        delete[] switchValues[i + 2];
    }
    switchValues.clear();

    m_dataOut = addOutputPort("DataOut0", "coDistributedObject", "data output");
    m_switchOut = addOutputPort("ControlOut0", "Int", "input selection");
}

/// Compute routine: load checkpoint file
int SwitchData::compute(const char *)
{
    int alternative = m_switchParam->getValue() - 2;
    const coDoInt *paramData = dynamic_cast<const coDoInt *>(m_switchIn->getCurrentObject());
    bool copyAttribute = false;
    if (paramData && paramData->getNumPoints() >= 1)
    {
        paramData->getPointValue(0, &alternative);
        copyAttribute = true;
    }
    if (alternative >= ninput)
        alternative = ninput - 1;

    std::vector<const coDistributedObject *> output;
    if (alternative == -1)
    {
        for (int i = 0; i < ninput; ++i)
        {
            const coDistributedObject *obj = m_dataIn[i]->getCurrentObject();
            if (obj)
            {
                obj->incRefCount();
                output.push_back(obj);
            }
        }
    }
    else if (alternative >= 0 && m_dataIn[alternative]->getCurrentObject())
    {
        const coDistributedObject *obj = m_dataIn[alternative]->getCurrentObject();
        obj->incRefCount();
        output.push_back(obj);
    }
    coDoSet *setOut = new coDoSet(m_dataOut->getObjName(), (int)output.size(), &output[0]);
    coDoInt *switchOut = new coDoInt(m_switchOut->getObjName(), 1);
    switchOut->setPointValue(0, alternative);

    if (copyAttribute)
    {
        setOut->copyAllAttributes(paramData);
        switchOut->copyAllAttributes(paramData);
    }
    else
    {
        coFeedback dataFeedback("SwitchData");
        dataFeedback.addPara(m_switchParam);
        dataFeedback.apply(setOut);

        coFeedback switchFeedback("SwitchData");
        switchFeedback.addPara(m_switchParam);
        switchFeedback.apply(switchOut);
    }

    m_dataOut->setCurrentObject(setOut);
    m_switchOut->setCurrentObject(switchOut);

    return CONTINUE_PIPELINE;
}

MODULE_MAIN(Filter, SwitchData)
