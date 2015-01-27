/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "readcomsol.h"

void ReadCOMSOL::createDialogElements()
{
    _fileBrowser = addFileBrowserParam("file_name_comsol", "COMSOL model file");
    _fileBrowser->setValue("c:\\", "*.mph");
    if (_paramOutVec != 0)
        delete[] _paramOutVec;
    _paramOutVec = new coChoiceParam *[_noPortsVec];
    for (unsigned char i = 0; i < _noPortsVec; i++)
    {
        const std::string name = getListName("Vector_data", i, false);
        const std::string desc = getListName("Value for vector output port", i, true);
        _paramOutVec[i] = addChoiceParam(name.c_str(), desc.c_str());
    }
    if (_paramOutScal != 0)
        delete[] _paramOutScal;
    _paramOutScal = new coChoiceParam *[_noPortsScal];
    for (unsigned char i = 0; i < _noPortsScal; i++)
    {
        const std::string name = getListName("Scalar_data", i, false);
        const std::string desc = getListName("Value for scalar output port", i, true);
        _paramOutScal[i] = addChoiceParam(name.c_str(), desc.c_str());
    }
}

void ReadCOMSOL::setParameterLists(void)
{
    _physics = _communicationComsol->readPhysics(TimeSteps::QualityTimeHarmonic_High);
    const unsigned char noPhysics = _physics->getNoPhysics();
    _paramDataVec.resize(0);
    _paramDataScal.resize(0);
    for (unsigned char i = 0; i < noPhysics; i++)
    {
        const unsigned char noPhysicalValues = _physics->getNoPhysicalValuesPhysic(i);
        for (unsigned char j = 0; j < noPhysicalValues; j++)
        {
            ApplicationModeData temp;
            temp.applicationMode = i;
            temp.compute = false;
            temp.name = _physics->getTextPhysicalValue(i, j);
            temp.physicalValue = j;
            if (_physics->isVector(i, j))
                _paramDataVec.push_back(temp);
            else
                _paramDataScal.push_back(temp);
        }
    }
    const char **tempArray = new const char *[_paramDataVec.size() + 1];
    tempArray[0] = "---";
    for (unsigned short k = 0; k < _paramDataVec.size(); k++)
        tempArray[k + 1] = _paramDataVec[k].name.c_str();
    for (unsigned char i = 0; i < _noPortsVec; i++)
        _paramOutVec[i]->setValue(_paramDataVec.size() + 1, tempArray, 0);
    delete[] tempArray;
    tempArray = new const char *[_paramDataScal.size() + 1];
    tempArray[0] = "---";
    for (unsigned short k = 0; k < _paramDataScal.size(); k++)
        tempArray[k + 1] = _paramDataScal[k].name.c_str();
    for (unsigned char i = 0; i < _noPortsScal; i++)
        _paramOutScal[i]->setValue(_paramDataScal.size() + 1, tempArray, 0);
    delete[] tempArray;
}

std::vector<std::vector<bool> > ReadCOMSOL::getExportList() const
{
    std::vector<std::vector<bool> > retVal;
    retVal.resize(_physics->getNoPhysics());
    for (unsigned char i = 0; i < _physics->getNoPhysics(); i++)
    {
        retVal[i].resize(_physics->getNoPhysicalValuesPhysic(i));
        for (unsigned char j = 0; j < _physics->getNoPhysicalValuesPhysic(i); j++)
            retVal[i][j] = false;
    }
    for (unsigned char k = 0; k < _noPortsVec; k++)
    {
        const unsigned short choice = _paramOutVec[k]->getValue();
        if (choice != 0)
            retVal[_paramDataVec[choice - 1].applicationMode][_paramDataVec[choice - 1].physicalValue] = true;
    }
    for (unsigned char k = 0; k < _noPortsScal; k++)
    {
        const unsigned short choice = _paramOutScal[k]->getValue();
        if (choice != 0)
            retVal[_paramDataScal[choice - 1].applicationMode][_paramDataScal[choice - 1].physicalValue] = true;
    }
    return retVal;
}

void ReadCOMSOL::setComputationPhysicalValues()
{
    const unsigned char noPhysics = _physics->getNoPhysics();
    for (unsigned char i = 0; i < noPhysics; i++)
    {
        const unsigned char noValues = _physics->getNoPhysicalValuesPhysic(i);
        for (unsigned char j = 0; j < noValues; j++)
            _physics->setComputePhysicalValue(i, j, false);
    }
    for (unsigned char i = 0; i < _noPortsVec; i++)
    {
        const unsigned char choice = _paramOutVec[i]->getValue();
        if (choice != 0)
        {
            const unsigned char noPhysic = _paramDataVec[choice - 1].applicationMode;
            const unsigned char noValue = _paramDataVec[choice - 1].physicalValue;
            _physics->setComputePhysicalValue(noPhysic, noValue, true);
        }
    }
    for (unsigned char i = 0; i < _noPortsScal; i++)
    {
        const unsigned char choice = _paramOutScal[i]->getValue();
        if (choice != 0)
        {
            const unsigned char noPhysic = _paramDataScal[choice - 1].applicationMode;
            const unsigned char noValue = _paramDataScal[choice - 1].physicalValue;
            _physics->setComputePhysicalValue(noPhysic, noValue, true);
        }
    }
}
