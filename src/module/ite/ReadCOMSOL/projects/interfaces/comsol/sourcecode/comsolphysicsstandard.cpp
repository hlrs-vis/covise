/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "comsolphysicsstandard.h"
#include <sstream>
#include <iostream>

ComsolPhysicsStandard::ComsolPhysicsStandard()
{
    _physics.resize(0);
    _timeSteps = 0;
}

ComsolPhysicsStandard::~ComsolPhysicsStandard()
{
}

void ComsolPhysicsStandard::addDataPhysics(const std::string type, const std::string tag)
{
    const unsigned int index = _physics.size();
    _physics.resize(index + 1);
    DataPhysics &physic = _physics[index];
    physic.type = type;
    physic.tag = tag;
    physic.physicalValues.resize(0);
    if (type == "Electrostatics")
        setPhysicalValuesElectrostatics(physic);
    else
    {
        if (type == "ElectromagneticWaves")
            setPhysicalValuesElectromagneticWaves(physic);
        else
        {
            if (type == "MovingMesh")
                setPhysicalValuesMovingMesh(physic);
            else
            {
                if (type == "InductionCurrents")
                    setPhysicalValuesMagnetostatics(physic, true);
            }
        }
    }
}

void ComsolPhysicsStandard::setTimeSteps(TimeSteps *timeSteps)
{
    _timeSteps = timeSteps;
}

unsigned int ComsolPhysicsStandard::getNoPhysics(void) const
{
    return (unsigned int)_physics.size();
}

unsigned int ComsolPhysicsStandard::getNoPhysicalValuesPhysic(const unsigned int no) const
{
    unsigned int retVal = 0;
    if (no < _physics.size())
        retVal = (unsigned int)_physics[no].physicalValues.size();
    return retVal;
}

std::string ComsolPhysicsStandard::getTextPhysicalValue(const unsigned int noPhysic, const unsigned int noPhysicalValue) const
{
    std::string retVal = "";
    if (noPhysic < _physics.size())
    {
        if (noPhysicalValue < _physics[noPhysic].physicalValues.size())
        {
            std::ostringstream buffer;
            buffer << _physics[noPhysic].physicalValues[noPhysicalValue].comsolName;
            buffer << " (";
            buffer << _physics[noPhysic].tag;
            buffer << ") - ";
            buffer << _physics[noPhysic].physicalValues[noPhysicalValue].description;
            retVal = buffer.str();
        }
    }
    return retVal;
}

bool ComsolPhysicsStandard::isVector(const unsigned int noPhysic, const unsigned int noPhysicalValue) const
{
    bool retVal = false;
    if (noPhysic < _physics.size())
    {
        if (noPhysicalValue < _physics[noPhysic].physicalValues.size())
            retVal = _physics[noPhysic].physicalValues[noPhysicalValue].vector;
    }
    return retVal;
}

std::string ComsolPhysicsStandard::getNameValue(const unsigned int noPhysic, const unsigned int noPhysicalValue) const
{
    std::string retVal = "";
    if (noPhysic < _physics.size())
    {
        if (noPhysicalValue < _physics[noPhysic].physicalValues.size())
            retVal = _physics[noPhysic].physicalValues[noPhysicalValue].comsolName;
    }
    return retVal;
}

std::string ComsolPhysicsStandard::getTag(const unsigned int noPhysic) const
{
    std::string retVal = "";
    if (noPhysic < _physics.size())
        retVal = _physics[noPhysic].tag;
    return retVal;
}

bool ComsolPhysicsStandard::getAverage(const unsigned int noPhysic, const unsigned int noPhysicalValue) const
{
    bool retVal = false;
    if (noPhysic < _physics.size())
    {
        if (noPhysicalValue < _physics[noPhysic].physicalValues.size())
            retVal = _physics[noPhysic].physicalValues[noPhysicalValue].average;
    }
    return retVal;
}

TimeSteps *ComsolPhysicsStandard::getTimeSteps(const bool firstTimeStepOnly) const
{
    if (_timeSteps != 0)
        _timeSteps->setFirstTimeStepOnly(firstTimeStepOnly);
    return _timeSteps;
}

bool ComsolPhysicsStandard::isDof(const unsigned int noPhysic, const unsigned int noPhysicalValue) const
{
    bool retVal = false;
    if (noPhysic < _physics.size())
    {
        if (noPhysicalValue < _physics[noPhysic].physicalValues.size())
            retVal = _physics[noPhysic].physicalValues[noPhysicalValue].dof;
    }
    return retVal;
}

bool ComsolPhysicsStandard::isMovingMesh() const
{
    bool retVal = false;
    for (unsigned int i = 0; i < _physics.size(); i++)
    {
        if (_physics[i].type == "MovingMesh")
            retVal = true;
    }
    return retVal;
}

bool ComsolPhysicsStandard::computePhysicalValue(const unsigned int noPhysic, const unsigned int noPhysicalValue) const
{
    bool retVal = false;
    if (noPhysic < _physics.size())
    {
        if (noPhysicalValue < _physics[noPhysic].physicalValues.size())
            retVal = _physics[noPhysic].physicalValues[noPhysicalValue].compute;
    }
    return retVal;
}

void ComsolPhysicsStandard::setPhysicalValuesElectrostatics(DataPhysics &physic)
{
    DataPhysicalValue temp;
    temp.comsolName = "V";
    temp.description = "scalar electric potential";
    temp.vector = false;
    temp.average = false;
    temp.dof = true;
    temp.compute = false;
    physic.physicalValues.push_back(temp);
    temp.comsolName = "E";
    temp.description = "electric field strength";
    temp.vector = true;
    temp.average = false;
    temp.dof = false;
    temp.compute = false;
    physic.physicalValues.push_back(temp);
    temp.comsolName = "D";
    temp.description = "electric flux density";
    temp.vector = true;
    temp.average = false;
    temp.dof = false;
    temp.compute = false;
    physic.physicalValues.push_back(temp);
    temp.comsolName = "We";
    temp.description = "electric energy density";
    temp.vector = false;
    temp.average = false;
    temp.dof = false;
    temp.compute = false;
    physic.physicalValues.push_back(temp);
    temp.comsolName = "P";
    temp.description = "polarization";
    temp.vector = true;
    temp.average = false;
    temp.dof = false;
    temp.compute = false;
    physic.physicalValues.push_back(temp);
}

void ComsolPhysicsStandard::setPhysicalValuesMagnetostatics(DataPhysics &physic, const bool currents)
{
    DataPhysicalValue temp;
    temp.comsolName = "H";
    temp.description = "magnetic field strenght";
    temp.vector = true;
    temp.average = false;
    temp.dof = false;
    temp.compute = false;
    physic.physicalValues.push_back(temp);
    temp.comsolName = "B";
    temp.description = "magnetic flux density";
    temp.vector = true;
    temp.average = false;
    temp.dof = false;
    temp.compute = false;
    physic.physicalValues.push_back(temp);
    if (currents)
    {
        temp.comsolName = "V";
        temp.description = "electric scalar potential";
        temp.vector = false;
        temp.average = false;
        temp.dof = true;
        temp.compute = false;
        physic.physicalValues.push_back(temp);
        temp.comsolName = "A";
        temp.description = "magnetic vector potential";
        temp.vector = true;
        temp.average = false;
        temp.dof = true;
        temp.compute = false;
        physic.physicalValues.push_back(temp);
        temp.comsolName = "J";
        temp.description = "electric current density";
        temp.vector = true;
        temp.average = false;
        temp.dof = false;
        temp.compute = false;
        physic.physicalValues.push_back(temp);
    }
    else
    {
        temp.comsolName = "Vm";
        temp.description = "magnetic scalar potential";
        temp.vector = false;
        temp.average = false;
        temp.dof = true;
        temp.compute = false;
        physic.physicalValues.push_back(temp);
    }
}

void ComsolPhysicsStandard::setPhysicalValuesSteadyCurrents(DataPhysics &physic)
{
    DataPhysicalValue temp;
    temp.comsolName = "V";
    temp.description = "scalar electric potential";
    temp.vector = false;
    temp.average = false;
    temp.dof = true;
    temp.compute = false;
    physic.physicalValues.push_back(temp);
    temp.comsolName = "J";
    temp.description = "electric current density";
    temp.vector = true;
    temp.average = false;
    temp.dof = false;
    temp.compute = false;
    physic.physicalValues.push_back(temp);
    temp.comsolName = "Qrh";
    temp.description = "resistive losses";
    temp.vector = false;
    temp.average = false;
    temp.dof = false;
    temp.compute = false;
    physic.physicalValues.push_back(temp);
}

void ComsolPhysicsStandard::setPhysicalValuesElectromagneticWaves(DataPhysics &physic)
{
    DataPhysicalValue temp;
    temp.comsolName = "E";
    temp.description = "electric field strength";
    temp.vector = true;
    temp.average = false;
    temp.dof = true;
    temp.compute = false;
    physic.physicalValues.push_back(temp);
    temp.comsolName = "D";
    temp.description = "electric displacement";
    temp.vector = true;
    temp.average = false;
    temp.dof = false;
    temp.compute = false;
    physic.physicalValues.push_back(temp);
    temp.comsolName = "H";
    temp.description = "magnetic field strength";
    temp.vector = true;
    temp.average = false;
    temp.dof = true;
    temp.compute = false;
    physic.physicalValues.push_back(temp);
    temp.comsolName = "B";
    temp.description = "magnetic flux density";
    temp.vector = true;
    temp.average = false;
    temp.dof = false;
    temp.compute = false;
    physic.physicalValues.push_back(temp);
    temp.comsolName = "Po";
    temp.description = "Poynting vector";
    temp.vector = true;
    temp.average = true;
    temp.dof = false;
    temp.compute = false;
    physic.physicalValues.push_back(temp);
}

void ComsolPhysicsStandard::setPhysicalValuesNavierStokes(DataPhysics &physic)
{
    DataPhysicalValue temp;
    temp.comsolName = "U";
    temp.description = "velocity field";
    temp.vector = true;
    temp.average = false;
    temp.dof = true;
    temp.compute = false;
    physic.physicalValues.push_back(temp);
    temp.comsolName = "p";
    temp.description = "pressure";
    temp.vector = false;
    temp.average = false;
    temp.dof = true;
    temp.compute = false;
    physic.physicalValues.push_back(temp);
}

void ComsolPhysicsStandard::setPhysicalValuesHeatTransfer(DataPhysics &physic)
{
    DataPhysicalValue temp;
    temp.comsolName = "T";
    temp.description = "temperature";
    temp.vector = false;
    temp.average = false;
    temp.dof = true;
    temp.compute = false;
    physic.physicalValues.push_back(temp);
    temp.comsolName = "tflux_";
    temp.description = "total heat flux";
    temp.vector = true;
    temp.average = false;
    temp.dof = false;
    temp.compute = false;
    physic.physicalValues.push_back(temp);
    temp.comsolName = "dflux_";
    temp.description = "conductive heat flux";
    temp.vector = true;
    temp.average = false;
    temp.dof = false;
    temp.compute = false;
    physic.physicalValues.push_back(temp);
    temp.comsolName = "cflux_";
    temp.description = "convective heat flux";
    temp.vector = true;
    temp.average = false;
    temp.dof = false;
    temp.compute = false;
    physic.physicalValues.push_back(temp);
}

void ComsolPhysicsStandard::setPhysicalValuesMovingMesh(DataPhysics &physic)
{
    DataPhysicalValue temp;
    temp.comsolName = "d";
    temp.description = "displacement";
    temp.vector = true;
    temp.average = false;
    temp.dof = false;
    temp.compute = false;
    physic.physicalValues.push_back(temp);
}

void ComsolPhysicsStandard::setComputePhysicalValue(const unsigned int noPhysic, const unsigned int noPhysicalValue, const bool value)
{
    if (noPhysic < _physics.size())
    {
        if (noPhysicalValue < _physics[noPhysic].physicalValues.size())
            _physics[noPhysic].physicalValues[noPhysicalValue].compute = value;
    }
}
