/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <api/coModule.h>
#include <do/coDoUnstructuredGrid.h>

using namespace covise;

class MoleculeRenderer : public coModule
{

private:
    virtual int compute(const char *port);
    virtual void param(const char *name, bool inMapLoading);
    coIntScalarParam *_p_NumberOfTimeSteps;
    coIntScalarParam *_p_renderedTimeSteps;
    float MinTemp, MaxTemp, ValTemp;
    double Temperature;

public:
    MoleculeRenderer(int argc, char *argv[]);
    virtual ~MoleculeRenderer();

    coInputPort *testtest;
    coOutputPort *renderedmolecules;
    coOutputPort *molecule_velocity;
    coOutputPort *molecule_kinetic_energy;
    coOutputPort *domain_pressure;
    coOutputPort *domain_temp;
    coOutputPort *boundingbox;
    coOutputPort *Tempport;
    coFloatSliderParam *_p_chTemp;
    coStringParam *_p_ipaddress;
    coStringParam *_p_tcpport;
};
