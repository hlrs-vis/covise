/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_VEHICLE_H
#define OSC_VEHICLE_H

#include "oscExport.h"
#include "oscNameUserData.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "oscFileHeader.h"
#include "oscPerformance.h"
#include "oscFile.h"
#include "oscAxles.h"
#include "oscDimensionTypeA.h"
#include "oscLighting.h"
#include "oscEyepoints.h"
#include "oscMirrors.h"
#include "oscFeatures.h"


namespace OpenScenario {

class OPENSCENARIOEXPORT vehicleClassType: public oscEnumType
{
public:
    static vehicleClassType *instance(); 
private:
    vehicleClassType();
    static vehicleClassType *inst;
};

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscVehicle: public oscNameUserData
{
public:
    oscVehicle()
    {
        OSC_OBJECT_ADD_MEMBER(fileHeader, "oscFileHeader");
        OSC_ADD_MEMBER(manufacturer);
        OSC_ADD_MEMBER(model);
        OSC_ADD_MEMBER(color);
        OSC_ADD_MEMBER(licensePlate);
        OSC_OBJECT_ADD_MEMBER(performance, "oscPerformance");
        OSC_OBJECT_ADD_MEMBER(geometry, "oscFile");
        OSC_OBJECT_ADD_MEMBER(dimension, "oscDimensionTypeA");
        OSC_OBJECT_ADD_MEMBER(axles, "oscAxles");
        OSC_OBJECT_ADD_MEMBER(lighting, "oscLighting");
        OSC_OBJECT_ADD_MEMBER(eyepoints, "oscEyepoints");
        OSC_OBJECT_ADD_MEMBER(mirrors, "oscMirrors");
        OSC_OBJECT_ADD_MEMBER(features, "oscFeatures");
        OSC_ADD_MEMBER(category);

        category.enumType = vehicleClassType::instance();
    };

    oscFileHeaderMember fileHeader;
    oscString manufacturer;
    oscString model;
    oscString color;
    oscString licensePlate;
    oscPerformanceMember performance;
    oscFileMember geometry;
    oscDimensionTypeAMember dimension;
    oscAxlesMember axles;
    oscLightingArrayMember lighting;
    oscEyepointsArrayMember eyepoints;
    oscMirrorsArrayMember mirrors;
    oscFeaturesArrayMember features;
    oscEnum category;

    enum vehicleClass
    {
        car,
        van,
        truck,
        trailer,
        bus,
        motorbike,
        bicycle,
        train,
        tram,
    };
};

typedef oscObjectVariable<oscVehicle *> oscVehicleMember;

}

#endif //OSC_VEHICLE_H
