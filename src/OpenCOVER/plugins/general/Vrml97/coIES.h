/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COIES_H
#define COIES_H


#include <cover/coVRPluginSupport.h>

class VRML97COVEREXPORT coIES
{
public:
    
    coIES(std::string fileName);
    virtual ~coIES();
    
    int lampToLuminaireGeometry;
    int numAnglesAndMultiplyingFactors;
    int numLamps;
    float lumensPerLamp;
    float multiplier;
    int numVerticalAngles;
    int numHorizontalAngles;
    int photometricType;
    int unitsType;
    float width;
    float length;
    float height;
    float ballastFactor;
    float ballastLampPhotometricFactor;
    float inputWatts;
    std::vector<float> tiltAngles;
    std::vector<float> tiltFactors;
    std::vector<float> hAngles;
    std::vector<float> vAngles;
    std::vector<float> candela;
    
    osg::Image *getTexture();

protected:
    std::string fileName;
    FILE *fp;
    bool readFloats(std::vector<float> &arr,int numValues);
    bool readData();
};
#endif
