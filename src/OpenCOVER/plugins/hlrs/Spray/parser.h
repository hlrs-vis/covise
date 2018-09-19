/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <config/CoviseConfig.h>
#include <config/coConfig.h>

#include <iostream>

#ifndef PARSER_H
#define PARSER_H

using namespace covise;

class parser
{
private:
    static parser* _instance;
    parser(){}
    parser(const parser&);
    ~parser(){}

    class parserGuard
    {
    public:
        ~parserGuard(){
            if(NULL != parser::_instance){
                delete parser::_instance;
                parser::_instance = NULL;
            }
        }

    };

    int emissionRate = 200;

    int numParticles = 10000;
    int numSamplings = 1000;
    int iterations = 4;
    float lowerPressureBound = 0.1;
    float upperPressureBound = 10;

    float densityOfFluid = 1.18;
    float densityOfParticle = 1000;
    float cwTurb = 0.4;
    float nu = 0.0000171;
    int reynoldsThreshold = 2230;
    int reynoldsLimit = 170000;
    float minimum = 0.0004;
    float deviation = 0.0001;
    float scaleFactor = 1000;
    float rendertime = 1;
    float alpha = 0.4;


    int colorThreshold = 100;
    int sphereRenderType = 1;

    std::string cwModelType = "STOKES";
    std::string samplingType = "square";

    std::string COVERpluginpath = "COVER.Plugin.Spray";


public:
    static parser* instance()
    {
        static parserGuard g;
        if(!_instance)
        {
            _instance = new parser;
        }
        return _instance;

    }

    void init()
    {
        emissionRate = coCoviseConfig::getInt(COVERpluginpath+".EmissionRate",200);

        numParticles = coCoviseConfig::getInt(COVERpluginpath+".NumParticles",10000);
        numSamplings = coCoviseConfig::getInt(COVERpluginpath+".NumSamplings",1000);
        iterations = coCoviseConfig::getInt(COVERpluginpath+".Iterations",4);
        lowerPressureBound = coCoviseConfig::getFloat(COVERpluginpath+".LowerPressureBound",0.1);
        upperPressureBound = coCoviseConfig::getFloat(COVERpluginpath+".UpperPressureBound",10.0);


        densityOfFluid = coCoviseConfig::getFloat(COVERpluginpath+".DensityOfFluid",1.27);
        densityOfParticle = coCoviseConfig::getFloat(COVERpluginpath+".DensityOfParticle",998.0);
        cwTurb = coCoviseConfig::getFloat(COVERpluginpath+".CwTurb",0.4);
        nu = coCoviseConfig::getFloat(COVERpluginpath+".Nu",0.0000171);
        reynoldsThreshold = coCoviseConfig::getInt(COVERpluginpath+".ReynoldsThreshold",2230);
        reynoldsLimit = coCoviseConfig::getInt(COVERpluginpath+".ReynoldsLimit",170000);
        minimum = coCoviseConfig::getFloat(COVERpluginpath+".Minimum",0.0004);
        deviation = coCoviseConfig::getFloat(COVERpluginpath+".Deviation",0.0001);
        scaleFactor = coCoviseConfig::getFloat(COVERpluginpath+".ScaleFactor",1000.0);
        rendertime = coCoviseConfig::getFloat(COVERpluginpath+".Rendertime",1.0);
        alpha = coCoviseConfig::getFloat(COVERpluginpath+".Alpha", 0.4);

        colorThreshold = coCoviseConfig::getInt(COVERpluginpath+".ColorThreshold", 100);
        sphereRenderType = coCoviseConfig::getInt(COVERpluginpath+".SphereRenderType", 1); /* 1 for ARB_POINT_SPRITES, 0 for CG_SHADER*/

        cwModelType = coCoviseConfig::getEntry("value",COVERpluginpath+".CwModelType","STOKES");
        samplingType = coCoviseConfig::getEntry("value",COVERpluginpath+".SamplingType","circle");

    }

    int getReqSamplings()
    {
        return numSamplings;
    }

    int getReqParticles()
    {
        return numParticles;
    }

    void setNumParticles(int newNumParticles)
    {
        numParticles = newNumParticles;
    }

    std::string getSamplingType()
    {
        return samplingType;
    }

    float getLowerPressureBound()
    {
        return lowerPressureBound;
    }

    float getUpperPressureBound()
    {
        return upperPressureBound;
    }

    int getEmissionRate()
    {
        return emissionRate;
    }

    void setEmissionRate(int newEmissionRate)
    {
        emissionRate = newEmissionRate;
    }

    std::string getCwModelType()
    {
        return cwModelType;
    }

    float getCwTurb()
    {
        return cwTurb;
    }

    float getDensityOfFluid()
    {
        return densityOfFluid;
    }

    float getNu()
    {
        return nu;
    }

    int getReynoldsThreshold()
    {
        return reynoldsThreshold;
    }

    int getColorThreshold()
    {
        return colorThreshold;
    }

    float getMinimum()
    {
        return minimum;
    }

    float getDeviation()
    {
        return deviation;
    }

    float getScaleFactor()
    {
        return scaleFactor;
    }

    void setScaleFactor(float newScaleFactor)
    {
        scaleFactor = newScaleFactor;
    }

    int getSphereRenderType()
    {
        return sphereRenderType;
    }

    float getDensityOfParticle()
    {
        return densityOfParticle;
    }

    float getRendertime()
    {
        return rendertime;
    }

    int getIterations()
    {
        return iterations;
    }

    int getReynoldsLimit()
    {
        return reynoldsLimit;
    }

    float getAlpha()
    {
        return alpha;
    }
};

#endif // PARSER_H
