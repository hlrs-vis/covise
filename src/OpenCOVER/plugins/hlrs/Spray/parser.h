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

        //std::cout << cwModelType <<" "<< sphereRenderType << " " << alpha << std::endl;


//Not needed anymore
//*******************************************************************************************
//        std::ifstream mystream(configFileName);
//        std::string line;
//        if(mystream.is_open())
//        {
//            while(std::getline(mystream,line))
//            {
//                if(line.empty())
//                {
//                    continue;
//                }
//                //empty lines
//                if(line.compare(0,1,"#") == 0)
//                {
//                    continue;                         //comment
//                }
//                if(line.compare("-") == 0)
//                {
//                    break;                            //imo better than EOF
//                }
//                std::stringstream ssLine(line);
//                std::getline(ssLine,line,'=');

//                try
//                {

//                    if(line.compare("particles ") == 0)
//                    {
//                        std::getline(ssLine,line,'\n');
//                        reqParticles = stof(line);
//                    }
//                    if(line.compare("samplings ") == 0)
//                    {
//                        std::getline(ssLine,line,'\n');
//                        reqSamplings = stof(line);
//                    }

//                    if(line.compare("samplingtype ") == 0)
//                    {
//                        std::getline(ssLine, line,'\n');
//                        if(line[0] = '0') line.erase(0,1);
//                        if(line.compare("square") || line.compare("circle"))
//                            samplingType = line;
//                    }

//                    if(line.compare("lower_pressure_bound ") == 0)
//                    {
//                        std::getline(ssLine,line,'\n');
//                        lowerPressureBound = stof(line);
//                    }

//                    if(line.compare("upper_pressure_bound ") == 0)
//                    {
//                        std::getline(ssLine,line,'\n');
//                        upperPressureBound = stof(line);
//                    }

//                    if(line.compare("cwModelType ") == 0)
//                    {
//                        std::getline(ssLine,line,'\n');
//                        if(line[0] = '0') line.erase(0,1);
//                        cwModelType = line;
//                    }

//                    if(line.compare("reynoldsThreshold ") == 0)
//                    {
//                        std::getline(ssLine,line,'\n');
//                        reynoldsThreshold = stof(line);
//                    }

//                    if(line.compare("nu ") == 0)
//                    {
//                        std::getline(ssLine,line,'\n');
//                        nu = stof(line);
//                    }

//                    if(line.compare("densityOfFluid ") == 0)
//                    {
//                        std::getline(ssLine,line,'\n');
//                        densityOfFluid = stof(line);
//                    }

//                    if(line.compare("densityOfParticle ") == 0)
//                    {
//                        std::getline(ssLine,line,'\n');
//                        densityOfParticle = stof(line);
//                    }

//                    if(line.compare("cwTurb ") == 0)
//                    {
//                        std::getline(ssLine,line,'\n');
//                        cwTurb = stof(line);
//                    }

//                    if(line.compare("newGenCreateCounter ") == 0)
//                    {
//                        std::getline(ssLine,line,'\n');
//                        newGenCreateCounter = stof(line);
//                    }

//                    if(line.compare("colorThreshold ") == 0)
//                    {
//                        std::getline(ssLine,line,'\n');
//                        colorThreshold = stof(line);
//                    }

//                    if(line.compare("minimum ") == 0)
//                    {
//                        std::getline(ssLine,line,'\n');
//                        minimum = stof(line);
//                    }

//                    if(line.compare("deviation ") == 0)
//                    {
//                        std::getline(ssLine,line,'\n');
//                        deviation = stof(line);
//                    }

//                    if(line.compare("scaleFactor ") == 0)
//                    {
//                        std::getline(ssLine,line,'\n');
//                        scaleFactor = stof(line);
//                    }

//                    if(line.compare("isAMD ") == 0)
//                    {
//                        std::getline(ssLine,line,'\n');
//                        isAMD = stof(line);
//                    }

//                    if(line.compare("rendertime ") == 0)
//                    {
//                        std::getline(ssLine,line,'\n');
//                        rendertime = stof(line);
//                    }

//                    if(line.compare("iterations ") == 0)
//                    {
//                        std::getline(ssLine,line,'\n');
//                        iterations = stof(line);
//                    }

//                    if(line.compare("reynoldsLimit ") == 0)
//                    {
//                        std::getline(ssLine,line,'\n');
//                        reynoldsLimit = stof(line);
//                    }

//                    if(line.compare("alpha ") == 0)
//                    {
//                        std::getline(ssLine,line,'\n');
//                        alpha = stof(line);
//                    }



//                }//try

//                catch(const std::invalid_argument& ia)
//                {
//                    std::cerr << "Invalid argument: " << ia.what() << std::endl;
//                }//catch
//            }
//            std::cout << "Config File successfully read!\n" << std::endl;
//        }
//        else
//        {
//            std::cout << "Could not open " << configFileName << "!" <<std::endl;
//            std::cout << "Standard values will be applied !" <<std::endl;
//        }
//        mystream.close();
    }

    int getReqSamplings()
    {
        return numSamplings;
    }

    int getReqParticles()
    {
        return numParticles;
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
