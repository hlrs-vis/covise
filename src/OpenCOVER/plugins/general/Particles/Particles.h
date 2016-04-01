/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// **************************************************************************
//
//			Source File
//
// * Description    : Read Particle data by RECOM
//
// * Class(es)      :
//
// * inherited from :
//
// * Author  : Uwe Wössner
//
// * History : started 25.3.2010
//
// **************************************************************************

#ifndef PARTICLES_H
#define PARTICLES_H
#include <cover/coVRPluginSupport.h>
#include <PluginUtil/coSphere.h>
#include <cover/coVRShader.h>
#include <util/coTypes.h>
using namespace covise;
using namespace opencover;

namespace osg
{
class Material;
class ShapeDrawable;
};

class TimeStepData
{
public:
    TimeStepData(int numParticles, unsigned int numFloats, unsigned int numInts);
    ~TimeStepData();
    float **values;
    int64_t **Ivalues;
    int numParticles;
    unsigned int numFloats;
    unsigned int numInts;
    osg::ref_ptr<osg::Geode> geode;
    osg::ref_ptr<coSphere> sphere;
    osg::ref_ptr<osg::Drawable> lines;
    osg::ref_ptr<osg::Vec4Array> colors;
};

class Particles
{
public:
    enum vartype
    {
        T_FLOAT = 0,
        T_INT = 1
    };
    enum modeType
    {
        M_PARTICLES = 0,
        M_LINES = 1
    };
    int getMode();

private:
    int numTimesteps;
    osg::Sequence *switchNode;
    FILE *fp;
    std::string shaderName;
    std::vector<std::string> variableNames;
    std::vector<enum vartype> variableTypes;
    std::vector<int> variableIndex;
    std::vector<float> variableMin;
    std::vector<float> variableMax;
    std::vector<float> variableScale;
    int interval;
    std::map<std::string, std::string> shaderParams;
    coVRShader *shader;
    bool doSwap;
    int ParticleMode;
    osg::Vec4f lineColor;

    int readFile(char *fn, int timestep);
    int readIMWFFile(char *fn, int timestep);
    int readIndentFile(char *fn, int timestep);
    int readBinaryTimestep(int timestep);
    int read64(uint64_t &val);
    int read32(int &val);
    TimeStepData **timesteps;
    unsigned int numInts;
    unsigned int numFloats;
    enum Format {
        IMWF,
        Particle,
        Indent,
    };
    Format format;
    unsigned int numHiddenVars;

public:
    unsigned int getNumValues()
    {
        return (numInts + numFloats);
    };
    std::vector<std::string> &getValueNames()
    {
        return variableNames;
    };

    std::string fileName;
    //constructor
    Particles(std::string filename, osg::Group *parent, int maxParticles);

    void setTimestep(int i);
    int getTimesteps()
    {
        return numTimesteps;
    };
    void summUpTimesteps(int increment);
    void updateColors(unsigned int valueNumber = 0, unsigned int aValueNumber = 0);
    void updateRadii(unsigned int valueNumber = 0);
    float getMin(int currentValue)
    {
        return variableMin[currentValue];
    };
    float getMax(int currentValue)
    {
        return variableMax[currentValue];
    };
    float getScale(int currentValue)
    {
        return variableScale[currentValue];
    };
    void colorizeAndResize(int timestep=-1);
    bool dump(std::string filename, int timestep, const float *xc, const float *yc, const float *zc) const;
    bool restore(std::string filename, int timestep);

    //destructor
    ~Particles();
};

#endif
