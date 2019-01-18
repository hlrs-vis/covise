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
// * Author  : Uwe Woessner
//
// * History : started 25.3.2010
//
// **************************************************************************

#include <cover/coVRPluginSupport.h>
#include "ParticleViewer.h"
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/ShapeDrawable>
#include <osg/Shape>
#include <osg/Geode>
#include <osg/CullFace>
#include <osg/Sequence>
#include <osg/Geometry>
#include <osg/BlendFunc>
#include <osg/AlphaFunc>
#include <osg/LineWidth>
#include <PluginUtil/coSphere.h>
#include <cover/coVRAnimationManager.h>
#include <cover/coHud.h>
#include <cover/OpenCOVER.h>
#include <cover/coVRConfig.h>
#include <cover/coVRShader.h>
#include <util/byteswap.h>

static const coSphere::RenderMethod sphereMode = coSphere::RENDER_METHOD_ARB_POINT_SPRITES;

//! Function to clean up a string for further computation
/*! Remove tabs, multpile whitespaces, comments etc.
 *\param Reference to the string which should be cleared
 *\param Reference to a string which holds all comment characters
 *\return Copy of the cleared input string
 */
std::string
clearString(const std::string &str, std::string comment)
{
    std::string argv = str;
    if (argv.length() == 0)
        return (argv);

    // Remove all comments from the string
    for (std::string::size_type pos = 0; pos < comment.length(); pos++)
    {
        if (argv.find(comment[pos]) != std::string::npos)
            argv = argv.erase(argv.find(comment[pos]), argv.length());
    }

    // Skip the rest if the whole line was a comment
    if (0 == argv.length())
        return (argv);

    // Replace tabs with a space
    std::string::size_type pos = argv.find("\t");
    while (pos != std::string::npos)
    {
        argv = argv.replace(pos, 1, " ");
        pos = argv.find("\t");
    }

    // Replace multiple spaces with just one
    pos = argv.find("  ");
    while (pos != std::string::npos)
    {
        argv = argv.replace(pos, 2, " ");
        pos = argv.find("  ");
    }

    // Erase first char if it is a space
    if ((argv[0] == ' ') && (argv.length() > 0))
        argv = argv.erase(0, 1);

    // Erase last char if it is a space
    pos = argv.length();
    if (pos-- > 0)
        if (argv[pos] == ' ')
            argv = argv.erase(pos, 1);

    // Clear String if it is just the EOL char
    if (!argv.empty() && argv[0] == '\0')
        argv.clear();

    return (argv);
}

int
split(std::vector<std::string> &fields, const std::string &str, std::string fieldseparator /* = " " */)
{
    fields.clear();
    std::string worker = clearString(str, "");

    // Replace tabs with a space
    std::string::size_type pos = worker.find(fieldseparator);
    while (pos != std::string::npos)
    {
        std::string sub = worker.substr(0, pos);
        worker = worker.erase(0, pos + 1);
        fields.push_back(sub);
        pos = worker.find(fieldseparator);
    }
    fields.push_back(worker);
    return (fields.size());
}

//! Template function for string conversion
/*!
 *\param Reference to output value
 *\param Reference to input string
 *\return 0 on success, 1 on failure
 */
template <typename T>
unsigned int
strConvert(T &param, const std::string &str, bool expcast = 0)
{
    // Touch it to prevent warning message, is only used in specialised routines
    expcast = expcast;
    std::stringstream sstr(str);
    sstr >> param;
    if (sstr.width() != 0)
        return (1);

    return (0);
}

struct particleData
{
    float x, y, z;
    float xv, yv, zv;
    float val[100];
};

#define LINE_LEN 1000
using namespace osg;
TimeStepData::TimeStepData(int numParticles, unsigned int nf, unsigned int ni)
    : values(0)
    , Ivalues(0)
    , numParticles(numParticles)
    , numFloats(nf)
    , numInts(ni)
{
    if (nf)
    {
        values = new float *[numFloats];
        for (unsigned int i = 0; i < numFloats; i++)
        {
            values[i] = new float[numParticles];
        }
    }
    if (ni)
    {
        Ivalues = new int64_t *[numInts];
        for (unsigned int i = 0; i < numInts; i++)
        {
            Ivalues[i] = new int64_t[numParticles];
        }
    }
}

TimeStepData::~TimeStepData()
{
    for (unsigned int i = 0; i < numFloats; i++)
    {
        delete[] values[i];
    }
    for (unsigned int i = 0; i < numInts; i++)
    {
        delete[] Ivalues[i];
    }
    delete[] values;
    delete[] Ivalues;
}

extern ParticleViewer *plugin;

int Particles::read64(uint64_t &val)
{
    int res = fread(&val, sizeof(uint64_t), 1, fp);
    if (doSwap)
    {
        byteSwap(val);
    }
    return res;
}
int Particles::read32(int &val)
{
    int res = fread(&val, sizeof(int), 1, fp);
    if (doSwap)
    {
        byteSwap(val);
    }
    return res;
}

Particles::Particles(std::string filename, osg::Group *parent, int maxParticles)
: numTimesteps(0)
, switchNode(NULL)
, fp(NULL)
, interval(20)
, shader(NULL)
, doSwap(false)
, ParticleMode(M_PARTICLES)
, timesteps(NULL)
, numInts(0)
, numFloats(0)
, format(Particle)
, numHiddenVars(6)
{
    (void)maxParticles;
    fileName = filename;
    lineColor.set(0, 0, 1, 1);
    format = Particle;
    std::string suffix = "particle";
    if (filename.length() > 6 && strcmp(filename.c_str() + filename.length() - 6, ".coord") == 0)
    {
        format = IMWF;
        suffix = "coord";
    }
    if (filename.length() > 6 && strcmp(filename.c_str() + filename.length() - 6, ".chkpt") == 0)
    {
        format = IMWF;
        suffix = "chkpt";
    }
    if (filename.length() > 6 && strcmp(filename.c_str() + filename.length() - 6, ".crist") == 0)
    {
        format = IMWF;
        suffix = "crist";
    }
    if (filename.length() > 7 && strcmp(filename.c_str() + filename.length() - 7, ".indent") == 0)
    {
        format = Indent;
        suffix = "indent";
    }
    size_t pos = filename.find("line");
    std::string filenamebegin = filename.substr(0, pos);
    if (pos != string::npos)
    {
        ParticleMode = M_LINES;
        if (filename[pos + 4] == '_')
        {
            std::string tmpShaderName = filename.substr(pos + 5);
            pos = tmpShaderName.find("_");
            if (pos != string::npos)
            {
                shaderName = tmpShaderName.substr(0, pos);
                do
                {
                    tmpShaderName = tmpShaderName.substr(pos + 1);
                    pos = tmpShaderName.find("=");
                    if (pos != string::npos)
                    {
                        std::string param = tmpShaderName.substr(0, pos);
                        tmpShaderName = tmpShaderName.substr(pos + 1);
                        pos = tmpShaderName.find("_");
                        if (pos != string::npos)
                        {
                            std::string paramValue = tmpShaderName.substr(0, pos);
                            if (param == "R")
                            {
                                float val;
                                sscanf(paramValue.c_str(), "%f", &val);
                                lineColor[0] = val;
                            }
                            else if (param == "G")
                            {
                                float val;
                                sscanf(paramValue.c_str(), "%f", &val);
                                lineColor[1] = val;
                            }
                            else if (param == "B")
                            {
                                float val;
                                sscanf(paramValue.c_str(), "%f", &val);
                                lineColor[2] = val;
                            }
                            shaderParams[param] = paramValue;
                        }
                    }
                } while (pos != string::npos);
            }
        }
    }

    switchNode=new osg::Sequence();
    if (shaderName.length() > 1)
    {
        cerr << "trying shader " << shaderName << endl;
    }
    switchNode->setName(filename);
    shader = coVRShaderList::instance()->get(shaderName, &shaderParams);
    if (parent)
    {
        parent->addChild(switchNode);
    }
    else
    {
        cover->getObjectsRoot()->addChild(switchNode);
    }

    size_t found = filenamebegin.rfind('.');
    if (found != string::npos)
    {
        if (format == IMWF)
        {
            found -= 6;
        }
        else if (format == Indent)
        {
            found = filenamebegin.rfind('.', found-1);
            if (found == std::string::npos)
                found = 0;
        }

        int minNumber=0;
        sscanf(filenamebegin.c_str() + found + 1, "%d", &minNumber);
        std::string filebeg = filenamebegin.substr(0, found + 1);
        numTimesteps = 0;
        while (true)
        {
            char *tmpFileName = new char[found + 200];
            if (format == IMWF)
            {
                sprintf(tmpFileName, "%s%05d.%s", filebeg.c_str(), minNumber + numTimesteps, suffix.c_str());
            }
            else if (format == Indent)
            {
                sprintf(tmpFileName, "%s%d.%s", filebeg.c_str(), minNumber + numTimesteps, suffix.c_str());
            }
            else
            {
                sprintf(tmpFileName, "%s%4d.particle", filebeg.c_str(), minNumber + numTimesteps);
            }
            FILE *fp = fopen(tmpFileName, "r");
            delete[] tmpFileName;
            if (fp)
            {
                numTimesteps++;
                fclose(fp);
            }
            else
            {
                break;
            }
        }
        std::cerr << "found " << numTimesteps << " timesteps" << std::endl;
        int skipfact = 1;
        if (numTimesteps > 2000)
        {
            skipfact = numTimesteps/2000;
        }
        if (skipfact < 1)
            skipfact = 1;
        numTimesteps /= skipfact;
        if (numTimesteps > 0)
        {
            timesteps = new TimeStepData *[numTimesteps];
            for (int i = 0; i < numTimesteps; i++)
            {
                char *tmpFileName = new char[found + 200];
                if (format == IMWF)
                {
                    sprintf(tmpFileName, "%s%05d.%s", filebeg.c_str(), minNumber + i*skipfact, suffix.c_str());
                    if (readIMWFFile(tmpFileName, i) > 0)
                    {
                    }
                    else
                    {
                        cerr << "could not open" << tmpFileName << endl;
                    }
                }
                else if (format == Indent)
                {
                    sprintf(tmpFileName, "%s%d.%s", filebeg.c_str(), minNumber + i*skipfact, suffix.c_str());
                    if (readIndentFile(tmpFileName, i) > 0)
                    {
                    }
                    else
                    {
                        cerr << "could not open" << tmpFileName << endl;
                    }
                }
                else
                {
                    sprintf(tmpFileName, "%s%4d.particle", filebeg.c_str(), minNumber + i*skipfact);
                    if (readFile(tmpFileName, i) > 0)
                    {
                    }
                    else
                    {
                        cerr << "could not open" << tmpFileName << endl;
                    }
                }
                delete[] tmpFileName;
            }
            if (format == Particle)
            {
                summUpTimesteps(interval);
                updateColors();
            }
        }
    }

    if (numTimesteps == 0 && format==Particle) // we did not find a number of files it might be a binary file
    {
        fp = fopen(filename.c_str(), "rb");
        if (fp)
        {
            uint64_t timestepNumber;
            int blockstart;
            int blockend;
            //numVars=2;
            numFloats = 2;
            numInts = 0;
            fread(&blockstart, sizeof(int), 1, fp);
            char header[201];
            char *tmp = (char *)&blockstart;
            if (tmp[0] == 'R' && tmp[1] == 'E' && tmp[2] == 'C' && tmp[3] == 'O')
            {
                fseek(fp, 0, SEEK_SET);
                fread(&header, 201, 1, fp);
                //RECOM VERSION 0.1 X Y Z VX VY VZ TT D O2 INL PARTN                                            RECOM
                //RECOM R R R R R R R R R I I   								RECOM
                //RECOM Min(float) Min Min Min...   								RECOM
                //RECOM Max(float) Max ...   								RECOM
                //RECOM RadiusScale(float) RadiusScale ...   								RECOM
                //RECOM Interval(int) NumTimeSteps(int)  								RECOM
                if (strncmp(header, "RECOM VERSION 0.2", 17) == 0)
                {
                    char header2[202];
                    memset(header2, 0, sizeof(header2));
                    if (fread(&header2, 201, 1, fp) != 1)
                    {
                        std::cerr << "Particles::Particles: failed to read header 1" << std::endl;
                    }
                    char *buf = header2;
                    buf += 6;
                    numFloats = 0;
                    while (*buf == 'R' || *buf == 'I')
                    {
                        if (*buf == 'R')
                        {
                            variableTypes.push_back(T_FLOAT);
                            if (numFloats >= 6)
                            {
                                variableIndex.push_back(numFloats - 6);
                            }
                            numFloats++;
                        }
                        else
                        {
                            variableTypes.push_back(T_INT);
                            variableIndex.push_back(numInts);
                            numInts++;
                        }
                        //numVars++;
                        buf += 2;
                    }
                    //numVars -= 6 ; // Pos and Velocity
                    numFloats -= 6; // Pos and Velocity
                    buf = header;
                    buf += 33;
                    unsigned int numVars = numFloats + numInts;
                    for (unsigned int i = 0; i < numVars; i++)
                    {
                        char *buf2 = buf + 1;
                        while (*buf2 != ' ' && *buf2 != '\0')
                        {
                            buf2++;
                        }
                        if (*buf2 != '\0')
                        {
                            *buf2 = '\0';
                            variableNames.push_back(std::string(buf));
                        }
                        else
                        {
                            fprintf(stderr, "Wrong header %s\n", header);
                            break;
                        }
                        buf = buf2 + 1;
                    }
                    std::vector<std::string> id_fields_;
                    float tmpf;

                    if (fread(&header2, 201, 1, fp) != 1)
                    {
                        std::cerr << "Particles::Particles: failed to read header 1" << std::endl;
                    }
                    split(id_fields_, header2, " ");
                    // Remove HEADER start and end ("RECOM")
                    for (std::vector<std::string>::iterator it = id_fields_.begin(); it != id_fields_.end();)
                    {
                        if ((*it) == "RECOM")
                            it = id_fields_.erase(it);
                        else
                            ++it;
                    }
                    for (unsigned int i = 6; i < numVars + 6; i++)
                    {
                        strConvert(tmpf, id_fields_[i], false);
                        variableMin.push_back(tmpf);
                    }

                    if (fread(&header2, 201, 1, fp) != 1)
                    {
                        std::cerr << "Particles::Particles: failed to read header 1" << std::endl;
                    }
                    split(id_fields_, header2, " ");
                    // Remove HEADER start and end ("RECOM")
                    for (std::vector<std::string>::iterator it = id_fields_.begin(); it != id_fields_.end();)
                    {
                        if ((*it) == "RECOM")
                            it = id_fields_.erase(it);
                        else
                            ++it;
                    }
                    for (unsigned int i = 6; i < numVars + 6; i++)
                    {
                        strConvert(tmpf, id_fields_[i], false);
                        variableMax.push_back(tmpf);
                    }

                    if (fread(&header2, 201, 1, fp) != 1)
                    {
                        std::cerr << "Particles::Particles: failed to read header 1" << std::endl;
                    }
                    split(id_fields_, header2, " ");
                    // Remove HEADER start and end ("RECOM")
                    for (std::vector<std::string>::iterator it = id_fields_.begin(); it != id_fields_.end();)
                    {
                        if ((*it) == "RECOM")
                            it = id_fields_.erase(it);
                        else
                            ++it;
                    }
                    for (unsigned int i = 6; i < numVars + numHiddenVars; i++)
                    {
                        strConvert(tmpf, id_fields_[i], false);
                        variableScale.push_back(tmpf);
                    }

                    // read interval
                    if (fread(&header2, 201, 1, fp) != 1)
                    {
                        std::cerr << "Particles::Particles: failed to read header 1" << std::endl;
                    }
                    buf = header2;
                    if (strlen(buf) >= 6)
                    {
                        buf += 6;
                        while (*buf == ' ' && *buf != '\0')
                        {
                            buf++;
                        }
                        sscanf(buf, "%d", &interval);
                    }
                }
                else
                {
                    fprintf(stderr, "Unknown file version %s\n", header);
                }

                if (fread(&blockstart, sizeof(int), 1, fp) != 1)
                {
                    std::cerr << "Particles::Particles: failed to read blockstart" << std::endl;
                }
            }
            else
            {
                for (int ctr = 0; ctr < 6; ++ctr)
                    variableTypes.push_back(T_FLOAT);
                numHiddenVars = 6;
                variableNames.push_back(std::string("TT"));
                variableTypes.push_back(T_FLOAT);
                variableIndex.push_back(0);
                variableNames.push_back(std::string("D"));
                variableTypes.push_back(T_FLOAT);
                variableIndex.push_back(1);
            }
            doSwap = false;
            if (blockstart != 8)
            {
                doSwap = true;
                byteSwap(blockstart);
            }
            if (blockstart == 8)
            {
                read64(timestepNumber);
                read32(blockend);
                if (blockstart == blockend)
                {
                    numTimesteps = (int)timestepNumber;
                    timesteps = new TimeStepData *[numTimesteps];
                    //bool failed = false;
                    for (int i = 0; i < numTimesteps; i++)
                    {
                        if (readBinaryTimestep(i) > 0)
                        {
                        }
                        else
                        {
                            //failed = true;
                            numTimesteps = i;
                            break;
                        }
                    }
                    //if(!failed)
                    //{
                    summUpTimesteps(interval);
                    updateColors();
                    //}
                }
            }
            fclose(fp);
        }
    }
}

void Particles::summUpTimesteps(int increment)
{
    for (int i = 1; i < numTimesteps; i++)
    {
        osg::Geode *geode = (osg::Geode *)switchNode->getChild(i);
        for (unsigned int n = 1; n < geode->getNumDrawables(); n++)
        {
            geode->removeDrawables(n);
        }
        int numTimestepsToAdd = i / increment; //every increment timestep, we add some particles from the start
        int index = i;
        for (int n = 0; n < numTimestepsToAdd; n++)
        {
            index -= increment;
            osg::Geode *earlyGeode = (osg::Geode *)switchNode->getChild(index);
            geode->addDrawable(earlyGeode->getDrawable(0));
        }
    }
}

int Particles::getMode()
{
    return ParticleMode;
}

int Particles::readBinaryTimestep(int timestep)
{
    osg::Geode *geode = new osg::Geode();
    int blockstart, blockend;
    char buf[LINE_LEN];

    uint64_t particleNumber;
    read32(blockstart);
    read64(particleNumber);
    read32(blockend);
    if (blockend != blockstart)
    {
        return -1;
    }

    int numParticles = particleNumber;
    sprintf(buf, "reading num=%d timestep=%d", numParticles, timestep);
    if (timestep % 50 == 0)
    {
        OpenCOVER::instance()->hud->setText2(buf);
        OpenCOVER::instance()->hud->redraw();
    }
    timesteps[timestep] = new TimeStepData(numParticles, numFloats, numInts);
    geode->setName(buf);
    float *xc = new float[numParticles];
    float *yc = new float[numParticles];
    float *zc = new float[numParticles];
    float *xv = new float[numParticles];
    float *yv = new float[numParticles];
    float *zv = new float[numParticles];
    float **values = timesteps[timestep]->values;
    int64_t **Ivalues = timesteps[timestep]->Ivalues;
    timesteps[timestep]->geode = geode;

    struct particleData oneParticle;
    int blockoverhead = 0;
    for (int i = 0; i < numParticles; i++)
    {
        read32(blockstart);
        if (blockstart != (6 + numFloats) * sizeof(float) + (numInts * sizeof(coInt64)) && blockoverhead == 0)
        {
            if (blockstart != 60)
            {
                cerr << "ERROR" << endl;
                cerr << "expected 32 byte particle data but got block with " << blockstart << " bytes " << endl;
                cerr << "while reading particle " << i << " of " << numParticles << " in timestep " << timestep << endl;
            }
            blockoverhead = blockstart - ((6 + numFloats) * sizeof(float) + (numInts * sizeof(coInt64)));
            if (blockoverhead < 0)
                return -1;
        }
        fread(&oneParticle, (6 + numFloats) * sizeof(float) + (numInts * sizeof(coInt64)), 1, fp);
        if (blockoverhead > 0)
        {
            //fseek(fp,blockoverhead,SEEK_CUR); // strange, blockstart sais 60 but it seems to be only 32....
        }
        unsigned int nf = 0;
        unsigned int ni = 0;
        unsigned int n = 0;
        unsigned int numVars = numFloats + numInts;
        for (unsigned int nv = 0; nv < numVars; nv++)
        {
            if (variableTypes[nv + numHiddenVars] == T_FLOAT)
            {
                values[nf][i] = oneParticle.val[n];
                n++;
                nf++;
            }
            else
            {
                Ivalues[ni][i] = *(int64_t *)(&oneParticle.val[n]);
                n += 2;
                ni++;
            }
        }
        if (blockoverhead > 0)
        {
            xc[i] = oneParticle.xv;
            yc[i] = oneParticle.yv;
            zc[i] = oneParticle.zv;
            xv[i] = oneParticle.x;
            yv[i] = oneParticle.y;
            zv[i] = oneParticle.z;
        }
        else
        {
            xc[i] = oneParticle.x;
            yc[i] = oneParticle.y;
            zc[i] = oneParticle.z;
            xv[i] = oneParticle.xv;
            yv[i] = oneParticle.yv;
            zv[i] = oneParticle.zv;
        }
        if (doSwap)
        {
            for (unsigned int n = 0; n < numInts; n++)
            {
                byteSwap(Ivalues[n][i]);
            }
            for (unsigned int n = 0; n < numFloats; n++)
            {
                byteSwap(values[n][i]);
            }
            values[1][i] *= 200;
            byteSwap(xc[i]);
            byteSwap(yc[i]);
            byteSwap(zc[i]);
            byteSwap(xv[i]);
            byteSwap(yv[i]);
            byteSwap(zv[i]);
        }
        read32(blockend);
        if (blockend != blockstart)
        {
            return -1;
        }
    }

    //if(iRenderMethod == coSphere::RENDER_METHOD_PARTICLE_CLOUD)
    //   transparent = true;
    osg::StateSet *geoState = geode->getOrCreateStateSet();
    //setDefaultMaterial(geoState, transparent);
    geoState->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    osg::BlendFunc *blendFunc = new osg::BlendFunc();
    blendFunc->setFunction(osg::BlendFunc::SRC_ALPHA, osg::BlendFunc::ONE_MINUS_SRC_ALPHA);
    geoState->setAttributeAndModes(blendFunc, osg::StateAttribute::ON);
    osg::AlphaFunc *alphaFunc = new osg::AlphaFunc();
    alphaFunc->setFunction(osg::AlphaFunc::ALWAYS, 1.0);
    geoState->setAttributeAndModes(alphaFunc, osg::StateAttribute::OFF);

    geode->setStateSet(geoState);

    if (ParticleMode == M_PARTICLES)
    {
        coSphere *sphere = new coSphere();
        sphere->setRenderMethod(sphereMode);
        sphere->setMaxRadius(1);

        timesteps[timestep]->sphere = sphere;
        //sphere->setColorBinding(colorbinding);
        float *radii = values[1];
        sphere->setCoords(numParticles, xc, yc, zc, radii);
        float f = plugin->getRadius();
        float *rn = new float[numParticles];
        for (int n = 0; n < numParticles; n++)
        {
            rn[n] = radii[n] * f;
        }
        sphere->updateRadii(rn);
        delete[] rn;

        geode->addDrawable(sphere);
        timesteps[timestep]->lines = NULL;
    }
    else
    {
        osg::Geometry *lines = new osg::Geometry();
        cover->setRenderStrategy(lines);

        // set up geometry
        osg::Vec3Array *vert = new osg::Vec3Array;
        osg::DrawArrays *primitives = new osg::DrawArrays(osg::PrimitiveSet::LINES, 0, numParticles * 2);
        for (int i = 0; i < numParticles; i++)
        {
            vert->push_back(osg::Vec3(xc[i], yc[i], zc[i]));
            vert->push_back(osg::Vec3(xc[i] + (xv[i] / 100), yc[i] + (yv[i] / 100), zc[i] + (zv[i] / 100)));
        }
        lines->setVertexArray(vert);
        lines->addPrimitiveSet(primitives);
        osg::Vec4Array *colArr = new osg::Vec4Array();
        colArr->push_back(lineColor);
        lines->setColorArray(colArr);
        lines->setColorBinding(osg::Geometry::BIND_OVERALL);

        timesteps[timestep]->sphere = NULL;
        timesteps[timestep]->lines = lines;
        timesteps[timestep]->colors = colArr;
        geode->addDrawable(lines);

        if (shader)
            shader->apply(geode);
        osg::LineWidth *lineWidth = new osg::LineWidth(6);
        geoState->setAttributeAndModes(lineWidth, osg::StateAttribute::ON);
    }
    switchNode->addChild(geode);
    switchNode->setValue(timestep);

    delete[] xc;
    delete[] yc;
    delete[] zc;
    delete[] xv;
    delete[] yv;
    delete[] zv;
    return numParticles;
}

int Particles::readFile(char *fn, int timestep)
{
    FILE *fp = fopen(fn, "r");
    if (fp)
    {

        int numParticles = 0;
        bool binary = false;
        char buf[LINE_LEN];
        osg::Geode *geode = new osg::Geode();
        // is it binary or ascii?
        // read binary num particles
        /* long long particleNumber;
      int blockstart;
      int blockend;
      fread(&blockstart,sizeof(int),1,fp);
      fread(&particleNumber,sizeof(long long),1,fp);
      fread(&blockend,sizeof(int),1,fp);
      if(blockstart!=blockend)
         binary=false;*/
        if (!binary)
        {
            fseek(fp, 0, SEEK_SET);

            while (!feof(fp))
            {
                fgets(buf, LINE_LEN, fp);
                numParticles++;
            }
            fseek(fp, 0, SEEK_SET);
        }
        if (timestep % 10 == 0)
        {
            sprintf(buf, "reading %s, num=%d timestep=%d", fn, numParticles, timestep);
            OpenCOVER::instance()->hud->setText2(buf);
            OpenCOVER::instance()->hud->redraw();
        }
        timesteps[timestep] = new TimeStepData(numParticles, numFloats, numInts);
        geode->setName(buf);
        float *xc = new float[numParticles];
        float *yc = new float[numParticles];
        float *zc = new float[numParticles];
        float *xv = new float[numParticles];
        float *yv = new float[numParticles];
        float *zv = new float[numParticles];
        float **values = timesteps[timestep]->values;
        timesteps[timestep]->geode = geode;

        int n = 0;
        struct particleData oneParticle;

        if (binary)
        {
            for (int i = 0; i < numParticles; i++)
            {
                int blockstart, blockend;
                fread(&blockstart, sizeof(int), 1, fp);
                fread(&oneParticle, (6 + numFloats) * sizeof(float) + numInts * sizeof(coInt64), 1, fp);
                fread(&blockend, sizeof(int), 1, fp);
                values[0][i] = oneParticle.val[0];
                values[1][i] = oneParticle.val[1] * 200;
                xc[i] = oneParticle.x;
                yc[i] = oneParticle.y;
                zc[i] = oneParticle.z;
                xv[i] = oneParticle.xv;
                yv[i] = oneParticle.yv;
                zv[i] = oneParticle.zv;
            }
        }
        else
        {
            //float radius = plugin->getRadius();
            while (!feof(fp))
            {
                if (n > numParticles)
                {
                    cerr << "oops read more than last time..." << endl;
                    break;
                }
                fgets(buf, LINE_LEN, fp);
                values[0][n] = 200.0;
                values[1][n] = 0.001;
                sscanf(buf, "%f %f %f %f %f %f %f %f", xc + n, yc + n, zc + n, xv + n, yv + n, zv + n, values[0] + n, values[1] + n);
                values[1][n] *= 200;
                n++;
            }
        }
        fclose(fp);

        //if(iRenderMethod == coSphere::RENDER_METHOD_PARTICLE_CLOUD)
        //   transparent = true;
        osg::StateSet *geoState = geode->getOrCreateStateSet();
        //setDefaultMaterial(geoState, transparent);
        geoState->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

        osg::BlendFunc *blendFunc = new osg::BlendFunc();
        blendFunc->setFunction(osg::BlendFunc::SRC_ALPHA, osg::BlendFunc::ONE_MINUS_SRC_ALPHA);
        geoState->setAttributeAndModes(blendFunc, osg::StateAttribute::ON);
        osg::AlphaFunc *alphaFunc = new osg::AlphaFunc();
        alphaFunc->setFunction(osg::AlphaFunc::ALWAYS, 1.0);
        geoState->setAttributeAndModes(alphaFunc, osg::StateAttribute::OFF);

        geode->setStateSet(geoState);

        coSphere *sphere = new coSphere();
        sphere->setRenderMethod(sphereMode);

        sphere->setMaxRadius(1);

        timesteps[timestep]->sphere = sphere;
        //sphere->setColorBinding(colorbinding);
        sphere->setCoords(numParticles, xc, yc, zc, values[1]);

        geode->addDrawable(sphere);
        switchNode->addChild(geode, true);

        delete[] xc;
        delete[] yc;
        delete[] zc;
        delete[] xv;
        delete[] yv;
        delete[] zv;

        colorizeAndResize(timestep);

        return numParticles;
    }
    else
    {
        return -1;
    }
    return 0;
}

int Particles::readIMWFFile(char *fn, int timestep)
{
    FILE *fp = fopen(fn, "r");
    if (fp)
    {

        int numParticles = 0;
        bool binary = false;
        char buf[LINE_LEN];
        osg::Geode *geode = new osg::Geode();
        int version = 0;
        while (!feof(fp))
        {
            fgets(buf, LINE_LEN, fp);
            if (buf[0] != '#')
            {
                numParticles++;
            }
            else
            {
                if (strncmp(buf, "#C number type mass x y z vx vy vz crist", 40) == 0)
                {
                    version = 1;
                }
                if (strncmp(buf, "#C number type mass x y z vx vy vz Epot eam_rho", 47) == 0)
                {
                    version = 2;
                }
            }
        }
        fseek(fp, 0, SEEK_SET);

        if (timestep % 10 == 0)
        {
            sprintf(buf, "reading %s, num=%d timestep=%d", fn, numParticles, timestep);
            OpenCOVER::instance()->hud->setText2(buf);
            OpenCOVER::instance()->hud->redraw();
            switchNode->setValue(timestep);
            //switchNode->setAllChildrenOff();
        }

        //42 1 1.572263 1.326834 62.641356 14.000000 14.000000 0.000000 0.000000
        numFloats = 4;
        numInts = 2;

        if (version == 0)
        {
            numHiddenVars = 3;
            variableTypes.push_back(T_FLOAT);
            variableTypes.push_back(T_FLOAT);
            variableTypes.push_back(T_FLOAT);
        }
        else
        {
            numHiddenVars = 6;
            variableTypes.push_back(T_FLOAT);
            variableTypes.push_back(T_FLOAT);
            variableTypes.push_back(T_FLOAT);
            variableTypes.push_back(T_FLOAT);
            variableTypes.push_back(T_FLOAT);
            variableTypes.push_back(T_FLOAT);
        }

        if (version == 0)
        {
            variableTypes.push_back(T_FLOAT);
            variableIndex.push_back(0);
            variableScale.push_back(1.0);
            variableMin.push_back(1.0);
            variableMax.push_back(18.0);
            variableNames.push_back(std::string("N_FE"));
            variableTypes.push_back(T_FLOAT);
            variableIndex.push_back(1);
            variableScale.push_back(1.0);
            variableMin.push_back(1.0);
            variableMax.push_back(18.0);
            variableNames.push_back(std::string("N_CU"));
            variableTypes.push_back(T_FLOAT);
            variableIndex.push_back(2);
            variableScale.push_back(1.0);
            variableMin.push_back(1.0);
            variableMax.push_back(18.0);
            variableNames.push_back(std::string("N_NI"));
            variableTypes.push_back(T_FLOAT);
            variableIndex.push_back(3);
            variableScale.push_back(1.0);
            variableMin.push_back(1.0);
            variableMax.push_back(18.0);
        }
        if (version == 1)
        {

            numFloats = 8;
            numInts = 2;
            variableTypes.push_back(T_FLOAT);
            variableIndex.push_back(0);
            variableScale.push_back(1.0);
            variableMin.push_back(1.0);
            variableMax.push_back(18.0);
            variableNames.push_back(std::string("Mass"));
            variableTypes.push_back(T_FLOAT);
            variableIndex.push_back(1);
            variableScale.push_back(1.0);
            variableMin.push_back(1.0);
            variableMax.push_back(18.0);
            variableNames.push_back(std::string("crist"));
        }
        if (version == 2)
        {
            numFloats = 9;
            numInts = 2;
            variableTypes.push_back(T_FLOAT);
            variableIndex.push_back(0);
            variableScale.push_back(1.0);
            variableMin.push_back(1.0);
            variableMax.push_back(18.0);
            variableNames.push_back(std::string("Mass"));
            variableTypes.push_back(T_FLOAT);
            variableIndex.push_back(1);
            variableScale.push_back(1.0);
            variableMin.push_back(1.0);
            variableMax.push_back(18.0);
            variableNames.push_back(std::string("Epot"));
            variableTypes.push_back(T_FLOAT);
            variableIndex.push_back(2);
            variableScale.push_back(1.0);
            variableMin.push_back(1.0);
            variableMax.push_back(18.0);
            variableNames.push_back(std::string("eam_rho"));
        }

        variableNames.push_back(std::string("Number"));
        variableTypes.push_back(T_INT);
        variableIndex.push_back(0);
        variableScale.push_back(1.0);
        variableMin.push_back(1.0);
        variableMax.push_back(18.0);
        variableNames.push_back(std::string("TYPE"));
        variableTypes.push_back(T_INT);
        variableIndex.push_back(1);
        variableScale.push_back(1.0);
        variableMin.push_back(1.0);
        variableMax.push_back(18.0);

        timesteps[timestep] = new TimeStepData(numParticles, numFloats, numInts);
        geode->setName(buf);
        float *xc = new float[numParticles];
        float *yc = new float[numParticles];
        float *zc = new float[numParticles];
        float **values = timesteps[timestep]->values;
        int64_t **Ivalues = timesteps[timestep]->Ivalues;
        timesteps[timestep]->geode = geode;

        int n = 0;
        struct particleData oneParticle;

        //float radius = plugin->getRadius();
        if (version == 0)
        {
            while (!feof(fp))
            {
                if (n > numParticles)
                {
                    cerr << "oops read more than last time..." << endl;
                    break;
                }
                fgets(buf, LINE_LEN, fp);
                if (buf[0] != '#')
                {
                    int num, type;
                    sscanf(buf, "%d %d %f %f %f %f %f %f %f", &num, &type, xc + n, yc + n, zc + n, values[0] + n, values[1] + n, values[2] + n, values[3] + n);
                    Ivalues[0][n] = num;
                    Ivalues[1][n] = type;
                    n++;
                }
            }
        }
        if (version == 1)
        {
            while (!feof(fp))
            {
                if (n > numParticles)
                {
                    cerr << "oops read more than last time..." << endl;
                    break;
                }
                fgets(buf, LINE_LEN, fp);
                if (buf[0] != '#')
                {
                    int num, type;
                    float tmpf;
                    sscanf(buf, "%d %d %f %f %f %f %f %f %f %f", &num, &type, values[0] + n, xc + n, yc + n, zc + n, &tmpf, &tmpf, &tmpf, values[1] + n);
                    Ivalues[0][n] = num;
                    Ivalues[1][n] = type;
                    n++;
                }
            }
        }
        if (version == 2)
        {
            while (!feof(fp))
            {
                if (n > numParticles)
                {
                    cerr << "oops read more than last time..." << endl;
                    break;
                }
                fgets(buf, LINE_LEN, fp);
                if (buf[0] != '#')
                {
                    int num, type;
                    float tmpf;
                    sscanf(buf, "%d %d %f %f %f %f %f %f %f %f %f", &num, &type, values[0] + n, xc + n, yc + n, zc + n, &tmpf, &tmpf, &tmpf, values[1] + n, values[2] + n);
                    Ivalues[0][n] = num;
                    Ivalues[1][n] = type;
                    n++;
                }
            }
        }
        fclose(fp);

        //if(iRenderMethod == coSphere::RENDER_METHOD_PARTICLE_CLOUD)
        //   transparent = true;
        osg::StateSet *geoState = geode->getOrCreateStateSet();
        //setDefaultMaterial(geoState, transparent);
        geoState->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

        osg::BlendFunc *blendFunc = new osg::BlendFunc();
        blendFunc->setFunction(osg::BlendFunc::SRC_ALPHA, osg::BlendFunc::ONE_MINUS_SRC_ALPHA);
        geoState->setAttributeAndModes(blendFunc, osg::StateAttribute::ON);
        osg::AlphaFunc *alphaFunc = new osg::AlphaFunc();
        alphaFunc->setFunction(osg::AlphaFunc::ALWAYS, 1.0);
        geoState->setAttributeAndModes(alphaFunc, osg::StateAttribute::OFF);

        geode->setStateSet(geoState);

        coSphere *sphere = new coSphere();
        sphere->setRenderMethod(sphereMode);

        sphere->setMaxRadius(1);

        timesteps[timestep]->sphere = sphere;
        //sphere->setColorBinding(colorbinding);
        sphere->setCoords(numParticles, xc, yc, zc);

        delete[] xc;
        delete[] yc;
        delete[] zc;

        geode->addDrawable(sphere);
        switchNode->addChild(geode, true);

        colorizeAndResize(timestep);

        return numParticles;
    }
    else
    {
        return -1;
    }
    return 0;
}

int Particles::readIndentFile(char *fn, int timestep)
{
    numInts = 2;
    numFloats = 2;
    numHiddenVars = 0;

    variableNames.push_back(std::string("Species"));
    variableTypes.push_back(T_INT);
    variableIndex.push_back(0);
    variableScale.push_back(1.0);
    variableMin.push_back(0);
    variableMax.push_back(2);

    variableNames.push_back(std::string("Structure Type"));
    variableTypes.push_back(T_INT);
    variableIndex.push_back(1);
    variableScale.push_back(1.0);
    variableMin.push_back(0);
    variableMax.push_back(4);

    variableNames.push_back(std::string("E Pot"));
    variableTypes.push_back(T_FLOAT);
    variableIndex.push_back(0);
    variableScale.push_back(1.0);
    variableMin.push_back(-3.);
    variableMax.push_back(-2.5);

    variableNames.push_back(std::string("Delta E Pot"));
    variableTypes.push_back(T_FLOAT);
    variableIndex.push_back(1);
    variableScale.push_back(1.0);
    variableMin.push_back(-0.1);
    variableMax.push_back(+0.1);

    std::string filename(fn);
    FILE *fp = NULL;
    bool read = !restore(filename+".dump", timestep);
    if (read)
    {
        fp = fopen(fn, "r");
        if (!fp)
        {
            return -1;
        }

        int numParticles = 0, numLines = 0;
        while (!feof(fp))
        {
            char buf[LINE_LEN];
            fgets(buf, LINE_LEN, fp);
            if (numLines == 0) {
                sscanf(buf, "%d", &numParticles);
            }
            ++numLines;
        }
        // first 2 lines are header, plus a terminating one
        if (numLines < 1 || numParticles != numLines-3)
        {
            std::cerr << "invalid files: #lines=" << numLines << ", but #particles=" << numParticles << " (should be #lines+2)" << std::endl; 
            fclose(fp);
            return -1;
        }
        std::cerr << "reading " << numParticles << " particles for timestep " << timestep << std::endl;

        fseek(fp, 0, SEEK_SET);
        timesteps[timestep] = new TimeStepData(numParticles, numFloats, numInts);

        osg::Geode *geode = new osg::Geode();
        geode->setName(fn);
        timesteps[timestep]->geode = geode;

        coSphere *sphere = new coSphere();
        sphere->setRenderMethod(sphereMode);
        sphere->setMaxRadius(1);
        timesteps[timestep]->sphere = sphere;
    }

    float **values = timesteps[timestep]->values;
    int64_t **Ivalues = timesteps[timestep]->Ivalues;
    int numParticles = timesteps[timestep]->numParticles;

    if (timestep % 10 == 0)
    {
        char buf[LINE_LEN];
        sprintf(buf, "reading %s, num=%d timestep=%d", fn, numParticles, timestep);
        OpenCOVER::instance()->hud->setText2(buf);
        OpenCOVER::instance()->hud->redraw();
        switchNode->setValue(timestep);
    }

    if (read)
    {
        float *xc = new float[numParticles];
        float *yc = new float[numParticles];
        float *zc = new float[numParticles];
        //float radius = plugin->getRadius();
        //
        int i = 0;
        int n = 0;
        while (!feof(fp))
        {
            if (n >= numParticles)
            {
                break;
            }

            char buf[LINE_LEN];
            fgets(buf, LINE_LEN, fp);
            if (i >= 2 && i<numParticles+2)
            {
                assert(n < numParticles);
                int species, type;
                //Type_1 2.11042 2.0290799 -3.4694503e-18 0 -2.93237 -0.00126
                int nf = sscanf(buf, "Type_%d %f %f %f %d %f %f", &species, xc + n, yc + n, zc + n, &type, values[0] + n, values[1] + n);
                if (nf != 7)
                {
                    std::cerr << "read error in line " << i << ": only got " << nf << " values (instead of 7)" << std::endl;
                }
                Ivalues[0][n] = species;
                Ivalues[1][n] = type;
                if (zc[n] > 0.01)
                {
                    // ignore label atoms
                    ++n;
                }
            }
            i++;
        }
        numParticles = n;
        timesteps[timestep]->numParticles = numParticles;
        fclose(fp);
        fp = NULL;

        coSphere *sphere = timesteps[timestep]->sphere;
        sphere->setCoords(numParticles, xc, yc, zc);

        dump(filename+".dump", timestep, xc, yc, zc);
        delete[] xc;
        delete[] yc;
        delete[] zc;
    }

    osg::Geode *geode = timesteps[timestep]->geode;
    //if(iRenderMethod == coSphere::RENDER_METHOD_PARTICLE_CLOUD)
    //   transparent = true;
    osg::StateSet *geoState = geode->getOrCreateStateSet();
    //setDefaultMaterial(geoState, transparent);
    geoState->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    osg::BlendFunc *blendFunc = new osg::BlendFunc();
    blendFunc->setFunction(osg::BlendFunc::SRC_ALPHA, osg::BlendFunc::ONE_MINUS_SRC_ALPHA);
    geoState->setAttributeAndModes(blendFunc, osg::StateAttribute::ON);
    osg::AlphaFunc *alphaFunc = new osg::AlphaFunc();
    alphaFunc->setFunction(osg::AlphaFunc::ALWAYS, 1.0);
    geoState->setAttributeAndModes(alphaFunc, osg::StateAttribute::OFF);

    geode->setStateSet(geoState);

    //sphere->setColorBinding(colorbinding);

    geode->addDrawable(timesteps[timestep]->sphere);
    switchNode->addChild(geode, true);

    colorizeAndResize(timestep);

    return numParticles;
}

void Particles::updateRadii(unsigned int valueNumber)
{
    float f = plugin->getRadius();
    for (int i = 0; i < numTimesteps; i++)
    {
        if (timesteps[i]->sphere)
        {
            int numParticles = timesteps[i]->numParticles;
            coSphere *s = timesteps[i]->sphere;
            float *rn = new float[numParticles];
            if (variableTypes[valueNumber + numHiddenVars] == T_FLOAT)
            {
                float *radii = timesteps[i]->values[variableIndex[valueNumber]];
                for (int n = 0; n < numParticles; n++)
                {
                    rn[n] = radii[n] * f;
                }
            }
            else
            {
                int64_t *radii = timesteps[i]->Ivalues[variableIndex[valueNumber]];
                for (int n = 0; n < numParticles; n++)
                {
                    rn[n] = radii[n] * f;
                }
            }

            s->updateRadii(rn);
            delete[] rn;
        }
    }
}
void Particles::updateColors(unsigned int valueNumber, unsigned int aValueNumber)
{
    for (int i = 0; i < numTimesteps; i++)
    {
        if (timesteps[i]->sphere)
        {
            int numParticles = timesteps[i]->numParticles;
            float *rc = new float[numParticles];
            float *gc = new float[numParticles];
            float *bc = new float[numParticles];
            float minVal = plugin->getMinVal();
            float maxVal = plugin->getMaxVal();
            if (variableTypes[valueNumber + numHiddenVars] == T_FLOAT)
            {
                float *temp = timesteps[i]->values[variableIndex[valueNumber]];
                for (int n = 0; n < numParticles; n++)
                {
                    osg::Vec4 c = plugin->getColor((temp[n] - minVal) / (maxVal - minVal), getMode());
                    rc[n] = c[0];
                    gc[n] = c[1];
                    bc[n] = c[2];
                }
            }
            else
            {
                int64_t *temp = timesteps[i]->Ivalues[variableIndex[valueNumber]];
                for (int n = 0; n < numParticles; n++)
                {
                    osg::Vec4 c = plugin->getColor((temp[n] - minVal) / (maxVal - minVal), getMode());
                    rc[n] = c[0];
                    gc[n] = c[1];
                    bc[n] = c[2];
                }
            }
            timesteps[i]->sphere->updateColors(rc, gc, bc);

            delete[] rc;
            delete[] gc;
            delete[] bc;
        }
        else
        {
            osg::Vec4Array *colArr;
            osg::Geometry *lines = dynamic_cast<osg::Geometry *>(timesteps[i]->lines.get());
            if (aValueNumber == 0) // constant Color
            {
                colArr = new osg::Vec4Array();
                colArr->push_back(lineColor);
                lines->setColorArray(colArr);
                lines->setColorBinding(osg::Geometry::BIND_OVERALL);
            }
            else
            {
                int numParticles = timesteps[i]->numParticles;
                colArr = new osg::Vec4Array(numParticles * 2);
                float minVal = plugin->getAMinVal();
                float maxVal = plugin->getAMaxVal();
                if (variableTypes[aValueNumber + numHiddenVars - 1] == T_FLOAT)
                {
                    float *temp = timesteps[i]->values[variableIndex[aValueNumber - 1]];
                    for (int n = 0; n < numParticles; n++)
                    {
                        osg::Vec4 c = plugin->getColor((temp[n] - minVal) / (maxVal - minVal), getMode());
                        colArr->at(n * 2).r() = c[0];
                        colArr->at(n * 2).g() = c[1];
                        colArr->at(n * 2).b() = c[2];
                        colArr->at(n * 2).a() = 1;
                        colArr->at(n * 2 + 1).r() = c[0];
                        colArr->at(n * 2 + 1).g() = c[1];
                        colArr->at(n * 2 + 1).b() = c[2];
                        colArr->at(n * 2 + 1).a() = 1;
                    }
                }
                else
                {
                    int64_t *temp = timesteps[i]->Ivalues[variableIndex[aValueNumber - 1]];
                    for (int n = 0; n < numParticles; n++)
                    {
                        osg::Vec4 c = plugin->getColor((temp[n] - minVal) / (maxVal - minVal), getMode());
                        colArr->at(n * 2).r() = c[0];
                        colArr->at(n * 2).g() = c[1];
                        colArr->at(n * 2).b() = c[2];
                        colArr->at(n * 2).a() = 1;
                        colArr->at(n * 2 + 1).r() = c[0];
                        colArr->at(n * 2 + 1).g() = c[1];
                        colArr->at(n * 2 + 1).b() = c[2];
                        colArr->at(n * 2 + 1).a() = 1;
                    }
                }
                lines->setColorArray(colArr);
                lines->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
            }
            if (lines)
            {
                lines->dirtyDisplayList();
            }
        }
    }
}

void Particles::setTimestep(int i)
{
    //switchNode->setAllChildrenOff();
    switchNode->setValue(i);

    //switchNode->setSingleChildOn(i);
}

//destructor
Particles::~Particles()
{
    switchNode->getParent(0)->removeChild(switchNode);
    for (int i = 0; i < numTimesteps; i++)
        delete timesteps[i];
    delete[] timesteps;
}

void Particles::colorizeAndResize(int timestep)
{
    int begin=0, end=numTimesteps;
    if (timestep >= 0)
    {
        begin = timestep;
        end = timestep+1;
    }

    for (int t=begin; t<end; ++t)
    {
        TimeStepData *td = timesteps[t];
        const int numParticles = td->numParticles;
        int64_t **Ivalues = td->Ivalues;
        float **values = td->values;

        if (format == Indent)
        {
            float *r = new float[numParticles];
            for (int n = 0; n < numParticles; n++)
            {
                r[n] = 10.; // should not happen

                float dEPot = td->values[1][n];
                int S = Ivalues[0][n];
                if (S == 0)
                {
                    r[n] = 0.1;
                }
                else if (S == 1)
                {
                    // aluminum
                    r[n] = 0.1+1.5*dEPot;
                }
                else if (S == 2)
                {
                    // indenter
                    r[n] = 0.4;
                }
            }
            td->sphere->updateRadii(r);
            delete[] r;

            float *rc = new float[numParticles];
            float *gc = new float[numParticles];
            float *bc = new float[numParticles];
            for (int n = 0; n < numParticles; n++)
            {
                int S = td->Ivalues[0][n];
                int t = td->Ivalues[1][n];
                if (S == 1)
                {
                    // aluminum
                    switch(t)
                    {
                        case 0:
                            //other, kommen am Rand der Versetzungen/Stapelfehler und an Oberflächen vor                                                                                                                                                                                      
                            rc[n] = 0.0;
                            gc[n] = 0.8;
                            bc[n] = 0.8;
                            break;
                        case 1:
                            //fcc (face centered cubic), ausgeblendet
                            rc[n] = 1.0;
                            gc[n] = 1.0;
                            bc[n] = 1.0;
                            break;
                        case 2:
                            //hcp (hexagonal closed packed), Stepelfehler/Versetzungen 
                            rc[n] = 1.0;
                            gc[n] = 0.0;
                            bc[n] = 0.0;
                            break;
                        case 3:
                            //bcc (body centered cubic), kommt nur vereinzelt vor
                            rc[n] = 1.0;
                            gc[n] = 1.0;
                            bc[n] = 0.0;
                            break;
                        case 4:
                            //icohedral, kommt nur vereinzelt vor
                            rc[n] = 1.0;
                            gc[n] = 0.0;
                            bc[n] = 1.0;
                            break;
                        default:
                            rc[n] = 1.0;
                            gc[n] = 1.0;
                            bc[n] = 1.0;
                            break;
                    }
                }
                else if (S == 2)
                {
                    // indenter
                    rc[n] = 0.5;
                    gc[n] = 0.5;
                    bc[n] = 0.5;
                }
                else
                {
                    rc[n] = 1.0;
                    gc[n] = 1.0;
                    bc[n] = 0.0;
                }
            }
            td->sphere->updateColors(rc, gc, bc);
            delete[] rc;
            delete[] gc;
            delete[] bc;
        }
        else if (format == IMWF)
        {
            float *r = new float[numParticles];
            for (int n = 0; n < numParticles; n++)
            {

                int T = Ivalues[1][n];
                if (T == 0)
                {
                    r[n] = 0.5;
                }
                else if (T == 1)
                {
                    r[n] = 1.0;
                }
                else
                {
                    r[n] = 2.0;
                }
            }
            td->sphere->updateRadii(r);
            delete[] r;

            float *rc = new float[numParticles];
            float *gc = new float[numParticles];
            float *bc = new float[numParticles];
            for (int n = 0; n < numParticles; n++)
            {
                int K = values[0][n];
                int T = Ivalues[1][n];
                if (T == 0)
                {
                    if (K == 12)
                    {
                        rc[n] = 0.5;
                        gc[n] = 0.5;
                        bc[n] = 1.0;
                    }
                    else if (K > 14)
                    {
                        rc[n] = 0.0;
                        gc[n] = 0.0;
                        bc[n] = 1.0;
                    }
                    else
                    {
                        rc[n] = 0.0;
                        gc[n] = 1.0;
                        bc[n] = 0.0;
                    }
                }
                else if (T == 1)
                {
                    if (K == 14)
                    {
                        rc[n] = 1.0;
                        gc[n] = 0.0;
                        bc[n] = 0.0;
                    }
                    if (K == 12)
                    {
                        rc[n] = 1.0;
                        gc[n] = 1.0;
                        bc[n] = 0.0;
                    }
                    else if (K > 14)
                    {
                        rc[n] = 1.0;
                        gc[n] = 0.0;
                        bc[n] = 1.0;
                    }
                    else
                    {
                        rc[n] = 1.0;
                        gc[n] = 0.5;
                        bc[n] = 0.0;
                    }
                }
                else
                {
                    if (K == 14)
                    {
                        rc[n] = 1.0;
                        gc[n] = 0.0;
                        bc[n] = 0.0;
                    }
                    if (K == 12)
                    {
                        rc[n] = 1.0;
                        gc[n] = 1.0;
                        bc[n] = 0.0;
                    }
                    else if (K > 14)
                    {
                        rc[n] = 1.0;
                        gc[n] = 0.0;
                        bc[n] = 1.0;
                    }
                    else
                    {
                        rc[n] = 1.0;
                        gc[n] = 0.5;
                        bc[n] = 0.0;
                    }
                }
            }
            td->sphere->updateColors(rc, gc, bc);
            delete[] rc;
            delete[] gc;
            delete[] bc;
        }
    }
}

template<typename T>
bool writeArr(const T *arr, size_t n, FILE *fp)
{
    int64_t sz = n;
    fwrite(&sz, sizeof(sz), 1, fp);
    return fwrite(arr, sizeof(T), n, fp) == n;
}

template<typename T>
ssize_t readArr(T *&arr, size_t n, FILE *fp)
{
    int64_t sz = 0;
    if(fread(&sz, sizeof(sz), 1, fp) != 1)
        return -1;
    if (n != sz)
        return -1;
    return fread(arr, sizeof(T), n, fp);
}

bool Particles::dump(std::string filename, int timestep, const float *xc, const float *yc, const float *zc) const
{
    const TimeStepData *td = timesteps[timestep];

    FILE *fp = fopen(filename.c_str(), "wb");
    if (!fp)
        return false;
    uint32_t endian=0x01020304;
    fwrite(&endian, sizeof(endian), 1, fp);
    int numParticles = td->numParticles;
    fwrite(&numParticles, sizeof(numParticles), 1, fp);
    fwrite(&numFloats, sizeof(numFloats), 1, fp);
    fwrite(&numInts, sizeof(numInts), 1, fp);

    writeArr(xc, numParticles, fp);
    writeArr(yc, numParticles, fp);
    writeArr(zc, numParticles, fp);

    for (int i=0; i<numFloats; ++i)
        writeArr(td->values[i], numParticles, fp);
    for (int i=0; i<numInts; ++i)
        writeArr(td->Ivalues[i], numParticles, fp);

    fclose(fp);
    return true;
}

bool Particles::restore(std::string filename, int timestep)
{
    bool ok = false;
    FILE *fp = fopen(filename.c_str(), "rb");
    if (!fp)
        return false;

    int numParticles = 0;
    float *xc=NULL, *yc=NULL, *zc=NULL;
    TimeStepData *td = NULL;
    osg::Geode *geode = NULL;
    coSphere *sphere = NULL;
    unsigned int ni=0, nf=0;

    uint32_t endian=0;
    fread(&endian, sizeof(endian), 1, fp);
    if (endian != 0x01020304)
        goto end;

    fread(&numParticles, sizeof(numParticles), 1, fp);
    fread(&nf, sizeof(nf), 1, fp);
    if (nf != numFloats)
        goto end;
    fread(&ni, sizeof(ni), 1, fp);
    if (ni != numInts)
        goto end;

    td = new TimeStepData(numParticles, numFloats, numInts);
    timesteps[timestep] = td;

    geode = new osg::Geode();
    geode->setName(filename);
    td->geode = geode;

    sphere = new coSphere();
    sphere->setRenderMethod(sphereMode);
    sphere->setMaxRadius(1);
    td->sphere = sphere;

    xc = new float[numParticles];
    yc = new float[numParticles];
    zc = new float[numParticles];
    readArr(xc, numParticles, fp);
    readArr(yc, numParticles, fp);
    readArr(zc, numParticles, fp);
    sphere->setCoords(numParticles, xc, yc, zc);
    delete[] xc;
    delete[] yc;
    delete[] zc;

    for (int i=0; i<numFloats; ++i)
        readArr(td->values[i], numParticles, fp);
    for (int i=0; i<numInts; ++i)
        readArr(td->Ivalues[i], numParticles, fp);

    //fprintf(stderr, "restored %d particles from %s\n", numParticles, filename.c_str());
    ok = true;
end:
    fclose(fp);
    return ok;
}
