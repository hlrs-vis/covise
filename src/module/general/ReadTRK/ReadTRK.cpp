/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//Header
#include "ReadTRK.h"
#include <iostream>
#include <string>
#include <sstream>
#include <do/coDoSet.h>
#include <boost/tokenizer.hpp>
const char* wschars = " \t\n\r\f\v{}[]";

// trim from end of string (right)
inline std::string& rtrim(std::string& s, const char* t = wschars)
{
    s.erase(s.find_last_not_of(t) + 1);
    return s;
}

// trim from beginning of string (left)
inline std::string& ltrim(std::string& s, const char* t = wschars)
{
    s.erase(0, s.find_first_not_of(t));
    return s;
}

// trim from both ends of string (right then left)
inline std::string& trim(std::string& s, const char* t = wschars)
{
    return ltrim(rtrim(s, t), t);
}

//CONSTRUCTOR
ReadTRK::ReadTRK(int argc, char *argv[])
    : coModule(argc, argv, "Read StarCCM TRK file")
{
    //PORTS:
    poLines = addOutputPort("TrackLines", "Lines", "Lines");

    poParticleResidenceTime = addOutputPort("ParticleResidenceTime", "Float", "poParticleResidenceTime");
    poVelocity = addOutputPort("Velocity", "Vec3", "Velocity");
    poParticleFlowRate = addOutputPort("ParticleFlowRate", "Float", "ParticleFlowRate");
    poParticleDiameter = addOutputPort("ParticleDiameter", "Float", "ParticleDiameter");


    //PARAMETERS:
    trkFilePath = addFileBrowserParam("FilePath", "Path of series image files including printf format string");
    trkFilePath->setValue(" ", "*.TRK;*.trk");
    trkFilePath->show();

    return;
}

//DESTRUCTOR
ReadTRK::~ReadTRK()
{
}

//COMPUTE-ROUTINE
int ReadTRK::compute(const char *)
{
    numChunks = 0;
    fp = fopen(trkFilePath->getValue(), "rb");

    dosParticleResidenceTime.clear();
    dosVelocity.clear();
    dosParticleFlowRate.clear();
    dosParticleDiameter.clear();
    dosLines.clear();

    while (!feof(fp))
    {
        if (readChunk() == false)
            break;
    }
    coDistributedObject** dos = new coDistributedObject*[dosParticleResidenceTime.size()+1];
    int i = 0;
    dos[0] = nullptr;
    for (const auto& d : dosParticleResidenceTime)
    {
        dos[i] = d;
        dos[i+1] = nullptr;
        i++;
    }
    coDoSet* dataObject = new coDoSet(poParticleResidenceTime->getObjName(), dos);
    delete[] dos;

    dos = new coDistributedObject * [dosVelocity.size()+1];
    i = 0;
    dos[0] = nullptr;
    for (const auto& d : dosVelocity)
    {
        dos[i] = d;
        dos[i + 1] = nullptr;
        i++;
    }
    dataObject = new coDoSet(poVelocity->getObjName(), dos);
    delete[] dos;

    dos = new coDistributedObject * [dosParticleFlowRate.size()+1];
    i = 0;
    dos[0] = nullptr;
    for (const auto& d : dosParticleFlowRate)
    {
        dos[i] = d;
        dos[i + 1] = nullptr;
        i++;
    }
    dataObject = new coDoSet(poParticleFlowRate->getObjName(), dos);
    delete[] dos;

    dos = new coDistributedObject * [dosParticleDiameter.size() + 1];
    i = 0;
    dos[0] = nullptr;
    for (const auto& d : dosParticleDiameter)
    {
        dos[i] = d;
        dos[i + 1] = nullptr;
        i++;
    }
    dataObject = new coDoSet(poParticleDiameter->getObjName(), dos);
    delete[] dos;

    dos = new coDistributedObject * [dosLines.size()+1];
    i = 0;
    dos[0] = nullptr;
    for (const auto& d : dosLines)
    {
        dos[i] = d;
        dos[i + 1] = nullptr;
        i++;
    }
    dataObject = new coDoSet(poLines->getObjName(), dos);
    delete[] dos;

    fclose(fp);


    return EXIT_SUCCESS;
}
std::string ReadTRK::genName(const std::string& bn)
{
    return bn + to_string(numChunks);
}

bool ReadTRK::readChunk()
{

    char buf[10000];
    buf[0] = '\0';
    char *res = fgets(buf, sizeof(buf), fp);
    if (res ==nullptr || feof(fp) || buf[0]=='\0')
    {
        return false;
    }
    size_t bufferSize = strlen(buf);
    std::string header = buf;
    int numPoints = 0;
    DataField df;
    std::string name;
    dataFields.clear();

    boost::tokenizer<boost::escaped_list_separator<char>> tokens(header);
    for (const auto& t : tokens)
    {
        boost::char_separator<char> sep(":");
        boost::tokenizer<boost::char_separator<char>> tokens(t, sep);
        if (tokens.begin() != tokens.end())
        {
            boost::tokenizer<boost::char_separator<char>>::const_iterator it = tokens.begin();

            std::string s1 = *(it);
            trim(s1);
            if (s1 == "'Variables'")
            {

                std::string s2 = *(++it);
                s1 = s2;
                trim(s1);
            }
            if (s1 == "'Size'")
            {

                std::string s2 = *(++it);
                sscanf(s2.c_str(), "%d", &numPoints);
            }
            if (s1 == "'DataSize'")
            {

                std::string s2 = *(++it);
                sscanf(s2.c_str(), "%d", &df.size);
            }
            if (s1 == "'Name'")
            {

                std::string s2 = *(++it);
                trim(s2);
                trim(s2, "'");
                df.name = s2;
            }
            if (s1 == "'DataType'")
            {

                std::string s2 = *(++it);
                trim(s2);
                if (s2 == "'Vector<3")
                    df.dataType = VEC3D;
                if (s2 == "'Float8'")
                    df.dataType = DOUBLE;
                if (s2 == "'Unsigned4'")
                    df.dataType = UNSIGNED_INT;
            }
            if (s1 == "'Type'")
            {
                std::string s2 = *(++it);
                sscanf(s2.c_str(), "%d", &df.type);
                dataFields.push_back(df);
            }

        }
    }
    if (numPoints == 0)
        return false;
    double* positions = nullptr;
    uint32_t* index = nullptr;
    uint32_t maxIndex = 0;
    float* residenceTime=nullptr;
    //uint32_t minIndex = 0xffffffff;
    for (const auto& d : dataFields)
    {
        if (d.name == "Position")
        {
            char* b = new char[numPoints * d.size];
            positions = (double*)b;
            fread(b, numPoints, d.size, fp);
        }
        if (d.name == "Parcel Index")
        {
            char* b = new char[numPoints * d.size];
            index = (uint32_t*)b;
            fread(b, numPoints, d.size, fp);
            for (int i = 0; i < numPoints; i++)
            {
                if (index[i] > maxIndex)
                    maxIndex = index[i];
               // else if (index[i] < minIndex)
               //     minIndex = index[i];
            }
            maxIndex++;
        }
        if (d.name == "Particle Residence Time")
        {
            char* b = new char[numPoints * d.size];
            double* data = (double*)b;
            fread(b, numPoints, d.size, fp);
            coDoFloat* dataObject = new coDoFloat(genName(poParticleResidenceTime->getObjName()), numPoints);
            dosParticleResidenceTime.push_back(dataObject);
            poParticleResidenceTime->setCurrentObject(dataObject);
            residenceTime = dataObject->getAddress();
            for (int i = 0; i < numPoints; i++)
                residenceTime[i] = (float)data[i];
            delete[] data;
        }
        if (d.name == "Particle Velocity")
        {
            char* b = new char[numPoints * d.size];
            double* data = (double*)b;
            fread(b, numPoints, d.size, fp);
            coDoVec3* dataObject = new coDoVec3(genName(poVelocity->getObjName()), numPoints);
            dosVelocity.push_back(dataObject);
            poVelocity->setCurrentObject(dataObject);
            float* fpx;
            float* fpy;
            float* fpz;
            dataObject->getAddresses(&fpx, &fpy, &fpz);
            for (int i = 0; i < numPoints; i++)
            {
                fpx[i] = (float)data[i * 3];
                fpy[i] = (float)data[i * 3 + 1];
                fpz[i] = (float)data[i * 3 + 2];
            }
            delete[] data;
        }
        if (d.name == "Particle Flow Rate")
        {
            char* b = new char[numPoints * d.size];
            double* data = (double*)b;
            fread(b, numPoints, d.size, fp);
            coDoFloat* dataObject = new coDoFloat(genName(poParticleFlowRate->getObjName()), numPoints);
            dosParticleFlowRate.push_back(dataObject);
            poParticleFlowRate->setCurrentObject(dataObject);
            float* fp = dataObject->getAddress();
            for (int i = 0; i < numPoints; i++)
                fp[i] = (float)data[i];
            delete[] data;
        }
        if (d.name == "Particle Diameter")
        {
            char* b = new char[numPoints * d.size];
            double* data = (double*)b;
            fread(b, numPoints, d.size, fp);
            coDoFloat* dataObject = new coDoFloat(genName(poParticleDiameter->getObjName()), numPoints);
            dosParticleDiameter.push_back(dataObject);
            poParticleDiameter->setCurrentObject(dataObject);
            float* fp = dataObject->getAddress();
            for (int i = 0; i < numPoints; i++)
                fp[i] = (float)data[i];
            delete[] data;
        }
    }

    std::vector<int> lineIndex(maxIndex, 0);


    int numLines = 0;
    for (int i = 0; i < numPoints; i++)
    {
        if (lineIndex[index[i]] == 0)
        {
            lineIndex[index[i]] = numLines;
            numLines++;
        }

    }
    std::vector<int> lineLength(numLines, 0);
    std::vector<double> lastIndex(numLines, 0.0);
    int numSplits = 0;
    //split lines with gaps
    if (residenceTime != nullptr)
    {
        for (int i = 0; i < numPoints; i++)
        {
            uint32_t line = lineIndex[index[i]];
            if (lastIndex[line] > 0 && ( lastIndex[line] < residenceTime[i] - 0.01 || lastIndex[line] > residenceTime[i]))
            {
                // splitLine
                uint32_t lastID = index[i];
                for (int ii = i; ii < numPoints; ii++)
                {
                    if (index[ii] == lastID)
                        index[ii] = maxIndex;
                }
                lineIndex.push_back(numLines);
                maxIndex++;
                lineLength.push_back(0);
                lastIndex.push_back(residenceTime[i]);
                numLines++;
                numSplits++;
            }
            lastIndex[line] = residenceTime[i];

        }
    }
    fprintf(stderr, "numSplits %d, chunk: %d\n", numSplits, numChunks);

    for (int i = 0; i < numPoints; i++)
    {
        uint32_t line = lineIndex[index[i]];
        lineLength[line]++;
    }


    float* xc, * yc, * zc;
    int* ll, * vl;
    coDoLines* lines = new coDoLines(genName(poLines->getObjName()), numPoints, numPoints, numLines);
    dosLines.push_back(lines);
    lines->getAddresses(&xc, &yc, &zc, &vl, &ll);
    for (int i = 0; i < numPoints; i++)
    {
        xc[i] = (float)positions[i * 3];
        yc[i] = (float)positions[i * 3 + 1];
        zc[i] = (float)positions[i * 3 + 2];
    }
    int v = 0;
    for (size_t l = 0; l < lineLength.size(); l++)
    {
        ll[l] = v;
        v += lineLength[l];
    }
    std::fill(lineLength.begin(), lineLength.end(), 0);
    for (int i = 0; i < numPoints; i++)
    {
        uint32_t line = lineIndex[index[i]];
        vl[ll[line] + lineLength[line]] = i;
        lineLength[line]++;
    }
    poLines->setCurrentObject(lines);

    delete[] positions;
    delete[] index;
    numChunks++;

    fgets(buf, sizeof(buf), fp);
    return true;
}

MODULE_MAIN(IO, ReadTRK)
