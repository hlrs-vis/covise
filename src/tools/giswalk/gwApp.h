/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef GWAPP_H
#define GWAPP_H

#include "gwExport.h"

#include "giswalk.h"
#include "gwTier.h"
#include <vector>
#include <string>

class GWAPPEXPORT gwParamSet
{
public:
    gwParamSet();
    int directedWalkFromValue;
    int directedWalkFromStep;
    float streightness;
    int reorientation;
    int scanStartFromStep;
    float visionRange;
    int satisfactoryHabitat;
    float maxSpeed[MaxHabitatValues];
    int stopAtBoundary[MaxHabitatValues];
    int settlement;
};

class GWAPPEXPORT gwApp
{
private:
    std::list<gwTier *> tiere;
    std::list<gwTier *> toteTiere;
    unsigned char *map;

    std::string ending;
    std::string base;

    int minLifeTime;
    int maxLifeTime;
    float percentColonizers;
    double G;
    double mDist;
    int lastNumHValues;
    static gwApp *inst;

public:
    static gwApp *instance()
    {
        if (inst == NULL)
            inst = new gwApp(NULL);
        return inst;
    };
    int xs;
    int ys;
    float XLLCorner;
    float YLLCorner;
    int numHabitatValues;
    std::vector<double> sigmoDist;
    vec2 size;
    vec2 pSize;
    bool stopEveryTransition;
    float deathRate[MaxHabitatValues];
    float transitionMatrix[MaxHabitatValues][MaxHabitatValues];
    gwParamSet params[gwTier::NumTypes];
    gwApp(const char *filename);

    void addAnimal(int id, gwTier::Typ type, int lifetime, float xpos, float ypos);
    void initialize();

    void readConfigFile(std::string filename);

    unsigned char *readHDR(std::string filename);
    unsigned char *readTXT(std::string filename);
    void setMap(unsigned char *m, int xs, int ys, float cellSize);
    void readStartPoints(std::string filename);

    void writeSVG();
    void writeShape();
    void run();
    void singleStep();
    inline unsigned char getValue(vec2 &pos)
    {
        int xc = pos[0] / pSize[0];
        int yc = pos[1] / pSize[1];
        return (map[(xc + (yc * xs)) * 4]);
    };
    inline int getValue(int x, int y)
    {
        if (x >= 0 && x < xs && y >= 0 && y < ys)
        {
            return map[(x + (y * xs)) * 4];
        }
        return 300;
    };
    void getPos(vec2 &pos, int iPos[2])
    {
        iPos[0] = (pos[0] / pSize[0]);
        iPos[1] = (pos[1] / pSize[1]);
    };
};
#endif
