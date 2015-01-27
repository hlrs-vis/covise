/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef GWTIER_H
#define GWTIER_H

#include <util/coExport.h>
#include "giswalk.h"

#ifdef WIN32
#ifdef GWAPP_EXPORT
#define GWAPPEXPORT COEXPORT
#else
#define GWAPPEXPORT COIMPORT
#endif
#else
#define GWAPPEXPORT
#endif

class GWAPPEXPORT gwTier
{
public:
    enum Typ
    {
        Philopatric = 0,
        Colonizers = 1,
        NumTypes
    };
    enum Motion
    {
        DirectedWalk,
        Died,
        RandomWalk
    };

private:
    Typ type;
    Typ origType;
    int lifeTime;
    int stepNum;
    int stepInHabitat;
    int reorientStep;
    vec2 pos;
    float direction;
    int oldHabitatValue;
    int currentHabitatValue;
    gwApp *app;
    Motion currentState;
    std::vector<vec2> path;
    int group;
    int id;

public:
    gwTier(int i, Typ t, gwApp *a, int group = 0);
    Motion move();
    Motion moveOld();
    int checkForNewHabitat(vec2 v);
    float getDir(int x, int y);
    void scan();
    void writeSVG(FILE *fp);
    void writeShape(FILE *fp);
    void setPos(float x, float y);
    void setLifeTime(int lt)
    {
        lifeTime = lt;
    };
    void setType(Typ t)
    {
        type = t;
        origType = t;
    };
    Typ getType()
    {
        return type;
    };
};

#endif
