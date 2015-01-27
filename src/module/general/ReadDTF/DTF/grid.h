/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __DTF_GRID_H_
#define __DTF_GRID_H_

#include "../Tools/classmanager.h"

namespace DTF
{
class ClassInfo_DTFGrid;

class Grid : public Tools::BaseObject
{
    friend class ClassInfo_DTFGrid;

private:
    vector<float> x, y, z;
    vector<int> cornerList;
    vector<int> elementList;
    vector<int> typeList;
    float *coordX, *coordY, *coordZ;
    int *corners, *elements, *types;

    Grid();
    Grid(string className, int objectID);

    static ClassInfo_DTFGrid classInfo;

    bool addCoords(Grid *grid, int &coordOffset);
    bool addCorners(Grid *grid, int coordOffset, int &cornerOffset);
    bool addElements(Grid *grid, int cornerOffset);
    bool addTypes(Grid *grid);

public:
    virtual ~Grid();

    bool setCoordList(vector<float> xCoords,
                      vector<float> yCoords,
                      vector<float> zCoords);

    bool setCornerList(vector<int> corners);
    bool setElementList(vector<int> elements);
    bool setTypeList(vector<int> types);

    int getNumCoords();
    int getNumCorners();
    int getNumElements();
    int getNumTypes();

    float *getCoordX();
    float *getCoordY();
    float *getCoordZ();

    int *getCornerList();
    int *getElementList();
    int *getTypeList();

    virtual void print();
    virtual void clear();

    bool addData(Grid *grid);
};

CLASSINFO(ClassInfo_DTFGrid, Grid);
};
#endif
