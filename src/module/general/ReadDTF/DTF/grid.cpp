/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "grid.h"

using namespace DTF;

CLASSINFO_OBJ(ClassInfo_DTFGrid, Grid, "DTF::Grid", INT_MAX);

Grid::Grid()
    : Tools::BaseObject()
{
    coordX = NULL;
    coordY = NULL;
    coordZ = NULL;
    corners = NULL;
    elements = NULL;
    types = NULL;

    INC_OBJ_COUNT(getClassName());
}

Grid::Grid(string className, int objectID)
    : Tools::BaseObject(className, objectID)
{
    coordX = NULL;
    coordY = NULL;
    coordZ = NULL;
    corners = NULL;
    elements = NULL;
    types = NULL;

    INC_OBJ_COUNT(getClassName());
}

Grid::~Grid()
{
    clear();
    DEC_OBJ_COUNT(getClassName());
}

bool Grid::setCoordList(vector<float> xCoords,
                        vector<float> yCoords,
                        vector<float> zCoords)
{
    if (xCoords.empty() || yCoords.empty() || zCoords.empty())
        return false;

    this->x = xCoords;
    this->y = yCoords;
    this->z = zCoords;
    return true;
}

bool Grid::setCornerList(vector<int> corners)
{
    if (corners.empty())
        return false;

    this->cornerList = corners;

    return true;
}

bool Grid::setElementList(vector<int> elemList)
{
    if (elemList.empty())
        return false;

    this->elementList = elemList;

    return true;
}

bool Grid::setTypeList(vector<int> types)
{
    if (types.empty())
        return false;

    typeList = types;

    return true;
}

int Grid::getNumCoords()
{
    return this->x.size();
}

int Grid::getNumCorners()
{
    return this->cornerList.size();
}

int Grid::getNumElements()
{
    return this->elementList.size();
}

int Grid::getNumTypes()
{
    return this->typeList.size();
}

float *Grid::getCoordX()
{
    if (!this->x.empty())
    {
        if (this->coordX != NULL)
        {
            delete[] this->coordX;
            this->coordX = NULL;
        }

        this->coordX = new float[this->x.size()];

        for (unsigned int i = 0; i < this->x.size(); i++)
            this->coordX[i] = this->x[i];

        return this->coordX;
    }
    else
        return NULL;
}

float *Grid::getCoordY()
{
    if (!this->y.empty())
    {
        if (this->coordY != NULL)
        {
            delete[] this->coordY;
            this->coordY = NULL;
        }

        this->coordY = new float[this->y.size()];

        for (unsigned int i = 0; i < this->y.size(); i++)
            this->coordY[i] = this->y[i];

        return this->coordY;
    }
    else
        return NULL;
}

float *Grid::getCoordZ()
{
    if (!this->z.empty())
    {
        if (this->coordZ != NULL)
        {
            delete[] this->coordZ;
            this->coordZ = NULL;
        }

        this->coordZ = new float[this->z.size()];

        for (unsigned int i = 0; i < this->z.size(); i++)
            this->coordZ[i] = this->z[i];

        return this->coordZ;
    }
    else
        return NULL;
}

int *Grid::getCornerList()
{
    if (!this->cornerList.empty())
    {
        if (this->corners != NULL)
        {
            delete[] this->corners;
            this->corners = NULL;
        }

        this->corners = new int[this->cornerList.size()];

        for (unsigned int i = 0; i < this->cornerList.size(); i++)
            this->corners[i] = this->cornerList[i];

        return this->corners;
    }
    else
        return NULL;
}

int *Grid::getElementList()
{
    if (!this->elementList.empty())
    {
        if (this->elements != NULL)
        {
            delete[] this->elements;
            this->elements = NULL;
        }
        this->elements = new int[this->elementList.size()];

        for (unsigned int i = 0; i < this->elementList.size(); i++)
            this->elements[i] = this->elementList[i];

        return this->elements;
    }
    else
        return NULL;
}

int *Grid::getTypeList()
{
    if (!this->typeList.empty())
    {
        if (this->types != NULL)
        {
            delete[] this->types;
        }

        this->types = new int[this->typeList.size()];

        for (unsigned int i = 0; i < this->typeList.size(); i++)
            this->types[i] = this->typeList[i];

        return this->types;
    }
    else
        return NULL;
}

void Grid::print()
{
    cout << "coordinates: " << endl;
    cout << "-------------" << endl;

    for (unsigned int i = 0; i < this->x.size(); i++)
    {
        cout << i << ": " << x[i] << ", " << y[i] << ", " << z[i] << endl;
    }

    cout << "corner list: " << endl;
    cout << "-------------" << endl;

    for (unsigned int i = 0; i < this->cornerList.size(); i++)
        cout << i << ": " << cornerList[i] << endl;

    cout << "element list: " << endl;
    cout << "--------------" << endl;

    for (unsigned int i = 0; i < this->elementList.size(); i++)
        cout << i << ": " << elementList[i] << endl;

    cout << "type list: " << endl;
    cout << "-----------" << endl;

    for (unsigned int i = 0; i < this->typeList.size(); i++)
        cout << i << ": " << typeList[i] << endl;
}

bool Grid::addData(Grid *grid)
{
    int coordOffset, cornerOffset;
    coordOffset = cornerOffset = 0;

    if (grid != NULL)
    {
        if (addCoords(grid, coordOffset))
            if (addCorners(grid, coordOffset, cornerOffset))
                if (addElements(grid, cornerOffset))
                    if (addTypes(grid))
                        return true;
    }

    return false;
}

void Grid::clear()
{
    if (coordX != NULL)
    {
        delete[] coordX;
        coordX = NULL;
    }
    if (coordY != NULL)
    {
        delete[] coordY;
        coordY = NULL;
    }
    if (coordZ != NULL)
    {
        delete[] coordX;
        coordX = NULL;
    }

    if (corners != NULL)
    {
        delete[] corners;
        corners = NULL;
    }
    if (elements != NULL)
    {
        delete[] elements;
        elements = NULL;
    }
    if (types != NULL)
    {
        delete[] types;
        types = NULL;
    }

    x.clear();
    y.clear();
    z.clear();
    cornerList.clear();
    elementList.clear();
    typeList.clear();
}

bool Grid::addCoords(Grid *grid, int &coordOffset)
{
    if (!grid->x.empty() && !grid->y.empty() && !grid->z.empty())
    {
        int coordOffset = this->x.size();

        this->x.resize(coordOffset + grid->x.size(), 0.0);
        this->y.resize(coordOffset + grid->y.size(), 0.0);
        this->z.resize(coordOffset + grid->z.size(), 0.0);

        for (unsigned int i = 0; i < grid->x.size(); i++)
        {
            this->x[i + coordOffset] = grid->x[i];
            this->y[i + coordOffset] = grid->y[i];
            this->z[i + coordOffset] = grid->z[i];
        }

        return true;
    }

    return false;
}

bool Grid::addCorners(Grid *grid, int coordOffset, int &cornerOffset)
{
    if (!grid->cornerList.empty())
    {
        cornerOffset = this->cornerList.size();

        for (unsigned int i = 0; i < grid->cornerList.size(); i++)
            this->cornerList.push_back(grid->cornerList[i] + coordOffset);

        return true;
    }

    return false;
}

bool Grid::addElements(Grid *grid, int cornerOffset)
{
    if (!grid->elementList.empty())
    {
        for (unsigned int i = 0; i < grid->elementList.size(); i++)
            this->elementList.push_back(grid->elementList[i] + cornerOffset);

        return true;
    }

    return false;
}

bool Grid::addTypes(Grid *grid)
{
    if (!grid->typeList.empty())
    {
        this->typeList.insert(this->typeList.end(),
                              grid->typeList.begin(),
                              grid->typeList.end());

        return true;
    }

    return false;
}
