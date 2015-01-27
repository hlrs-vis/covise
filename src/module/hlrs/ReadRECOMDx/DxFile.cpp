/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <util/coviseCompat.h>
#include <api/coModule.h>

#include "DxFile.h"
#include "parser.h"

//search the position in the file
bool DxFile::seekEnd()
{
    //Data in dx files begin after an "end" tag
    //which is alone in a single line
    std::string line;
    while (!input_.eof())
    {
        std::getline(input_, line);
        std::transform(line.begin(), line.end(), line.begin(), tolower);

        size_t start = line.find_first_not_of(" ");
        if (start != std::string::npos)
        {

            size_t e = line.find("end", start);
            if (e != std::string::npos && line.find_first_not_of(" ", start + 3) == std::string::npos)
            {
                dataStart_ = input_.tellg();
                return true;
            }
        }
    }
    dataStart_ = 0;
    return false;
}

DxFile::DxFile(const char *filename, bool selfContained)
{
    selfContained_ = selfContained;
    input_.open(filename, ios::in);
    valid_ = !(!input_);
    if (!valid_)
    {
        Covise::sendError("file %s ccould not be opened", filename);
        return;
    }
    if (selfContained_)
    {
        seekEnd();
    }
    else
    {
        dataStart_ = 0;
    }
}

DxFile::~DxFile()
{
    input_.close();
}

void DxFile::setPos(int pos)
{
    input_.seekg(dataStart_ + pos);
}

void DxFile::readCoords(float *x_coord,
                        float xScale,
                        float *y_coord,
                        float yScale,
                        float *z_coord,
                        float zScale,
                        int offset,
                        int items,
                        int byteOrder)
{

    bool needsSwap = ((byteOrder == Parser::LSB) && machineIsBigEndian()) || ((byteOrder == Parser::MSB) && machineIsLittleEndian());

    int i;
    setPos(offset);
    for (i = 0; i < items; i++)
    {
        input_.read((char *)x_coord, sizeof(float));
        if (needsSwap)
        {
            byteSwap(*x_coord);
        }
        (*x_coord) *= xScale;
        x_coord++;
        input_.read((char *)y_coord, sizeof(float));
        if (needsSwap)
        {
            byteSwap(*y_coord);
        }
        (*y_coord) *= yScale;
        y_coord++;
        input_.read((char *)z_coord, sizeof(float));
        if (needsSwap)
        {
            byteSwap(*z_coord);
        }
        (*z_coord) *= zScale;
        z_coord++;
    }
}

void DxFile::readConnections(int *connections,
                             int offset,
                             int shape,
                             int items,
                             int byteOrder)
{

    bool needsSwap = ((byteOrder == Parser::LSB) && machineIsBigEndian()) || ((byteOrder == Parser::MSB) && machineIsLittleEndian());

    int i;
    setPos(offset);
    if (shape == 4)
    {
        int res[4];
        for (i = 0; i < items; i++)
        {
            input_.read((char *)res, 4 * sizeof(int));
            if (needsSwap)
            {
                byteSwap(res, 4);
            }
            //data explorer has another vertex ordering than covise
            *connections++ = res[0];
            *connections++ = res[1];
            *connections++ = res[3];
            *connections++ = res[2];
        }
    }
    else if (shape == 3)
    {
        int res[3];
        for (i = 0; i < items; i++)
        {
            input_.read((char *)res, 3 * sizeof(int));
            if (needsSwap)
            {
                byteSwap(res, 3);
            }

            *connections++ = res[0];
            *connections++ = res[1];
            *connections++ = res[2];
        }
    }
    else if (shape == 8)
    {
        int res[8];
        for (i = 0; i < items; i++)
        {
            input_.read((char *)res, 8 * sizeof(int));
            if (needsSwap)
            {
                byteSwap(res, 8);
            }
            //data explorer has another vertex ordering than covise
            *connections++ = res[0];
            *connections++ = res[1];
            *connections++ = res[3];
            *connections++ = res[2];
            *connections++ = res[4];
            *connections++ = res[5];
            *connections++ = res[7];
            *connections++ = res[6];
        }
    }
}

void DxFile::readData(float **data, int offset,
                      int shape, int items, int byteOrder, float &min, float &max)
{
    min = FLT_MAX;
    max = -FLT_MAX;

    bool needsSwap = ((byteOrder == Parser::LSB) && machineIsBigEndian()) || ((byteOrder == Parser::MSB) && machineIsLittleEndian());

    setPos(offset);
    int j, i;
    for (i = 0; i < items; i++)
    {
        for (j = 0; j < shape; j++)
        {
            input_.read((char *)&(data[j][i]), sizeof(float));
            if (needsSwap)
            {
                byteSwap(data[j][i]);
            }
            if (data[j][i] < min)
                min = data[j][i];
            if (data[j][i] > max)
                max = data[j][i];
        }
    }
}
