/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// **************************************************************************
//
//			Source File
//
// * Description    : Molecule plugin module for the Cover Covise Renderer
//                    Reads Molecule Structures based on the Jorgensen Model
//                    The data is provided from the Itt / University Stuttgart
//
// * Class(es)      :
//
// * inherited from :
//
// * Author  : Thilo Krueger
//
// * History : started 6.3.2001
//
// **************************************************************************

#include "Molecule.h"
#include <math.h>
#include <iostream>
#include <cstring>

MoleculeStructure::MoleculeStructure(FILE *datafile)
    : cubeSize(0) //size of cube of molecules
    , numberOfMolecules(0)
    , numberOfSites(0)
    , m_version(0)
{

    int state;
    memset(molIndex, 0, sizeof(molIndex));
    fp = datafile;
    readFile();

    //reset file pointer to beginning of file
    //important to read the rest of the file correctly
    state = fseek(fp, 0, SEEK_SET);
    if (state != 0)
        fprintf(stderr, "File read error!");
}

MoleculeStructure::~MoleculeStructure()
{
}

void MoleculeStructure::readFile()
{
    int siteCounter = 0;
    int i = 0;

    char line[LINESIZE];
    printf("reading molecule descriptions...\n");
    while (fgets(line, LINESIZE, fp) != NULL)
    {
        char *first = line;

        // skip blank lines
        while (*first != '\0' && strcmp(first, " ") == 0)
            first++;

        if (*first == '\0')
            //read the next line
            continue;

        if (*first == '~')
        {
            i++;
            first++;

            int numScanned = sscanf(first, " %d  LJ %f %f %f %f %d %d %d %d",
                                    &molIndex[i], &x[i], &y[i], &z[i], &sigma[i], &(colorRGBA[i][0]), &(colorRGBA[i][1]), &(colorRGBA[i][2]), &(colorRGBA[i][3]));
            printf("scanned #%d:\n", i);
            //additional format specification; RGBA values instead of a color index
            if (numScanned == 6)
            {
                color[i] = colorRGBA[i][0];
                printf("%d %f %f %f %f %d\n", molIndex[i], x[i], y[i], z[i], sigma[i], color[i]);
                //old version
                m_version = 0;
            }
            else if (numScanned == 8)
            {
                colorRGBA[i][3] = 255;
                printf("%d %f %f %f %f %d %d %d %d\n", molIndex[i], x[i], y[i], z[i], sigma[i], colorRGBA[i][0], colorRGBA[i][1], colorRGBA[i][2], colorRGBA[i][3]);
                m_version = 1;
            }
            else if (numScanned == 9)
            {
                printf("%d %f %f %f %f %d %d %d %d\n", molIndex[i], x[i], y[i], z[i], sigma[i], colorRGBA[i][0], colorRGBA[i][1], colorRGBA[i][2], colorRGBA[i][3]);
                //new version
                m_version = 1;
            }
            else if (numScanned != 6 && numScanned != 8 && numScanned != 9)
            { //ignore badly formed lines
                printf("badly formed line %d\n", numScanned);
                i--;
                continue;
            }

            siteCounter++;
            if (siteCounter == ATOMS)
            {
                printf("too many sites (max. %d)\n", ATOMS);
                break;
            }
        }

        if (*first == '#')
        {
            first++;

            if (sscanf(first, " %f", &cubeSize) != 1)
            {
                std::cerr << "MoleculeStructure::readFile: sscanf failed" << std::endl;
            }
            //we have all information, so we break the while-loop here
            break;
        }
    }

    numberOfMolecules = molIndex[i]; //not absolutely correct, will change later...
    numberOfSites = i;

    printf("done reading %d molecules\n", numberOfMolecules);
    printf("done reading %d numberOfSites\n", numberOfSites);

    return;
}

bool MoleculeStructure::getMolIDs(int type, std::vector<int> *iIDgroup)
{
    if (type < 1 || type > numberOfMolecules)
        return false;

    for (int atom = 0; atom <= numberOfSites; atom++)
    {
        if (molIndex[atom] == type)
        {
            (*iIDgroup).push_back(atom);
        }
    }
    return true;
}

bool MoleculeStructure::getmolIndex(int i, int *molIndex)
{
    if (i < 1 || i > numberOfSites)
        return false;
    else
    {
        *molIndex = this->molIndex[i];
        return true;
    }
}

int MoleculeStructure::getVersion()
{
    return m_version;
}

bool MoleculeStructure::getXYZ(int i, float *fX, float *fY, float *fZ)
{
    if (i < 1 || i > numberOfSites)
    {
        *fX = 0.0f;
        *fY = 0.0f;
        *fZ = 0.0f;
        printf("-%d-", i);
        return false;
    }
    else
    {
        *fX = this->x[i];
        *fY = this->y[i];
        *fZ = this->z[i];
        return true;
    }
}

bool MoleculeStructure::getSigma(int i, float *sigma)
{
    if (i < 1 || i > numberOfSites)
        return false;
    else
    {
        *sigma = this->sigma[i];
        return true;
    }
}

bool MoleculeStructure::getColor(int i, int *color)
{
    if (i < 1 || i > numberOfSites || m_version == 1)
        return false;
    else
    {
        //   printf("i: %d\n", i);
        *color = this->color[i];
        return true;
    }
}

bool MoleculeStructure::getColorRGBA(int i, int *fR, int *fG, int *fB, int *fA)
{
    if (i < 1 || i > numberOfSites || m_version == 0)
        return false;
    else
    {
        *fR = colorRGBA[i][0];
        *fG = colorRGBA[i][1];
        *fB = colorRGBA[i][2];
        *fA = colorRGBA[i][3];
        return true;
    }
}

float MoleculeStructure::getBoxSize()
{
    return cubeSize;
}

int MoleculeStructure::getNumberOfMolecules()
{
    return numberOfMolecules;
}

int MoleculeStructure::getNumberOfSites()
{
    return numberOfSites;
}
