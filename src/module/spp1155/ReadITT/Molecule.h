/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// **************************************************************************
//
//			.h File
//
// * Description    : Molecule classes for common use in covise's mapeditor
//                    Reads Molecule Structures based on the Jorgensen Model
//                    The data is provided from the Itt / University Stuttgart
//
// * Class(es)      : MoleculeStructure, Frame
//
// * inherited from :
//
// * Author  : Thilo Krueger
// * revised by: Thomas van Reimersdahl, Lehrstuhl fr Informatik,
//               Uni Koeln, (vr@uni-koeln.de)
//
// * History : started 28.5.2001
//   revision: 01.2005
//
// **************************************************************************

#ifndef _MOLECULE_H
#define _MOLECULE_H
#define LINESIZE 128
#define ATOMS 32 // maximum nuber of sites in all molecules
#define MAGNIFICATION 10 // magnification factor for molecules

#include <vector>
#include <cstdio>
class MoleculeStructure
{
private:
    // ITTs data structure
    int molIndex[ATOMS]; //index number of molecule
    float x[ATOMS];
    float y[ATOMS];
    float z[ATOMS];
    float sigma[ATOMS]; //size
    int color[ATOMS];
    int colorRGBA[ATOMS][4];
    float cubeSize; //size of cube of molecules
    int numberOfMolecules;
    int numberOfSites;
    int m_version;

    FILE *fp;

    // member functions
    void readFile();

public:
    //constructor
    MoleculeStructure(FILE *fp);

    //destructor
    ~MoleculeStructure();

    bool getMolIDs(int i, std::vector<int> *iIDgroup);

    bool getmolIndex(int i, int *molIndex);
    bool getXYZ(int i, float *fX, float *fY, float *fZ);
    bool getSigma(int i, float *sigma);
    bool getColor(int i, int *color);
    bool getColorRGBA(int i, int *fR, int *fG, int *fB, int *fA);
    float getBoxSize();
    int getNumberOfMolecules();
    int getNumberOfSites();
    int getVersion();
};

#endif
