/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READNEURON_H
#define _READNEURON_H

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                      (C)2005 HLRS   ++
// ++ Description: ReadNeuron module                                      ++
// ++                                                                     ++
// ++ Author:  Uwe                                                        ++
// ++                                                                     ++
// ++                                                                     ++
// ++ Date:  12.2005                                                      ++
// ++**********************************************************************/
#include <math.h>
#include <api/coSimpleModule.h>
using namespace covise;
#include <vector>

#include <util/coviseCompat.h>
#include <do/coDoData.h>
#include <do/coDoSet.h>

#define LINE_SIZE 1024
#define SO_LINE_SIZE 2048
#define MAXTIMESTEPS 2048

class ReadNeuron : public coSimpleModule
{
    COMODULE

public:
    ReadNeuron(int argc, char **argv);

private:
    //////////  inherited member functions
    virtual int compute(const char *port);
    void getLine();
    void getPoint(int i, float fraction, float &x, float &y, float &z);

    void readLine(int numSegments);
    char bfr[2048];
    char line[SO_LINE_SIZE];
    char oldLine[SO_LINE_SIZE];

    int numPoints;
    int numSomaPoints;
    int numLines;
    int numSomaLines;
    int numSomas;
    int numSynapses;
    int ID;
    int synID;

    int timesteps;
    list<coDoFloat *> somaDataList;
    list<coDoFloat *>::iterator somaData_Iter;

    std::vector<int> lineList;
    std::vector<int> somalineList;
    std::vector<float> rad;
    std::vector<float> somaRad;
    std::vector<float> IDs;
    std::vector<float> synIDs;
    std::vector<float> xPoints;
    std::vector<float> yPoints;
    std::vector<float> zPoints;

    std::vector<float> somaData;

    std::vector<float> xSomaP;
    std::vector<float> ySomaP;
    std::vector<float> zSomaP;

    std::vector<float> pX;
    std::vector<float> pY;
    std::vector<float> pZ;
    std::vector<char *> names;

    FILE *file;
    FILE *somafile;
    char *c;
    ////////// ports
    coOutputPort *m_portLines;
    coOutputPort *m_portRad;
    coOutputPort *m_portID;
    coOutputPort *m_portSoLines;
    coOutputPort *m_portSoRad;
    coOutputPort *m_portSoData;
    coOutputPort *m_portSynapses;
    coOutputPort *m_portSynID;

    ///////// params
    coFileBrowserParam *m_pParamFile;
    coFileBrowserParam *m_pSomaFile;
    coIntScalarParam *numt;

    ////////// member variables;
    int maxNumberOfMolecules;
    int numberOfTimesteps;
    char *m_filename;
    char *m_somafilename;
};
#endif
