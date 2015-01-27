/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TRACELINEEXTRACTOR_H_
#define TRACELINEEXTRACTOR_H_
/****************************************************************************\ 
**                                                            (C)1999 RUS   **
**                                                                          **
** Description:												             **
**																			 **
**																			 **
**                                                                          **
** Name:																	 **
** Category:																 **
**                                                                          **
** Author: D. Rainer		                                                 **
**                                                                          **
** History:  								                                 **
** September-99  					       		                             **
**                                                                          **
**                                                                          **
\****************************************************************************/

#include <map>
#include <list>
#include <api/coModule.h>
using namespace covise;
#include <do/coDoSet.h>
//#include <do/coDoLines.h>

#define NODES 1
#define TRACELINES 2
#define DOFS 4

#define ALL 7 //sum of the others

using namespace std;

static const char *functionNames[] = {
    "0 - General or Unknown",
    "1 - Time Response",
    "2 - Auto Spectrum",
    "3 - Cross Spectrum",
    "4 - Frequency Response Function",
    "5 - Transmissibility",
    "6 - Coherence",
    "7 - Auto Correlation",
    "8 - Cross Correlation",
    "9 - Power Spectral Density (PSD)",
    "10 - Energy Spectral Density (ESD)",
    "11 - Probability Density Function",
    "12 - Spectrum",
    "13 - Cumulative Frequency Distribution",
    "14 - Peaks Valley",
    "15 - Stress/Cycles",
    "16 - Strain/Cycles",
    "17 - Orbit",
    "18 - Mode Indicator Function",
    "19 - Force Pattern",
    "20 - Partial Power",
    "21 - Partial Coherence",
    "22 - Eigenvalue",
    "23 - Eigenvector",
    "24 - Shock Response Spectrum",
    "25 - Finite Impulse Response Filter",
    "26 - Multiple Coherence",
    "27 - Order Function"
};

#define PI 3.14159265

class TraceLineExtractor : public coModule
{

private:
    /*	struct ltstr				//for maps with char* keys
	{
		bool operator()(const char* s1, const char* s2) const
		{
			return strcmp(s1, s2) < 0;
		}
	};
  */

    struct sPoint
    {
        union
        {
            int defcosysnum; //dataset 15   : definition coordinate system number
            int expcosysnum; //dataset 2411 : export coordinate system number
        };
        int discosysnum; //displacement coordinate system number
        int color; //should also be _int64 but probably not necessary
        float coords[3]; //coordinate p[0] = x, p[1] = y, p[2] = z

        sPoint(int def, int dis, int col, float x, float y, float z)
        {
            defcosysnum = def;
            discosysnum = dis;
            color = col;
            coords[0] = x;
            coords[1] = y;
            coords[2] = z;
        }
    };

    struct sTraceLine
    {
        char name[80];
        unsigned int numLines;
        unsigned int numVertices;
        int *indices; //a list of indices
        //		int*			vertList;	//a list of vertices
    };

    //	map<unsigned int, sPoint*>		nodeMap;					//<node label, point>
    map<unsigned int, sTraceLine *> traceLineMap; //<line number, line specification>
    map<unsigned int, unsigned int> substituteMap; //substitutes node labels(from the file) with float array positions (coordinates)
    //also used, if a .map file is specified

    map<short, char *> functionTypeNamesMap; //map for function type names at nodal DOFs dataset
    map<short, list<unsigned int> > functionTypeSets; //map which contains a list, in which are all dataset numbers (of dataset58) of
    //one specific function are stored

    int *vertList; //TODO: obsolete
    int *lineList; //TODO: obsolete

    float **coordinates; //it contains the models x, y and z values
    unsigned int numCoordinates; //the size of the coordinates array

    int *cornerList;
    int numCorners;

    int *typeList;
    int numTypes;

    int *elementList;
    int numElements;

    bool updateNodesInput; //force update for nodes input
    bool updateTraceLineInput; //force traceLineIndices input
    bool updateDOFsInput; //force update for DOFs input

    short selectedFunction;

    char **choice_traceIDText; //traceline names
    int numChoice_traceID; //size of traceline list

    char **choice_functionTypeText; //function names in the function choice param
    int numChoice_functionTypeText; //size of the functionTypeText list

    bool use_mapFile; //true if a map file is used

    //TODO: add coordination and function lists

    coInputPort *nodes;
    coInputPort *traceLineIndices;
    coInputPort *DOFs;
    coInputPort *datasetFileheader; //fileheader
    coInputPort *datasetUnitDesc; //unit description

    coOutputPort *traceLine;
    coOutputPort *traceLineData;

    coChoiceParam *choice_traceLineID;
    coChoiceParam *choice_coordDir; //choose the coordinate direction, or "all directions"
    coChoiceParam *choice_functionType; //choose the specific function to show

    coIntSliderParam *dataPosSlider; //Slider for selecting the frequency
    coIntSliderParam *animationStepsSlider;
    coFloatSliderParam *multiplierSlider; //Slider for selecting the frequency

    coStringParam *traceLineSelection; //enter number of tracelines, you want to show, e.g. 0-6;8

#ifdef WRITE_TO_FILE_ENABLE
    coBooleanParam *choice_writeToFile; //if set, the current choice of tracelines will be written to a file
    coFileBrowserParam *file_writeToFileName; //name of the file, in which data will be written
#endif

    coBooleanParam *choice_useMapFile;
    coFileBrowserParam *file_mapFile; //name of the .map file

    //  member functions
    virtual int compute(const char *port);
    virtual void param(const char *name, bool inMapLoading);
    virtual void postInst();
    virtual void quit(/* const char* */);

    void UnpackDataset(coDoSet *set);
    void writeDatasetToFile(FILE *f, coDoSet *dataset);
    void Clean(int what);
    //	void			calcMatrix(float **mat, float tx, float ty, float tz, float rx, float ry, float rz); //(re)calculate a matrix for translation, and rotation
    int readMapFile(const char *filename);
    list<unsigned int> getSelectedTracelines();

public:
    TraceLineExtractor(int argc, char *argv[]);
    virtual ~TraceLineExtractor();
};
#endif
