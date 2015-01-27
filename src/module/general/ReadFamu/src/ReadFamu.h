/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                     (C)2005 HLRS   ++
// ++ Description: ReadFamu module                                       ++
// ++                                                                    ++
// ++ Author:  Uwe                                                       ++
// ++                                                                    ++
// ++                                                                    ++
// ++ Date:  6.2006                                                      ++
// ++**********************************************************************/

#ifndef ___ReadFamu_h__
#define ___ReadFamu_h__

#include <math.h>
#include <api/coSimpleModule.h>
using namespace covise;
#include <string>
#include "OutputHandler.h" // an output handler for displaying information on the screen.
#include "MeshFileTransParser.h" // a mesh file parser.
#include "ResultsFileParser.h" // a results file parser.

#include <util/coviseCompat.h>
#include <vector>

//#define __MEM_LEAK__

class ReadFamu : public coSimpleModule, public OutputHandler
{
public:
    ReadFamu(int argc, char **argv);
    virtual ~ReadFamu();

private:
    // ports
    coOutputPort *_mesh;
    coOutputPort *_dataPort[NUMRES];

    // params for GUI
    coFileBrowserParam *_meshFileName;
    coFileBrowserParam *_resultsFileName;

    coBooleanParam *_startSim;
    coFileBrowserParam *_planeFile;
    coFileBrowserParam *_FamuExePath;
    coStringParam *_FamuArgs;
    coFileBrowserParam *_in_FirstFile;
    coFileBrowserParam *_in_SecondFile;
    coFileBrowserParam *_in_ThirdFile;

    coFileBrowserParam *_targetFile;

    coBooleanParam *_subdivideParam;
    coBooleanParam *_scaleDisplacements;
    coFloatParam *_periodicAngle;
    coIntScalarParam *_periodicTimeSteps;

    coIntScalarParam *_p_numt;
    coIntScalarParam *_p_skip;
    coStringParam *_p_selection;

    //params for the plane

    coFloatVectorParam *bottomLeft;
    coFloatVectorParam *bottomRight;
    coFloatVectorParam *topRight;
    coFloatVectorParam *topLeft;

    //variables for reserving the original points
    float origBottomLeft[3], origBottomRight[3], origTopRight[3], origTopLeft[3];

    //params for the transforming factors

    coFloatSliderParam *scaleFactor;
    coFloatVectorParam *moveDist;

    //params for the rotate factor
    coFloatSliderParam *XYDegree;
    coFloatSliderParam *YZDegree;
    coFloatSliderParam *ZXDegree;

    //param for the reset
    coBooleanParam *reset;

    //move Params for the Isolator
    coFloatVectorParam *moveIsol, *scaleIsol;

private:
    virtual int compute(const char *port);

    void getMeshDataTrans(MeshDataTrans **meshDataStat);
    void sendResultsToPorts(ResultsFileData *resultsFileData,
                            MeshDataTrans *meshDataTrans);

    virtual void displayString(const char *s);
    virtual void displayError(const char *s);
    virtual void displayError(std::string s);
    virtual void displayError(std::string s, std::string strQuotes);
};
#endif
