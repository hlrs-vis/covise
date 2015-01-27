/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_ASCII_DYNA_H_
#define _READ_ASCII_DYNA_H_

#include <util/coviseCompat.h>
#include <util/coExport.h>
#include <vector>
#include <string>
#include <sstream>

#ifdef __sgi
using namespace std;
#endif

class SCAEXPORT ReadASCIIDyna
{
public:
    enum Direction
    {
        X,
        Y
    };
    static void MergeNodes(const vector<float> &exc,
                           const vector<float> &eyc,
                           const vector<float> &ezc,
                           vector<int> &ecl,
                           const vector<int> &epl,
                           float tolerance, float cutX = 0.0, float cutY = 0.0);

    static void Mirror2(vector<float> &exc, vector<float> &eyc, vector<float> &ezc,
                        vector<int> &ecl, vector<int> &epl, Direction);

    static int readEmbossingResults(ifstream &emb_conn, ifstream &emb_displ,
                                    ifstream &emb_thick,
                                    std::vector<int> &epl, std::vector<int> &cpl,
                                    std::vector<float> &exc,
                                    std::vector<float> &eyc,
                                    std::vector<float> &ezc, std::vector<float> &dicke);
    static int cleanFiles(const vector<string> &keepEnd);
    static const float CONVERSION_FACTOR;
    static const char *tissueTypes[];
    static const float tissueThickness[];
    static const char *noppenFormChoices[];

    static const float noppenHoeheLimits_[];
    static const float ausrundungsRadiusLimits_[];
    static const float abnutzungsRadiusLimits_[];
    static const float noppenWinkelLimits_[];
    static const float laenge1Limits_[];
    static const float laenge2Limits_[];
    static const float gummiHaerteLimits_[];
    static const float anpressDruckLimits_[];
    static const float *getMinMax(int level);

    // read pseudoparameters
    static int readIntSlider(istringstream &strText, int *addr, int len);
    static int readChoice(istringstream &strText, int *addr, int len);
    static int readBoolean(istringstream &strText, int *addr, int len);
    static int readFloatSlider(istringstream &strText, float *addr, int len);

    static void loadFileNames(string &, string &, string &);

    // knob generator
    static int ellipticKnob(float height, float groundRad, float abrasRad,
                            float winkel, float laeng1, float laeng2,
                            int angDiv, int upDiv, int downDiv,
                            vector<int> &pl, vector<int> &vl,
                            vector<float> &xl, vector<float> &yl, vector<float> &zl,
                            vector<float> &nxl, vector<float> &nyl, vector<float> &nzl);

    static int rectangularKnob(float height, float groundRad, float abrasRad,
                               float winkel, float laeng1, float laeng2,
                               int angDiv, int upDiv, int downDiv,
                               vector<int> &pl, vector<int> &vl,
                               vector<float> &xl, vector<float> &yl, vector<float> &zl,
                               vector<float> &nxl, vector<float> &nyl, vector<float> &nzl);

    static void SCA_Calls(const char *, vector<string> &arguments);

private:
    static int MarkCoords(std::vector<int> &cpl, std::vector<int> &mark);
    static int readDisplacements(ifstream &emb_displ,
                                 std::vector<int> &mark,
                                 std::vector<float> &exc,
                                 std::vector<float> &eyc,
                                 std::vector<float> &ezc);
    static int AddDisplacements(std::vector<float> &exc, std::vector<float> &eyc, std::vector<float> &ezc,
                                std::vector<float> &pxc, std::vector<float> &pyc, std::vector<float> &pzc,
                                std::vector<int> &mark);
    static float CrazyFormat(char number[8]);
};
#endif
