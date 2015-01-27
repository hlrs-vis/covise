/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _EMBOSSING_SIMULATION_H_
#define _EMBOSSING_SIMULATION_H_

#include <api/coModule.h>
using namespace covise;
#include "ResultDataBase.h"
#include "ResultFloatParam.h"
#include "ResultEnumParam.h"

#include <sstream>
#include <fstream>
#include <vector>
#include <string>

class EmbossingSimulation : public coModule
{
public:
    EmbossingSimulation(int argc, char *argv[]);
    ~EmbossingSimulation();

protected:
    virtual int compute(const char *port);

private:
    coInputPort *p_PraegeParam_;
    coInputPort *p_colors_;
    coOutputPort *p_Done_;
    coIntSliderParam *p_ndivMet_; // esize param for metal part
    coIntSliderParam *p_ndivPap_; // esize param for paper part
    string simDir_;

    float noppenHoehe_;
    bool readNoppenHoehe_;
    float ausrundungsRadius_;
    bool readAusrundungsRadius_;
    float abnutzungsRadius_;
    bool readAbnutzungsRadius_;
    float noppenWinkel_;
    bool readNoppenWinkel_;
    int noppenForm_;
    bool readNoppenForm_;
    float laenge1_;
    bool readLaenge1_;
    float laenge2_;
    bool readLaenge2_;
    int tissueTyp_;
    bool readTissueTyp_;
    float gummiHaerte_;
    bool readGummiHaerte_;
    float anpressDruck_;
    bool readAnpressDruck_;
    // .........................
    void setBooleanFalse();
    void outputDummies();
    void gotDummies();
    int ANSYSInputAndLaunch();
    int CorrectLSDynaFormat();
    int checkReadFlags();
    int LSDYNALaunch();
    int createDirectories(string &);
    int LaunchANSYSForLS();
    int checkKnobPath(string &getPath,
                      std::vector<Candidate *> &FinalCandidates);
};
#endif
