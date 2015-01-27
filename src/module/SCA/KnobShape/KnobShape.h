/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  MODULE KnobShape
//
//  it finds a knob
//
//  Initial version:   21.10.97 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2002 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
#ifndef _KNOB_SHAPE_H_
#define _KNOB_SHAPE_H_

#include <api/coModule.h>
using namespace covise;

#include <vector>
#include <string>
#ifdef __sgi
using namespace std;
#endif

#include <iostream>

//#include <util/coString.h>

class KnobShape : public coModule
{
public:
    KnobShape(int argc, char *argv[]);
    ~KnobShape();

protected:
    virtual int compute(const char *port);

private:
    coIntScalarParam *p_upDiv_;
    coIntScalarParam *p_downDiv_;
    coFloatParam *p_tolerance_;
    // coFloatParam *p_tolerance_;
    coInputPort *p_knobParam_;
    coOutputPort *p_showPoly_;
    coOutputPort *p_showNormals_;
    coOutputPort *p_handOutPoly_;

    int checkReadFlags();
    int checkKnobPath(string &getPath);

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
};
#endif
