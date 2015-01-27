/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  MODULE Design
//
//  Pattern design module
//
//  Initial version:   22.05.97 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2002 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
#ifndef _DESIGN_SCA_H_
#define _DESIGN_SCA_H_

#include <api/coModule.h>
using namespace covise;
#include <string>
#include <iostream>
#include <vector>

#ifdef __sgi
using namespace std;
#endif

class Design : public coModule
{
public:
    Design(int argc, char *argv[]);
    ~Design();

protected:
    virtual int compute(const char *port);

private:
    coFloatParam *p_tolerance_;
    coInputPort *p_designParam_;
    coInputPort *p_Knob_;
    coOutputPort *p_grundZelle_;
    coOutputPort *p_noppenPositionen_;
    coOutputPort *p_noppenColors_;
    coOutputPort *p_show_grundZelle_;
    coOutputPort *p_show_noppenPositionen_;
    coOutputPort *p_show_noppenColors_;
    coOutputPort *p_show_fuesse_;
    coOutputPort *p_show_phys_fuesse_;
    coOutputPort *p_cutX_;
    coOutputPort *p_cutY_;

    int maxLen_;

    int checkReadFlags();
    void outputDummies();
    int readIntSlider(istringstream &strText, int *addr);
    int readChoice(istringstream &strText, int *addr);
    int readIntScalar(istringstream &strText, int *addr);
    int readFloatSlider(istringstream &strText, float *addr);
    int readFloatVector(istringstream &strText, int len, float *addr);

    void Periodicity(vector<float> &xc, vector<float> &yc, vector<float> &zc,
                     float width, float height, float deltax, float deltay);

    float grundZellenHoehe_;
    bool readGrundZellenHoehe_;
    float grundZellenBreite_;
    bool readGrundZellenBreite_;
    int anzahlLinien_;
    bool readAnzahlLinien_;
    int anzahlPunkteProLinie_;
    bool readAnzahlPunkteProLinie_;
    float versatz_;
    bool readVersatz_;
    int DanzahlReplikationenX_;
    bool readDAnzahlReplikationenX_;
    int DanzahlReplikationenY_;
    bool readDAnzahlReplikationenY_;
    // free or parametric
    int free_or_param_;
    bool readFree_or_param_;
    // knob related values
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
    // free case
    enum
    {
        MAX_POINTS = 50
    };

    float freie_noppen_[MAX_POINTS][2];
    bool readFreie_noppen_[MAX_POINTS];
    int num_points_;
    bool readNum_points_;

    int getKnobSize(float *xsize, float *ysize);
    void MakeTheKnobs(const vector<float> &x_c,
                      const vector<float> &y_c,
                      const vector<float> &z_c);

    void setTransformAttribute(std::string &);
    int DesignRules(vector<float> &x_c, vector<float> &y_c, vector<float> &colors,
                    bool knob_select, float xknob, float yknob);
    void ShowFuesse(vector<float> &x_c, vector<float> &y_c);
    void ShowPhysFuesse(vector<float> &x_c, vector<float> &y_c);
    float xknob_;
    float yknob_;
};
#endif
