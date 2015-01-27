/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  MODULE Embossing
//
//  Embossing results
//
//  Initial version:   22.05.97 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2002 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
#ifndef _EMBOSSING_SCA_H_
#define _EMBOSSING_SCA_H_

#include <api/coModule.h>
using namespace covise;
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>

class Embossing : public coModule
{
public:
    Embossing(int argc, char *argv[]);
    ~Embossing();

protected:
    virtual int compute(const char *port);

private:
    bool permitTraction_;
    void outputMssg();
    void getImageName(std::string &);

    coInputPort *p_PraegeParam_;
    coInputPort *p_domain_;
    coInputPort *p_points_;
    coInputPort *p_colors_;
    coInputPort *p_wait_;
    coOutputPort *p_VernetzteGrundZelle_;
    coOutputPort *p_permitTraction_;
    coOutputPort *p_image_;

    std::string simDir_;

    int maxLen_;

    int checkReadFlags();
    int DesignRules(std::vector<float> &, std::vector<float> &);
    void outputDummies();
    bool gotDummies();
    void setBooleanFalse();

    int createANSYSMesh();
    int ANSYSInputAndLaunch(coDoPolygons *erhebung, coDoPolygons *thickObj);
    int LaunchANSYS();
    int readAndOutputMesh();

    void setTransformAttribute(std::string &);
    float barrelDiameter_; // using this variable is bad style,
    // but other alternatives are confronted
    // with a nasty bug when reusing an input object(!?)

    // set in gotDummies
    int no_points_;
    std::vector<float> xp_;
    std::vector<float> yp_;
    float width_;
    float height_;

    int readIntSlider(istringstream &strText, int *addr);
    int readChoice(istringstream &strText, int *addr);
    int readFloatSlider(istringstream &strText, float *addr);

    // set variables embConnFile, embDispFile
    int loadFileNames(std::string &, std::string &, std::string &, std::string &);
    /*   int readEmbossingResults(ifstream& emb_conn,ifstream& emb_displ,
         ifstream& emb_displ_exp,
         ifstream& emb_thick,
         vector<int>& epl,vector<int>& cpl,
         vector<float>& exc,
         vector<float>& eyc,
         vector<float>& ezc,
         vector<float>& dicke);

      static int MarkCoords(vector<int>& cpl,std::vector<int>& mark);
      int readDisplacements(ifstream& emb_displ,std::vector<int>& mark,
         vector<float>& exc,vector<float>& eyc,vector<float>& ezc,
         bool newFormat,int dickeSize);
*/
    enum Direction
    {
        X,
        Y
    };
    void Mirror2(vector<float> &exc, vector<float> &eyc, vector<float> &ezc,
                 vector<int> &ecl, vector<int> &epl, Direction);

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
    int anzahlReplikationenX_;
    bool readAnzahlReplikationenX_;
    int anzahlReplikationenY_;
    bool readAnzahlReplikationenY_;
    int presentationMode_;
    bool readPresentationMode_;
    // keep last input data
    bool SameData();
    bool FileExists();
    void keepInputData();

    bool trueData_;
    float l_noppenHoehe_;
    float l_ausrundungsRadius_;
    float l_abnutzungsRadius_;
    float l_noppenWinkel_;
    int l_noppenForm_;
    float l_laenge1_;
    float l_laenge2_;
    int l_tissueTyp_;
    float l_gummiHaerte_;
    float l_anpressDruck_;

    // info about desing
    bool SameDesign();
    void loadDesign();
    // previous design
    float l_width_;
    float l_height_;
    std::vector<float> l_xp_;
    std::vector<float> l_yp_;
};
#endif
