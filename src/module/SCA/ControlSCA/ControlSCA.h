/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  MODULE ControlSCA
//
//  Central control of modules involved in the E-Type Project
//
//  Initial version:   22.05.97 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2002 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _CONTROL_SCA_H_
#define _CONTROL_SCA_H_

#include <api/coModule.h>
using namespace covise;
#include "ResultDataBase.h"

#include <vector>
#include <string>

using namespace std;

class ControlSCA : public coModule
{
public:
    /// Constructor
    ControlSCA();
    /// Destructor
    ~ControlSCA();

protected:
    virtual int compute(const char *port);
    virtual void postInst();

private:
    // parameter containers for the successive tasks
    // the info associated with this param groups
    // is made available to other modules
    std::vector<coUifPara *> p_knob_;
    std::vector<coUifPara *> p_design_;
    std::vector<coUifPara *> p_praegung_;
    std::vector<coUifPara *> p_zug_;
    std::vector<coUifPara *> p_readANSYS_;

    // The tasks
    enum
    {
        NO_EXECUTE = 0,
        KNOB_SELECT = 1,
        DESIGN = 2,
        PRAEGUNG = 3,
        VERNETZUNG = 4,
        ZUG = 5,
        READ_ANSYS = 6
    };
    // possible tasks at the moment
    vector<int> whatIsPossible_;
    // used to check if a task is possible:
    // uses whatIsPossible_ above
    bool isPossible(int choice);
    // writes a readable message informing about possible tasks
    void writeChoices(string &mssg);
    // setWhatIsPossible updates whatIsPossible_
    void setWhatIsPossible();
    // Maximum number of bumps in free design
    enum
    {
        MAX_POINTS = 50
    };
    // KnobState_ is describes possible decisions the user
    // may have to take regarding the convenience of accepting
    // an available bump or not.
    enum KnobState
    {
        INIT_KNOB,
        CONFIRM_KNOB,
        DECIDE_KNOB,
        READY_KNOB,
        SIMULATE_KNOB
    };
    KnobState KnobState_;
    // parameters:
    // p_whatExecute_: typically you use the choices in sequence
    coChoiceParam *p_whatExecute_;
    // p_SheetDimensions_: standard sheet sizes
    coChoiceParam *p_SheetDimensions_;
    // p_blatHoehe_, p_blatBreite_: manual sheet size
    coFloatSliderParam *p_blatHoehe_;
    coFloatSliderParam *p_blatBreite_;
    // p_tolerance_: relative tolerance (>0 <1)
    // in order to find similar bumps.
    // Increasing this number, more bumps
    // are deemed to be similar.
    coFloatParam *p_tolerance_;
    // p_DataBaseShape_: informs the user about
    // available bumps whose geometry (no tissue
    // type, no stamping force and no rubber info
    // is used for comparison) is close
    // to the actual bump
    coChoiceParam *p_DataBaseShape_;
    // p_DataBaseStrategy_: the same as p_DataBaseShape_,
    // but tissue type, stamping force
    // and rubber hardness are taken into account
    coChoiceParam *p_DataBaseStrategy_;

    virtual void param(const char *paramName, bool inMapLoading);

    // Design stuff
    // reads free-design info from a file
    int ReadCAD_Datei();

    // Design params
    coFileBrowserParam *p_cad_datei_;
    // geometry of the basic cell, from
    // which the design is derived by
    // mirror replication
    coFloatSliderParam *p_grundZellenHoehe_;
    coFloatSliderParam *p_grundZellenBreite_;

    // ignore this parameters: not used
    coIntSliderParam *p_anzahlLinien_;
    coIntSliderParam *p_anzahlPunkteProLinie_;
    coIntSliderParam *p_versatz_;

    // These parameters are internally used
    // by the module for the DESIGN phase,
    // but the user should ignore them.
    // They describe the number
    // of replications of a basic cell in order
    // to fill a paper sheet.
    coIntSliderParam *p_DanzahlReplikationenX;
    coIntSliderParam *p_DanzahlReplikationenY;

    // Set p_startEmbossing_ to true if you really
    // want to calculate or visualise a paper
    // bump. This was introduced in order to prevent
    // that the user "accidentally" starts an lsdyna
    // simulation.
    coBooleanParam *p_startEmbossing_;
    // this flag indicates we have performed
    // an embossing task
    bool oldStartEmbossing_;

    // master parameters
    coChoiceParam *p_free_or_param_;
    // parametric design (not free design)
    coFloatSliderParam *p_winkel_;
    coFloatSliderParam *p_noppen_abstand_;
    // free design: use p_grundZellenHoehe_ und p_grundZellenBreite_
    // to specify the geometry.
    // p_freie_noppen_: coordinate points of paper bump centers
    coFloatVectorParam *p_freie_noppen_[MAX_POINTS];
    // number of paper bump positions specified in p_freie_noppen_
    coIntScalarParam *p_num_points_;

    // used to look in the database for similar bumps (only
    // geometric parameters are considered)
    int checkKnobPath(string &getPath, std::vector<Candidate *> &FinalCandidates);
    // this function restricts the results of the previous function
    // including non-geometric parameters too
    int checkFullKnobPath(std::vector<Candidate *> &FinalCandidates);

    // modify slider values (they may also change min and max values)
    static void SetIntSliderValue(coIntSliderParam *, int num);
    static void SetFloatSliderValue(coFloatSliderParam *, float val);

    // keep old knob parameters when setting
    // parameters according to the values of a paper
    // bump available in the database. The old parameters
    // could be reset by RestoreOldValues(), but this function
    // is never used in this version
    void setNewValues(string &path, bool abgekuerzt = false);
    void loadOldFloats();
    void RestoreOldValues();
    float old_floats_[20];
    std::vector<coFloatSliderParam *> parList_;

    // Visualisation: sheet or roll
    coChoiceParam *p_PresentationMode_;
    // coChoiceParam *p_DataBaseStrategy_;

    // geometric params for knob specification
    coFloatSliderParam *p_noppenHoehe_;
    coFloatSliderParam *p_ausrundungsRadius_;
    coFloatSliderParam *p_abnutzungsRadius_;
    coFloatSliderParam *p_noppenWinkel_;
    // form (knob specification: elliptic or rectangular)
    coChoiceParam *p_noppenForm_;
    // length 1, 2 (knob specification)
    // these are diameters or side lengths
    // depending on p_noppenForm_. This describes
    // lengths of the surface that first contacts
    // paper during embossing (imagine p_abnutzungsRadius_ is 0)
    coFloatSliderParam *p_laenge1_;
    coFloatSliderParam *p_laenge2_;

    // non-geometric params for embossing specification
    // tissue type, rubber hardness and contact force
    coChoiceParam *p_tissueTyp_;
    coFloatSliderParam *p_gummiHaerte_;
    coFloatSliderParam *p_anpressDruck_;

    // These parameters are internally used
    // by the module for the VISUALISATION phase,
    // but the user should ignore them.
    // They describe the number
    // of replications of a basic cell in order
    // to fill a paper sheet.
    coIntSliderParam *p_anzahlReplikationenX;
    coIntSliderParam *p_anzahlReplikationenY;

    // p_barrelDiameter_ is only relevant in
    // roll visualisation
    coFloatSliderParam *p_barrelDiameter_;
    // p_thicknessInterpolation_ should be a value
    // between 0 and 1
    coFloatSliderParam *p_thicknessInterpolation_;
    coIntSliderParam *p_numberOfSheets_;
    // Zugsimulation
    coIntSliderParam *p_Displacement_;
    coIntSliderParam *p_NumOutPoints_;
    // ReadANSYS
    coChoiceParam *p_sol_;
    coChoiceParam *p_nsol_;
    coChoiceParam *p_esol_;
    coChoiceParam *p_stress_;
    coChoiceParam *p_top_bottom_;

    // ports
    coOutputPort *p_COVERInteractor_;
    coOutputPort *p_knobParam_;
    coOutputPort *p_designParam_;
    coOutputPort *p_praegeParam_;
    coOutputPort *p_blat_;
    coOutputPort *p_zugParam_;
    coOutputPort *p_cutX_;
    coOutputPort *p_cutY_;

    coOutputPort *p_PaperImage_;
    coOutputPort *p_BaseImage_;
    coOutputPort *p_PappeImage_;
    void ImageToTexture();

    // member functions
    void outputDummies();
    void outputKnob(int);
    void outputDesign(int);
    void outputPraegung(int);
    static float BumpHeight(std::vector<Candidate *> &FinalCandidates);
    void outputZug(int);
    void outputInteractor();
    void outputBorder(int);
    void outputCuts(float hoehe);

    int knob_found_;
    int full_knob_found_;
    string geometryPath_;
    void dynamicList(std::vector<Candidate *> &FinalCandidates);
    void dynamicFullList(std::vector<Candidate *> &FinalCandidates);
    bool isAcceptable(Candidate &cand, float abweichung);
    void addToEntries(std::vector<string *> &entries, Candidate &cand);
    void addToFullEntries(std::vector<string *> &entries, Candidate &cand);
};
#endif
