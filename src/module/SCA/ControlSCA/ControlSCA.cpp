/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ControlSCA.h"
//#include <util/coString.h>
#include "ReadASCIIDyna.h"
#include "DatabaseUtils.h"
#include <do/coDoText.h>

int
main(int argc, char *argv[])
{
    ControlSCA *application = new ControlSCA;
    application->start(argc, argv);
    return 1;
}

ControlSCA::ControlSCA()
    : coModule(0, 0, "SCA control")
    , KnobState_(INIT_KNOB)
    , oldStartEmbossing_(false)
{
    static const char *NodeChoices[] = {
        "none", "UX", "UY", "UZ", "U",
        "ROTX", "ROTY", "ROTZ", "ROT"
    };
    static const char *ElementChoices[] = {
        "none", "Stresses", "Elastic strains",
        "Plastic strains", "Creep strains"
    };
    static const char *SolidComponents[] = {
        "none", "XX", "YY", "ZZ", "XY", "YZ", "ZX",
        "T1", "T2", "T3", "TI", "TIGE"
    };
    static const char *TopBottomOpts[] = { "Top", "Bottom", "Average" };
    static const char *DataBaseStrategyChoices[] = { "My knob" };

    whatIsPossible_.push_back(NO_EXECUTE); // only select size is possible

    p_SheetDimensions_ = paraSwitch("SheetDimensions", "Toipa125x98");
    paraCase("Toipa140x98");
    paraEndCase();
    paraCase("Toipa125x110");
    paraEndCase();
    paraCase("Kuetue"
             "246x260");
    paraEndCase();
    paraCase("Kuetue"
             "246x230");
    paraEndCase();
    paraCase("Custom");
    p_blatHoehe_ = addFloatSliderParam("blatHoehe", "length");
    p_blatHoehe_->setValue(50, 300, 125);

    p_blatBreite_ = addFloatSliderParam("blatBreite", "width");
    p_blatBreite_->setValue(50, 300, 98);
    paraEndCase();
    paraEndSwitch();

    p_whatExecute_ = paraSwitch("whatExecute", "Please, enter leaf size");
    paraCase("KnobSelect");
    p_DataBaseShape_ = addChoiceParam("dbaseShape", "Data base shapes");
    static const char *DataBaseShapeChoices = "My knob";
    p_DataBaseShape_->setValue(1, &DataBaseShapeChoices, 0);
    // Knob stuff
    p_noppenHoehe_ = addFloatSliderParam("noppenHoehe", "Knob height");
    p_noppenHoehe_->setValue(ReadASCIIDyna::noppenHoeheLimits_[0],
                             ReadASCIIDyna::noppenHoeheLimits_[1],
                             ReadASCIIDyna::noppenHoeheLimits_[2]);

    p_ausrundungsRadius_ = addFloatSliderParam("ausrundungsRadius", "Ground radius");
    p_ausrundungsRadius_->setValue(ReadASCIIDyna::ausrundungsRadiusLimits_[0],
                                   ReadASCIIDyna::ausrundungsRadiusLimits_[1],
                                   ReadASCIIDyna::ausrundungsRadiusLimits_[2]);

    p_abnutzungsRadius_ = addFloatSliderParam("abnutzungsRadius", "Abrasion radius");
    p_abnutzungsRadius_->setValue(ReadASCIIDyna::abnutzungsRadiusLimits_[0],
                                  ReadASCIIDyna::abnutzungsRadiusLimits_[1],
                                  ReadASCIIDyna::abnutzungsRadiusLimits_[2]);

    p_noppenWinkel_ = addFloatSliderParam("noppenWinkel", "Flank angle");
    p_noppenWinkel_->setValue(ReadASCIIDyna::noppenWinkelLimits_[0],
                              ReadASCIIDyna::noppenWinkelLimits_[1],
                              ReadASCIIDyna::noppenWinkelLimits_[2]);

    p_noppenForm_ = addChoiceParam("noppenForm", "Noppenform");
    p_noppenForm_->setValue(3, ReadASCIIDyna::noppenFormChoices, 0);

    p_laenge1_ = addFloatSliderParam("laenge1", "First length");
    p_laenge1_->setValue(ReadASCIIDyna::laenge1Limits_[0],
                         ReadASCIIDyna::laenge1Limits_[1],
                         ReadASCIIDyna::laenge1Limits_[2]);

    p_laenge2_ = addFloatSliderParam("laenge2", "Second length");
    p_laenge2_->setValue(ReadASCIIDyna::laenge2Limits_[0],
                         ReadASCIIDyna::laenge2Limits_[1],
                         ReadASCIIDyna::laenge2Limits_[2]);

    p_tolerance_ = addFloatParam("Tolerance", "Relative tolerance");
    p_tolerance_->setValue(0.01);

    paraEndCase();
    paraCase("Design");
    p_free_or_param_ = paraSwitch("free_or_param", "Parametric or free design");
    paraCase("Free");
    p_cad_datei_ = addFileBrowserParam("cad_datei", "CAD file");
    p_cad_datei_->setValue("/tmp/foo.knb", "*.knb");

    p_grundZellenHoehe_ = addFloatSliderParam("grundZellenHoehe", "Basic cell height");
    p_grundZellenHoehe_->setValue(2.0, 20.0, 4.0);

    p_grundZellenBreite_ = addFloatSliderParam("grundZellenBreite", "Basic cell width");
    p_grundZellenBreite_->setValue(2.0, 20.0, 4.0);

    p_num_points_ = addInt32Param("NumPoints", "Number of knobs");
    p_num_points_->setValue(0);

    int noppen_counter;
    for (noppen_counter = 0; noppen_counter < MAX_POINTS; ++noppen_counter)
    {
        char name[256], descr[256];
        sprintf(name, "Noppen_%d", noppen_counter + 1);
        sprintf(descr, "Knob coordinates %d", noppen_counter + 1);
        float outofdomain[2] = { -1.0, -1.0 };
        p_freie_noppen_[noppen_counter] = addFloatVectorParam(name, descr);
        p_freie_noppen_[noppen_counter]->setValue(2, outofdomain);
    }

    paraEndCase();
    paraCase("Parametric");
    p_winkel_ = addFloatSliderParam("Winkel", "winkel");
    p_winkel_->setValue(0, 89, 60);

    p_noppen_abstand_ = addFloatSliderParam("KnobDistance", "knob distance");
    p_noppen_abstand_->setValue(1, 5, 2);
    paraEndCase();
    paraEndSwitch();
    // darstellung
    p_DanzahlReplikationenX = addIntSliderParam("DanzahlReplikationenX", "Number of replications in the X direction");
    p_DanzahlReplikationenX->setValue(1, 10, 4);

    p_DanzahlReplikationenY = addIntSliderParam("DanzahlReplikationenY", "Number of replications in the Y direction");
    p_DanzahlReplikationenY->setValue(1, 10, 4);

    paraEndCase();
    paraCase("Embossing");
    p_startEmbossing_ = addBooleanParam("startEmbossing", "start embossing");
    p_startEmbossing_->setValue(0);

    p_DataBaseStrategy_ = addChoiceParam("DataBaseStrategy",
                                         "MyKnob");
    p_DataBaseStrategy_->setValue(1, DataBaseStrategyChoices, 0);

    p_tissueTyp_ = addChoiceParam("tissueTyp", "Tissue type");
    p_tissueTyp_->setValue(6, ReadASCIIDyna::tissueTypes, 0);

    p_gummiHaerte_ = addFloatSliderParam("gummiHaerte", "Rubber hardness");
    p_gummiHaerte_->setValue(ReadASCIIDyna::gummiHaerteLimits_[0],
                             ReadASCIIDyna::gummiHaerteLimits_[1],
                             ReadASCIIDyna::gummiHaerteLimits_[2]); // Schinkoreit?

    p_anpressDruck_ = addFloatSliderParam("anpressDruck", "Contact pressure");
    p_anpressDruck_->setValue(ReadASCIIDyna::anpressDruckLimits_[0],
                              ReadASCIIDyna::anpressDruckLimits_[1],
                              ReadASCIIDyna::anpressDruckLimits_[2]); // Schinkoreit?

    // darstellung
    paraEndCase();
    paraCase("TissueVisualisation");
    p_anzahlReplikationenX = addIntSliderParam("anzahlReplikationenX", "Number of replications in the X direction");
    p_anzahlReplikationenX->setValue(0, 10, 4);

    p_anzahlReplikationenY = addIntSliderParam("anzahlReplikationenY", "Number of replications in the Y direction");
    p_anzahlReplikationenY->setValue(0, 10, 4);

    static const char *presentations[] = { "Sheet", "Roll" };
    p_PresentationMode_ = addChoiceParam("presentationMode", "Presentation mode");
    p_PresentationMode_->setValue(2, presentations, 0);

    p_barrelDiameter_ = addFloatSliderParam("barrelDiameter",
                                            "barrel diameter");
    p_barrelDiameter_->setValue(30, 100, 50);

    p_thicknessInterpolation_ = addFloatSliderParam("thicknessInterpolation",
                                                    "Thickness interpolation param.");
    p_thicknessInterpolation_->setValue(0, 1, 0.65);

    p_numberOfSheets_ = addIntSliderParam("NumberOfSheets", "number of sheets");
    p_numberOfSheets_->setValue(100, 300, 200);

    paraEndCase();
    paraCase("Zug");
    p_Displacement_ = addIntSliderParam("Displacement", "Last relative traction displacement");
    p_Displacement_->setValue(0, 20, 10); // Schinkoreit?
    p_NumOutPoints_ = addIntSliderParam("NumOutPoints", "Number of output points");
    p_NumOutPoints_->setValue(5, 20, 10); // Schinkoreit?
    paraEndCase();
    paraCase("ReadANSYS");
    p_sol_ = paraSwitch("Solution", "Please enter your choice");
    paraCase("OnlyGeometry");
    paraEndCase();
    paraCase("NodeData");
    p_nsol_ = addChoiceParam("DOF_Solution", "Degrees of freedom");
    p_nsol_->setValue(9, NodeChoices, 0);
    paraEndCase();
    paraCase("ElementData");
    p_esol_ = addChoiceParam("Derived_Solution", "Derived variables");
    p_esol_->setValue(5, ElementChoices, 0);

    p_stress_ = addChoiceParam("SolidShellComponents", "Stress components");
    p_stress_->setValue(12, SolidComponents, 0);

    p_top_bottom_ = addChoiceParam("TopBottom", "Top, bottom, average");
    p_top_bottom_->setValue(3, TopBottomOpts, 0);

    paraEndCase();
    paraEndSwitch();
    paraEndCase();
    paraEndSwitch();

    // Old parametric case
    p_anzahlLinien_ = addIntSliderParam("anzahlLinien", "Number of lines in a basic cell");
    p_anzahlLinien_->setValue(0, 10, 3);

    p_anzahlPunkteProLinie_ = addIntSliderParam("anzahlPunkteProLinie", "Number of points per line");
    p_anzahlPunkteProLinie_->setValue(1, 10, 3);

    p_versatz_ = addIntSliderParam("versatz", "Lateral off-set in rows until periodicity");
    p_versatz_->setValue(0, 3, 0);

    // ports
    p_knobParam_ = addOutputPort("knobParams", "Text", "Knob parameters");
    p_designParam_ = addOutputPort("designParams", "Text", "Design parameters");
    p_praegeParam_ = addOutputPort("praegeParams", "Text", "Stamping parameters");
    p_zugParam_ = addOutputPort("zugParams", "Text", "Traction parameters");
    p_blat_ = addOutputPort("Blat", "Lines", "Leaf border lines");
    p_COVERInteractor_ = addOutputPort("COVERInteractor", "Points", "COVER interactor");
    p_cutX_ = addOutputPort("cutX", "Text", "CutGeomerty X params");
    p_cutY_ = addOutputPort("cutY", "Text", "CutGeomerty Y params");

    // image_to_texture
    p_PaperImage_ = addOutputPort("PaperImage", "Text", "Paper image");
    p_BaseImage_ = addOutputPort("BaseImage", "Text", "Base image");
    p_PappeImage_ = addOutputPort("PappeImage", "Text", "Pappe image");
}

void
ControlSCA::ImageToTexture()
{
    // depending on tissue..
    string text_path = getenv("SCA_PATH");
    text_path += "/TEXTURES/";

    // paper_image
    string paper_image = text_path;
    if (p_whatExecute_->getValue() != 3)
    {
        paper_image += ReadASCIIDyna::tissueTypes[p_tissueTyp_->getValue()];
    }
    else
    {
        paper_image += "White";
    }
    paper_image += ".tif";
    coDoText *PaperImage = new coDoText(p_PaperImage_->getObjName(), paper_image.length() + 1);
    char *addr = NULL;
    PaperImage->getAddress(&addr);
    strcpy(addr, paper_image.c_str());
    p_PaperImage_->setCurrentObject(PaperImage);

    // base image
    string base_image = text_path;
    base_image += "Base.tif";
    coDoText *BaseImage = new coDoText(p_BaseImage_->getObjName(), base_image.length() + 1);
    BaseImage->getAddress(&addr);
    strcpy(addr, base_image.c_str());
    p_BaseImage_->setCurrentObject(BaseImage);

    // pappe image
    string pappe_image = text_path;
    pappe_image += "Pappe.tif";
    coDoText *PappeImage = new coDoText(p_PappeImage_->getObjName(), pappe_image.length() + 1);
    PappeImage->getAddress(&addr);
    strcpy(addr, pappe_image.c_str());
    p_PappeImage_->setCurrentObject(PappeImage);
}

void
ControlSCA::outputCuts(float hoehe)
{
    char buf[1024];

    sprintf(buf, "distance %g\nnormal ", -float(p_blatBreite_->getValue()));
    strcat(buf, "-1.0 0.0 0.0");
    coDoText *doTextX = new coDoText(p_cutX_->getObjName(), strlen(buf) + 1);
    char *addr;
    doTextX->getAddress(&addr);
    strcpy(addr, buf);
    p_cutX_->setCurrentObject(doTextX);

    sprintf(buf, "distance %g \nnormal 0.0 -1.0 0.0", -hoehe);
    coDoText *doTextY = new coDoText(p_cutY_->getObjName(), strlen(buf) + 1);
    doTextY->getAddress(&addr);
    strcpy(addr, buf);
    p_cutY_->setCurrentObject(doTextY);
}

ControlSCA::~ControlSCA()
{
}

static void
outputParams(coOutputPort *p_port, std::vector<coUifPara *> &params)
{
    string TextParams;
    int param;
    for (param = 0; param < params.size(); ++param)
    {
        TextParams += params[param]->getName();
        TextParams += '\n';
        TextParams += params[param]->getValString();
        TextParams += '\n';
    }
    coDoText *doText = new coDoText(p_port->getObjName(), TextParams.length() + 1);
    char *addr;
    doText->getAddress(&addr);
    strcpy(addr, TextParams.c_str());

    p_port->setCurrentObject(doText);
}

void
ControlSCA::param(const char *paramName, bool inMapLoading)
{
    static const char *list[] = { "My knob" };
    static const char *listStr[] = { "Disabled" };
    if (inMapLoading)
    {
        p_DataBaseShape_->setValue(1, list, 0);
        p_DataBaseStrategy_->setValue(1, listStr, 0);
        return;
    }

    if (p_whatExecute_->getValue() == KNOB_SELECT
        && strcmp(p_DataBaseShape_->getName(), paramName) == 0
        && p_DataBaseShape_->getValue() > 0)
    {
        string pathS = p_DataBaseShape_->getActLabel();
        setNewValues(pathS, true);
    }

    if (strcmp(p_gummiHaerte_->getName(), paramName) == 0
        || strcmp(p_anpressDruck_->getName(), paramName) == 0
        || strcmp(p_tissueTyp_->getName(), paramName) == 0
        || (strcmp(p_whatExecute_->getName(), paramName) == 0
            && p_whatExecute_->getValue() == PRAEGUNG))
    {
        p_startEmbossing_->setValue(0);
        oldStartEmbossing_ = false;
        // find out whether there are knob versions available
        // with the same geometry
        std::vector<Candidate *> FinalCandidates;
        if (knob_found_ == 0)
        {
            // starting directory is geometryPath_
            full_knob_found_ = checkFullKnobPath(FinalCandidates);
        }
        else
        {
            full_knob_found_ = 2;
        }
        dynamicFullList(FinalCandidates);
        // clean FinalCandidates
        int cand;
        for (cand = 0; cand < FinalCandidates.size(); ++cand)
        {
            delete FinalCandidates[cand];
        }
        // old situation...
        // p_DataBaseStrategy_->setValue(1,listStr,1);
        // p_DataBaseStrategy_->disable();
    }

    if (p_whatExecute_->getValue() == PRAEGUNG
        && strcmp(p_DataBaseStrategy_->getName(), paramName) == 0
        && p_DataBaseStrategy_->getValue() > 0)
    {
        string pathS = p_DataBaseStrategy_->getActLabel();
        // set non-geometric knob params
        const char *start = pathS.c_str();
        start = strstr(start, "=");
        if (start)
        {
            float number;
            ++start;
            if (sscanf(start, "%g", &number) == 1)
            {
                SetFloatSliderValue(p_gummiHaerte_, number);
            }
            start = strstr(start, "=");
            if (start)
            {
                float number;
                ++start;
                if (sscanf(start, "%g", &number) == 1)
                {
                    SetFloatSliderValue(p_anpressDruck_, number);
                }
            }
        }
    }

    if (p_noppenForm_->getValue() == 2 && (strcmp(p_laenge1_->getName(), paramName) == 0 || strcmp(p_laenge2_->getName(), paramName) == 0))
    {
        if (strcmp(p_laenge1_->getName(), paramName) == 0)
        {
            p_laenge2_->setValue(p_laenge1_->getValue());
        }
        else
        {
            p_laenge1_->setValue(p_laenge2_->getValue());
        }
    }
    if (strcmp(p_laenge1_->getName(), paramName) == 0 || strcmp(p_laenge2_->getName(), paramName) == 0 || strcmp(p_noppenHoehe_->getName(), paramName) == 0)
    {
        //float oldvalAusrund = p_ausrundungsRadius_->getValue();
        //float oldvalAbnutz = p_abnutzungsRadius_->getValue();
        float minL = (p_laenge1_->getValue() < p_laenge2_->getValue()) ? p_laenge1_->getValue() : p_laenge2_->getValue();
        minL *= 0.5;
        if (minL >= 0.5 * p_noppenHoehe_->getValue())
        {
            minL = 0.5 * p_noppenHoehe_->getValue();
        }
        p_ausrundungsRadius_->setMax(minL);
        p_abnutzungsRadius_->setMax(minL);
    }
    if (strcmp(p_whatExecute_->getName(), paramName) == 0)
    {
        if (p_whatExecute_->getValue() == KNOB_SELECT && KnobState_ != CONFIRM_KNOB)
        {
            KnobState_ = INIT_KNOB;
        }
    }
    if (strcmp(p_noppenHoehe_->getName(), paramName) == 0
        || strcmp(p_ausrundungsRadius_->getName(), paramName) == 0
        || strcmp(p_abnutzungsRadius_->getName(), paramName) == 0
        || strcmp(p_noppenWinkel_->getName(), paramName) == 0
        || strcmp(p_noppenForm_->getName(), paramName) == 0
        || strcmp(p_laenge1_->getName(), paramName) == 0
        || strcmp(p_laenge2_->getName(), paramName) == 0
           //|| strcmp(p_tissueTyp_->getName(),paramName)==0
           //|| strcmp(p_gummiHaerte_->getName(),paramName)==0
           //|| strcmp(p_anpressDruck_->getName(),paramName)==0
        )
    {
        KnobState_ = INIT_KNOB;
        // p_DataBaseStrategy_->disable();
    }

    if (p_free_or_param_->getValue() == 1 && // free design
        (strcmp(p_cad_datei_->getName(), paramName) == 0 || strcmp(p_free_or_param_->getName(), paramName) == 0))
    {
        ReadCAD_Datei();
    }
}

#include <sstream>
#include <fstream>

// #define _DXF_DATEI_

int
ControlSCA::ReadCAD_Datei()
{
    // CAD reader functionalities are expected here
    ifstream cadfile(p_cad_datei_->getValue());
    if (!cadfile.rdbuf()->is_open())
    {
        sendWarning("Could not open knob-file");
        return -1;
    }
    float grZellex;
    float grZelley;
    if (!(cadfile >> grZellex >> grZelley))
    {
        sendWarning("Could not read basic cell dimensions");
        return -1;
    }
    p_grundZellenBreite_->setValue(grZellex);
    p_grundZellenHoehe_->setValue(grZelley);

    int no_points;
    if (!(cadfile >> no_points))
    {
        sendWarning("Could not read number of knobs");
        return -1;
    }
    if (no_points > MAX_POINTS)
    {
        sendWarning("Too many knobs in file, some may be ignored");
        no_points = MAX_POINTS;
    }
    char buffer[1024];
    int count_points = 0;
    while (cadfile.getline(buffer, 1024) && count_points < no_points)
    {
        istringstream codeBuf(buffer);
        float x, y;
        if (!(codeBuf >> x >> y))
        {
            continue;
        }
        p_freie_noppen_[count_points]->setValue(0, x);
        p_freie_noppen_[count_points]->setValue(1, y);
        ++count_points;
    }
    // set the other p_freie_noppen_ parameters to the value -1,-1
    int paramIndex;
    for (paramIndex = count_points; paramIndex < MAX_POINTS; ++paramIndex)
    {
        p_freie_noppen_[paramIndex]->setValue(0, -1.0);
        p_freie_noppen_[paramIndex]->setValue(1, -1.0);
    }

    if (count_points < no_points)
    {
        sendWarning("Less knob positions than expected could be read");
        return -1;
    }

    p_num_points_->setValue(count_points);
    return 0;

#ifdef _DXF_DATEI_
    int no_points = 0;
    // ifstream cadfile(p_cad_datei_->getValue());
    ifstream cadfile("R_Edges_Nested2.dxf");
    ofstream outfile("R_Edges_Nested2.txt");
    char buffer[1024];
    while (cadfile.getline(buffer, 1024) && no_points < 10000 /* MAX_POINTS */)
    {
        if (strncmp(buffer, "CIRCLE", 6) != 0)
        {
            continue;
        }
        int count_coord = 0;
        float x, y;
        while (cadfile.getline(buffer, 1024))
        {
            if (buffer[0] != ' ')
            {
                continue;
            }
            istringstream codeBuf(buffer);
            int code;
            if (!(codeBuf >> code))
            {
                sendWarning("Could not read center coordinate code");
                return -1;
            }
            switch (code)
            {
            case 10:
                cadfile >> x;
                ++count_coord;
                break;
            case 20:
                cadfile >> y;
                ++count_coord;
                break;
            default:
                break;
            }
            if (count_coord == 2)
            {
                break;
            }
        }
        // p_freie_noppen_[no_points]->setValue(0,x);
        // p_freie_noppen_[no_points]->setValue(1,y);
        outfile << x << "   " << y << endl;
        ++no_points;
    }
    // p_num_points_->setValue(no_points);
    return 0;
#endif
}

void
ControlSCA::outputKnob(int whatExecute)
{
    if (whatExecute != KNOB_SELECT
        /*!(
        whatExecute==KNOB_SELECT
        || (
             whatExecute==DESIGN
           && (KnobState_ == CONFIRM_KNOB || KnobState_==DECIDE_KNOB)
           )
      )*/
        )
    {
        outputParams(p_knobParam_, p_knob_);
        return;
    }

    // check if previous selected knob is available
    // if not warn the user, that he may have to
    // launch an LS-DYNA calculation, if the
    // embossing phase has to be later executed.
    // We give him the chance of accepting the
    // "nearest" value...
    string getPath;
    // 0 -> exact hit
    // 1 -> approx.
    // 2 -> not found
    std::vector<Candidate *> FinalCandidates;
    knob_found_ = checkKnobPath(getPath, FinalCandidates);
    if (knob_found_ == 0)
    {
        geometryPath_ = getPath;
    }
    // create dynamic list for suggestions
    dynamicList(FinalCandidates);
    int cand;
    for (cand = 0; cand < FinalCandidates.size(); ++cand)
    {
        delete FinalCandidates[cand];
    }

    KnobState_ = READY_KNOB;

    /*
      if( whatExecute==DESIGN && KnobState_ == CONFIRM_KNOB){
         // show approximate knob
         setNewValues(getPath);
         p_DataBaseStrategy_->enable();
      }

      if( whatExecute==DESIGN && KnobState_ == DECIDE_KNOB){
         if(p_DataBaseStrategy_->getValue()==1){ // accept approx
            // KnobState_ = READY_KNOB;
         }
   else{  // I insist on using my knob
   // KnobState_ = SIMULATE_KNOB;
   sendInfo("Using your knob");
   RestoreOldValues();
   }
   p_DataBaseStrategy_->disable();
   }

   outputParams(p_knobParam_,p_knob_);
   coDistributedObject *knob = p_knobParam_->getCurrentObject();
   if(whatExecute == KNOB_SELECT ||
   (whatExecute==DESIGN && KnobState_ == DECIDE_KNOB) ||
   (whatExecute==DESIGN && KnobState_ == CONFIRM_KNOB) ){
   */
    outputParams(p_knobParam_, p_knob_);
    coDistributedObject *knob = p_knobParam_->getCurrentObject();
    knob->addAttribute("KNOB_SELECT", "");
    //   }
}

void
ControlSCA::outputDesign(int whatExecute)
{
    if (whatExecute < DESIGN)
    {
        p_designParam_->setCurrentObject(new coDoText(p_designParam_->getObjName(), 0));
        return;
    }
    switch (KnobState_)
    {
    case INIT_KNOB:
        sendInfo("Please, choose a knob first");
        outputDesign(NO_EXECUTE);
        return;
    case CONFIRM_KNOB:
        sendInfo("Accept DB approximation or use exact knob?");
        KnobState_ = DECIDE_KNOB; // we have to take a decision
        outputDesign(NO_EXECUTE);
        return;
    case DECIDE_KNOB:
        if (p_DataBaseStrategy_->getValue() == 0) // accept approx
        {
            KnobState_ = READY_KNOB;
        }
        else // I insist on using my knob
        {
            KnobState_ = SIMULATE_KNOB;
        }
        outputDesign(NO_EXECUTE);
        return;
    case READY_KNOB:
        // output design params as they are now
        break;
    case SIMULATE_KNOB:
        // output design params as they are now
        break;
    }
    if (p_free_or_param_->getValue() == 2) // simplified parametric design
    {
        float newGrundZellenBreite = p_noppen_abstand_->getValue();
        float newGrundZellenHoehe = newGrundZellenBreite * tan(M_PI * p_winkel_->getValue() / 180.0);
        if (p_grundZellenHoehe_->getMin() > newGrundZellenHoehe)
        {
            p_grundZellenHoehe_->setMin(newGrundZellenHoehe);
        }
        if (p_grundZellenHoehe_->getMax() < newGrundZellenHoehe)
        {
            p_grundZellenHoehe_->setMax(newGrundZellenHoehe);
        }
        if (p_grundZellenBreite_->getMin() > newGrundZellenBreite)
        {
            p_grundZellenBreite_->setMin(newGrundZellenBreite);
        }
        if (p_grundZellenBreite_->getMax() < newGrundZellenBreite)
        {
            p_grundZellenBreite_->setMax(newGrundZellenBreite);
        }
        p_grundZellenBreite_->setValue(newGrundZellenBreite);
        p_grundZellenHoehe_->setValue(newGrundZellenHoehe);
        p_anzahlLinien_->setValue(1);
        p_anzahlPunkteProLinie_->setValue(1);
        p_versatz_->setValue(0);
    }
    // calculate here number of replications
    // according to sheet and basic cell
    SetIntSliderValue(p_DanzahlReplikationenX, int(rint(
                                                   float(p_blatBreite_->getValue()) / p_grundZellenBreite_->getValue())));
    SetIntSliderValue(p_DanzahlReplikationenY, int(rint(
                                                   float(p_blatHoehe_->getValue()) / p_grundZellenHoehe_->getValue())));
    outputParams(p_designParam_, p_design_);
    coDistributedObject *design = p_designParam_->getCurrentObject();
    if (p_whatExecute_->getValue() == DESIGN)
    {
        design->addAttribute("SHOW_SCA_DESIGN", "");
    }
    else if (p_whatExecute_->getValue() == KNOB_SELECT)
    {
        design->addAttribute("KNOB_SELECT", "");
    }
}

void
ControlSCA::outputPraegung(int whatExecute)
{
    if (whatExecute != PRAEGUNG && whatExecute != VERNETZUNG)
    {
        outputCuts(p_blatHoehe_->getValue());
        p_praegeParam_->setCurrentObject(new coDoText(p_praegeParam_->getObjName(), 0));
        return;
    }
    // find out whether there are knob versions available
    // with the same geometry
    std::vector<Candidate *> FinalCandidates;
    if (knob_found_ == 0)
    {
        // starting directory is geometryPath_
        full_knob_found_ = checkFullKnobPath(FinalCandidates);
    }
    else
    {
        full_knob_found_ = 2;
    }
    dynamicFullList(FinalCandidates);

    // try to get the height of a paper bump
    float bumpHeight = 0.0;
    if (whatExecute == VERNETZUNG)
    {
        bumpHeight = BumpHeight(FinalCandidates);
        if (bumpHeight > 0.0)
        {
            sendInfo("BumpHeight is %g", bumpHeight);
        }
    }
    // clean FinalCandidates
    int cand;
    for (cand = 0; cand < FinalCandidates.size(); ++cand)
    {
        delete FinalCandidates[cand];
    }

    if (whatExecute == PRAEGUNG && p_startEmbossing_->getValue() == 0)
    {
        outputCuts(p_blatHoehe_->getValue());
        p_praegeParam_->setCurrentObject(new coDoText(p_praegeParam_->getObjName(), 0));
        return;
    }

    p_startEmbossing_->setValue(0);
    oldStartEmbossing_ = true;

    // calculate here number of replications
    // according to roll diameter and basic cell
    // the roll diameter is determined by the kind of tissue
    float barrelRadius = p_barrelDiameter_->getValue() * 0.5;
    float Uthick = ReadASCIIDyna::tissueThickness[p_tissueTyp_->getValue()];
    float Gthick = Uthick + 0.6 * p_noppenHoehe_->getValue();
    if (bumpHeight != 0.0)
    {
        Gthick = Uthick + bumpHeight;
    }
    float thick = Uthick + Gthick * p_thicknessInterpolation_->getValue();
    float echteHoehe;
    if (p_PresentationMode_->getValue() == 0)
    {
        float echterRadius = sqrt(barrelRadius * barrelRadius + thick * p_numberOfSheets_->getValue() * p_blatHoehe_->getValue() / M_PI);
        echteHoehe = 2 * M_PI * echterRadius;
        SetIntSliderValue(p_anzahlReplikationenX, int(rint(
                                                      float(p_blatBreite_->getValue()) / p_grundZellenBreite_->getValue())));
        SetIntSliderValue(p_anzahlReplikationenY, int(rint(
                                                      echteHoehe / p_grundZellenHoehe_->getValue())));
    }
    else
    {
        echteHoehe = p_blatHoehe_->getValue();
        SetIntSliderValue(p_anzahlReplikationenX, int(rint(
                                                      float(p_blatBreite_->getValue()) / p_grundZellenBreite_->getValue())));
        SetIntSliderValue(p_anzahlReplikationenY, int(rint(
                                                      float(echteHoehe) / p_grundZellenHoehe_->getValue())));
    }
    outputCuts(echteHoehe);
    outputParams(p_praegeParam_, p_praegung_);
    // add attribute: PRAEGUNG or VERNETZUNG
    coDistributedObject *addAtt = p_praegeParam_->getCurrentObject();
    if (whatExecute == PRAEGUNG)
    {
        addAtt->addAttribute("PRAEGUNG", "");
    }
    else if (whatExecute == VERNETZUNG)
    {
        addAtt->addAttribute("VERNETZUNG", "");
    }
}

void
ControlSCA::outputZug(int whatExecute)
{
    if (whatExecute != ZUG && whatExecute != READ_ANSYS)
    {
        p_zugParam_->setCurrentObject(new coDoText(p_zugParam_->getObjName(), 0));
        return;
    }

    if (p_whatExecute_->getValue() != ZUG
        || !oldStartEmbossing_
           // @@@ || p_startEmbossing_->getValue()==0
        )
    {
        // output a dummy
        p_zugParam_->setCurrentObject(new coDoText(p_zugParam_->getObjName(), 0));
        if (p_whatExecute_->getValue() == ZUG
            && !oldStartEmbossing_
               // @@@ && p_startEmbossing_->getValue()==0
            )
        {
            sendWarning("Embossing should be executed prior to a traction simulation");
            return;
        }
    }
    else
    {
        outputParams(p_zugParam_, p_zug_);
    }
    if (p_whatExecute_->getValue() == READ_ANSYS
        || p_whatExecute_->getValue() == ZUG)
    {
        string readANSYSattr;
        int param;
        std::vector<coUifPara *> params;
        params.push_back(p_sol_);
        params.push_back(p_nsol_);
        params.push_back(p_esol_);
        params.push_back(p_stress_);
        params.push_back(p_top_bottom_);
        for (param = 0; param < params.size(); ++param)
        {
            readANSYSattr += params[param]->getName();
            readANSYSattr += ' ';
            readANSYSattr += params[param]->getValString();
            readANSYSattr += '\n';
        }
        p_zugParam_->getCurrentObject()->addAttribute("READ_ANSYS",
                                                      readANSYSattr.c_str());
    }
    return;
}

void
ControlSCA::postInst()
{
    p_DataBaseStrategy_->disable();
    coUifPara *p_knob[] = {
        p_noppenHoehe_,
        p_ausrundungsRadius_,
        p_abnutzungsRadius_,
        p_noppenWinkel_,
        /*
                                p_tissueTyp_,
                                p_gummiHaerte_,
                                p_anpressDruck_,
      */
        p_noppenForm_,
        p_laenge1_,
        p_laenge2_,
        p_DataBaseShape_,
        NULL
    };
    coUifPara *p_design[] = {
        p_grundZellenHoehe_,
        p_grundZellenBreite_,
        p_anzahlLinien_,
        p_anzahlPunkteProLinie_,
        p_versatz_,
        p_DanzahlReplikationenX,
        p_DanzahlReplikationenY,
        p_free_or_param_,
        p_winkel_,
        p_noppen_abstand_,
        p_num_points_, // p_freie_noppen_ comes below
        p_noppenHoehe_,
        p_ausrundungsRadius_,
        p_abnutzungsRadius_,
        p_noppenWinkel_,
        p_tissueTyp_,
        p_gummiHaerte_,
        p_anpressDruck_,
        p_noppenForm_,
        p_laenge1_,
        p_laenge2_,
        NULL
    };
    int param;
    for (param = 0; p_knob[param]; ++param)
    {
        p_knob_.push_back(p_knob[param]);
    }
    for (param = 0; p_design[param]; ++param)
    {
        p_design_.push_back(p_design[param]);
    }
    // add p_freie_noppen_
    for (param = 0; param < MAX_POINTS; ++param)
    {
        p_design_.push_back(p_freie_noppen_[param]);
    }

    coUifPara *p_praegung[] = {
        p_noppenHoehe_,
        p_ausrundungsRadius_,
        p_abnutzungsRadius_,
        p_noppenWinkel_,
        p_tissueTyp_,
        p_gummiHaerte_,
        p_anpressDruck_,
        p_anzahlReplikationenX,
        p_anzahlReplikationenY,
        p_barrelDiameter_,
        p_thicknessInterpolation_,
        p_numberOfSheets_,

        p_noppenForm_,
        p_laenge1_,
        p_laenge2_,
        p_PresentationMode_,
        p_startEmbossing_,
        NULL
    };
    for (param = 0; p_praegung[param]; ++param)
    {
        p_praegung_.push_back(p_praegung[param]);
    }

    p_zug_.push_back(p_Displacement_);
    p_zug_.push_back(p_NumOutPoints_);

    p_readANSYS_.push_back(p_sol_);
    p_readANSYS_.push_back(p_nsol_);
    p_readANSYS_.push_back(p_esol_);
    p_readANSYS_.push_back(p_stress_);
    p_readANSYS_.push_back(p_top_bottom_);

    p_blatHoehe_->show();
    p_blatBreite_->show();

    parList_.push_back(p_noppenHoehe_);
    parList_.push_back(p_ausrundungsRadius_);
    parList_.push_back(p_abnutzungsRadius_);
    parList_.push_back(p_noppenWinkel_);
    parList_.push_back(p_laenge1_);
    parList_.push_back(p_laenge2_);
    /*
      parList_.push_back(p_gummiHaerte_);
      parList_.push_back(p_anpressDruck_);
   */
}

#include "ResultDataBase.h"
#include "ResultFloatParam.h"
#include "ResultEnumParam.h"
#include "ReadASCIIDyna.h"
#include <float.h>

int
ControlSCA::checkFullKnobPath(std::vector<Candidate *> &FinalCandidates)
{
    const char *sca_path = geometryPath_.c_str();
    ResultDataBase dataB(sca_path);
    std::vector<ResultParam *> list;
    ResultEnumParam v0("tissueTyp", 6, ReadASCIIDyna::tissueTypes,
                       p_tissueTyp_->getValue());
    ResultFloatParam v1("gummiHaerte", p_gummiHaerte_->getValue(), 3);
    ResultFloatParam v2("anpressDruck", p_anpressDruck_->getValue(), 3);

    list.push_back(&v0);
    list.push_back(&v1);
    list.push_back(&v2);

    float diff = 0.0;
    const char *path = dataB.searchForResult(diff, list, FinalCandidates,
                                             p_tolerance_->getValue(), 3);
    if (path == NULL)
    {
        sendWarning("Could not find embossing results with the given parameters in DB");
        return 2;
    }
    return (diff != 0.0);
}

// 0 -> exact hit
// 1 -> approx.
// 2 -> not found
int
ControlSCA::checkKnobPath(string &getPath, std::vector<Candidate *> &FinalCandidates)
{
    const char *sca_path_c = getenv("SCA_PATH");
    if (!sca_path_c)
    {
        sendWarning("Please, set SCA_PATH environment variable");
        return 2;
    }
    string sca_path(sca_path_c);
    sca_path += "/DATABASE/";
    ResultDataBase dataB(sca_path.c_str());
    std::vector<ResultParam *> list; // prepare a list with knob parameter values:
    // noppenHoehe, ausrundungsRadius, abnutzungsRadius, noppenWinkel,
    // noppenForm, laenge1, laenge2, tissueTyp, gummiHaerte, anpressDruck
    ResultFloatParam v0("noppenHoehe", p_noppenHoehe_->getValue(), 3);
    ResultFloatParam v1("ausrundungsRadius", p_ausrundungsRadius_->getValue(), 3);
    ResultFloatParam v2("abnutzungsRadius", p_abnutzungsRadius_->getValue(), 3);
    ResultFloatParam v3("noppenWinkel", p_noppenWinkel_->getValue(), 3);
    // char *labs[] = { "Raute", "Ellipse" };
    int nop_form = p_noppenForm_->getValue();
    if (nop_form == 2) // Kreis
    {
        nop_form = 1;
    }
    ResultEnumParam v4("noppenForm", 2, ReadASCIIDyna::noppenFormChoices, nop_form);
    ResultFloatParam v5("laenge1", p_laenge1_->getValue(), 3);
    ResultFloatParam v6("laenge2", p_laenge2_->getValue(), 3);
    list.push_back(&v0);
    list.push_back(&v1);
    list.push_back(&v2);
    list.push_back(&v3);
    list.push_back(&v4);
    list.push_back(&v5);
    list.push_back(&v6);

    float diff = 0.0;
    const char *path = dataB.searchForResult(diff, list, FinalCandidates,
                                             p_tolerance_->getValue(), 7);

    if (path == NULL)
    {
        sendWarning("Could not find a knob with the given parameters");
        return 2;
    }
    getPath = path;
    return (diff != 0.0);
}

void
ControlSCA::SetIntSliderValue(coIntSliderParam *param, int num)
{
    if (param->getMin() > num)
        param->setMin(num);
    if (param->getMax() < num)
        param->setMax(num);
    param->setValue(num);
}

void
ControlSCA::SetFloatSliderValue(coFloatSliderParam *param, float val)
{
    if (param->getMin() > val)
        param->setMin(val);
    if (param->getMax() < val)
        param->setMax(val);
    param->setValue(val);
}

void
ControlSCA::RestoreOldValues()
{
    int par;
    for (par = 0; par < parList_.size(); ++par)
    {
        parList_[par]->setValue(old_floats_[par]);
    }
}

void
ControlSCA::loadOldFloats()
{
    int par;
    for (par = 0; par < parList_.size(); ++par)
    {
        old_floats_[par] = parList_[par]->getValue();
    }
}

void
ControlSCA::setNewValues(string &pathS, bool abgekuerzt)
{
    static const char *abkuerzungen[] = {
        "nH=", "rdRad=", "abRad=", "knAng=",
        "lng1=", "lng2="
    };
    const char *path = pathS.c_str();
    int par;
    for (par = 0; par < parList_.size(); ++par)
    {
        const char *thisParam = NULL;
        if (abgekuerzt)
        {
            thisParam = strstr(path, abkuerzungen[par]);
        }
        else
        {
            thisParam = strstr(path, parList_[par]->getName());
        }
        thisParam = strstr(thisParam, "=");
        ++thisParam;
        float number;
        sscanf(thisParam, "%g", &number);
        old_floats_[par] = parList_[par]->getValue();
        if (number < parList_[par]->getMin())
        {
            parList_[par]->setMin(number);
        }
        if (number > parList_[par]->getMax())
        {
            parList_[par]->setMax(number);
        }
        parList_[par]->setValue(number);
    }
}

void
ControlSCA::outputDummies()
{
    outputKnob(NO_EXECUTE);
    outputDesign(NO_EXECUTE);
    outputPraegung(NO_EXECUTE);
    outputZug(NO_EXECUTE);
}

bool
ControlSCA::isPossible(int choice)
{
    int i;
    for (i = 0; i < whatIsPossible_.size(); ++i)
    {
        if (choice == whatIsPossible_[i])
        {
            return true;
        }
    }
    cerr << "++++++++++++ Possible options +++++++++++++++++" << endl;
    for (i = 0; i < whatIsPossible_.size(); ++i)
    {
        cerr << "Option: " << whatIsPossible_[i] << endl;
    }
    cerr << "You tried option: " << choice << endl;
    cerr << "+++++++++++++++++++++++++++++++++++++++++++++++" << endl;
    return false;
}

void
ControlSCA::writeChoices(string &mssg)
{
    static const char *beschr[] = {
        "leaf size", "knob geometry",
        "design", "embossing", "visualisation",
        "traction",
        "ANSYS results"
    };
    mssg = "Sorry: acceptable tasks are: ";
    int i;
    for (i = 0; i < whatIsPossible_.size(); ++i)
    {
        mssg += beschr[i];
        if (i != whatIsPossible_.size() - 1)
        {
            mssg += ", ";
        }
    }
    mssg += ".";
}

void
ControlSCA::setWhatIsPossible()
{
    whatIsPossible_.clear();
    switch (p_whatExecute_->getValue())
    {
    case NO_EXECUTE:
        whatIsPossible_.push_back(NO_EXECUTE);
        whatIsPossible_.push_back(KNOB_SELECT);
        break;
    case KNOB_SELECT:
        whatIsPossible_.push_back(NO_EXECUTE);
        whatIsPossible_.push_back(KNOB_SELECT);
        whatIsPossible_.push_back(DESIGN);
        break;
    case DESIGN:
        whatIsPossible_.push_back(NO_EXECUTE);
        whatIsPossible_.push_back(KNOB_SELECT);
        whatIsPossible_.push_back(DESIGN);
        whatIsPossible_.push_back(PRAEGUNG);
        break;
    case PRAEGUNG:
        whatIsPossible_.push_back(NO_EXECUTE);
        whatIsPossible_.push_back(KNOB_SELECT);
        whatIsPossible_.push_back(DESIGN);
        whatIsPossible_.push_back(PRAEGUNG);
        whatIsPossible_.push_back(VERNETZUNG);
        break;
    case VERNETZUNG:
        whatIsPossible_.push_back(NO_EXECUTE);
        whatIsPossible_.push_back(KNOB_SELECT);
        whatIsPossible_.push_back(DESIGN);
        whatIsPossible_.push_back(PRAEGUNG);
        whatIsPossible_.push_back(VERNETZUNG);
        whatIsPossible_.push_back(ZUG);
        break;
    case ZUG:
        whatIsPossible_.push_back(NO_EXECUTE);
        whatIsPossible_.push_back(KNOB_SELECT);
        whatIsPossible_.push_back(DESIGN);
        whatIsPossible_.push_back(PRAEGUNG);
        whatIsPossible_.push_back(VERNETZUNG);
        whatIsPossible_.push_back(ZUG);
        whatIsPossible_.push_back(READ_ANSYS);
        break;
    default:
        whatIsPossible_.push_back(NO_EXECUTE);
        whatIsPossible_.push_back(KNOB_SELECT);
        whatIsPossible_.push_back(DESIGN);
        whatIsPossible_.push_back(PRAEGUNG);
        whatIsPossible_.push_back(VERNETZUNG);
        whatIsPossible_.push_back(ZUG);
        whatIsPossible_.push_back(READ_ANSYS);
        break;
    }
}

int
ControlSCA::compute(const char *port)
{
    (void)port; // silence compiler

    if (!isPossible(p_whatExecute_->getValue()))
    {
        string mssg;
        writeChoices(mssg);
        sendInfo("%s", mssg.c_str());
        return FAIL;
    }
    if (p_tolerance_->getValue() <= 0.0 || p_tolerance_->getValue() > 1.0)
    {
        sendWarning("Relative tolerance should lie between 0.0 and 1.0");
        return FAIL;
    }
    p_DataBaseShape_->setValue(0);
    sendInfo("______________");
    switch (p_SheetDimensions_->getValue())
    {
    case 0:
        SetFloatSliderValue(p_blatHoehe_, 125);
        SetFloatSliderValue(p_blatBreite_, 98);
        break;
    case 1:
        SetFloatSliderValue(p_blatHoehe_, 140);
        SetFloatSliderValue(p_blatBreite_, 125);
        break;
    case 2:
        SetFloatSliderValue(p_blatHoehe_, 125);
        SetFloatSliderValue(p_blatBreite_, 110);
        break;
    case 3:
        SetFloatSliderValue(p_blatHoehe_, 246);
        SetFloatSliderValue(p_blatBreite_, 260);
        break;
    case 4:
        SetFloatSliderValue(p_blatHoehe_, 246);
        SetFloatSliderValue(p_blatBreite_, 230);
        break;
    default:
        break;
    }

    if (p_whatExecute_->getValue() == 0)
    {
        outputDummies();
        outputBorder(p_whatExecute_->getValue());
        outputInteractor();
        setWhatIsPossible();
        ImageToTexture();
        return SUCCESS;
    }

    int whatExecute = p_whatExecute_->getValue();
    outputBorder(whatExecute);
    outputKnob(whatExecute);
    outputDesign(whatExecute);
    outputPraegung(whatExecute);
    outputZug(whatExecute);
    outputInteractor();
    ImageToTexture();

    setWhatIsPossible();
    if (p_whatExecute_->getValue() < PRAEGUNG && p_whatExecute_->getValue() != PRAEGUNG && p_whatExecute_->getValue() != VERNETZUNG)
    {
        oldStartEmbossing_ = false;
    }
    return SUCCESS;
}

void
ControlSCA::outputBorder(int whatExecute)
{
    // produce border line
    if (whatExecute > 0)
    {
        coDoLines *border = new coDoLines(p_blat_->getObjName(), 0, 0, 0);
        p_blat_->setCurrentObject(border);
        return;
    }
    coDoLines *border = new coDoLines(p_blat_->getObjName(), 4, 5, 1);
    int *ll, *vl;
    float *xc, *yc, *zc;
    border->getAddresses(&xc, &yc, &zc, &vl, &ll);
    ll[0] = 0;
    vl[0] = 0;
    vl[1] = 1;
    vl[2] = 2;
    vl[3] = 3;
    vl[4] = 0;
    xc[0] = 0.0;
    yc[0] = 0.0;
    zc[0] = 0.0;
    xc[1] = p_blatBreite_->getValue();
    yc[1] = 0.0;
    zc[1] = 0.0;
    xc[2] = p_blatBreite_->getValue();
    yc[2] = p_blatHoehe_->getValue();
    zc[2] = 0.0;
    xc[3] = 0.0;
    yc[3] = p_blatHoehe_->getValue();
    zc[3] = 0.0;
    p_blat_->setCurrentObject(border);
}

#include <api/coFeedback.h>

static void
addFeedbackParams(coFeedback &feedback, std::vector<coUifPara *> &params)
{
    int param;
    for (param = 0; param < params.size(); ++param)
    {
        feedback.addPara(params[param]);
    }
}

void
ControlSCA::outputInteractor()
{
    coDoPoints *dummy = new coDoPoints(p_COVERInteractor_->getObjName(), 1);
    float *x_coord, *y_coord, *z_coord;
    dummy->getAddresses(&x_coord, &y_coord, &z_coord);
    *x_coord = 0.0;
    *y_coord = 0.0;
    *z_coord = 0.0;
    coFeedback feedback("SCA");
    feedback.addPara(p_whatExecute_);
    feedback.addPara(p_blatHoehe_);
    feedback.addPara(p_blatBreite_);
    feedback.addPara(p_SheetDimensions_);
    feedback.addPara(p_DataBaseStrategy_);
    feedback.addPara(p_DataBaseShape_);
    addFeedbackParams(feedback, p_design_);
    addFeedbackParams(feedback, p_praegung_);
    addFeedbackParams(feedback, p_zug_);
    addFeedbackParams(feedback, p_readANSYS_);
    feedback.apply(dummy);
    p_COVERInteractor_->setCurrentObject(dummy);
}

bool
ControlSCA::isAcceptable(Candidate &cand, float abweichung)
{
    if (cand.getDiff() == FLT_MAX || cand.getDiff() == 0.0)
    {
        return false;
    }
    std::vector<float> &diffArray = cand.getDiffArray();
    int param;
    for (param = 0; param < diffArray.size(); ++param)
    {
        if (diffArray[param] > abweichung)
        {
            return false;
        }
    }
    return true;
}

void
ControlSCA::addToEntries(std::vector<string *> &entries, Candidate &cand)
{
    // cerr << cand.getWholePath() << endl;
    const char *path = cand.getWholePath().c_str();
    char *buf = new char[strlen(path)];
    // find the first '='
    int param = 0;
    static const char *abkuerzungen[] = {
        "nH=", "rdRad=", "abRad=", "knAng=",
        "Frm=", "lng1=", "lng2="
    };
    const char *start = strstr(path, "=");
    ++start;
    const char *end = strstr(start, "/");
    if (!end)
    {
        delete[] buf;
        return;
    }
    strcpy(buf, abkuerzungen[param]);
    strncat(buf, start, end - start);
    strcat(buf, ", ");
    // entries.push_back(new coString(buf));
    while (1)
    {
        // look for the next
        path = end + 1;
        if (!(*path))
        {
            break;
        }
        ++param;
        if (param == 4) // jump over knob shape
        {
            end = strstr(path, "/");
            if (!end)
            {
                break;
            }
        }
        start = strstr(path, "=");
        if (!start)
        {
            break;
        }
        ++start;
        end = strstr(start, "/");
        if (!end)
        {
            break;
        }
        strcat(buf, abkuerzungen[param]);
        strncat(buf, start, end - start);
        // entries.push_back(new coString(buf));
        if (param < 6)
        {
            strcat(buf, ", ");
        }
        else
        {
            break;
        }
    }
    entries.push_back(new string(buf));
    delete[] buf;
}

void
ControlSCA::addToFullEntries(std::vector<string *> &entries, Candidate &cand)
{
    // cerr << cand.getWholePath() << endl;
    const char *path = cand.getWholePath().c_str();
    char *buf = new char[strlen(path)];
    // find the first '='
    int param = 0;
    static const char *abkuerzungen[] = { "gummiHaerte=", "anpressDruck=" };
    const char *start = strstr(path, "gummiHaerte=");
    start = strstr(start, "=");
    ++start;
    const char *end = strstr(start, "/");
    if (!end)
    {
        delete[] buf;
        return;
    }
    strcpy(buf, abkuerzungen[param]);
    strncat(buf, start, end - start);
    strcat(buf, ", ");
    // entries.push_back(new coString(buf));
    while (1)
    {
        // look for the next
        path = end + 1;
        if (!(*path))
        {
            break;
        }
        ++param;
        start = strstr(path, "=");
        if (!start)
        {
            break;
        }
        ++start;
        end = strstr(start, "/");
        if (!end)
        {
            break;
        }
        strcat(buf, abkuerzungen[param]);
        strncat(buf, start, end - start);
        // entries.push_back(new coString(buf));
        if (param < 1)
        {
            strcat(buf, ", ");
        }
        else
        {
            break;
        }
    }
    entries.push_back(new string(buf));
    delete[] buf;
}

void
ControlSCA::dynamicList(std::vector<Candidate *> &FinalCandidates)
{
    int cand;
    std::vector<string *> entries;
    if (knob_found_ != 0)
    {
        entries.push_back(new string("Knob geometry not in DB"));
    }
    else
    {
        entries.push_back(new string("Knob geometry available in DB"));
    }
    for (cand = 0; cand < FinalCandidates.size(); ++cand)
    {
        if (isAcceptable(*FinalCandidates[cand], p_tolerance_->getValue()))
        {
            addToEntries(entries, *FinalCandidates[cand]);
        }
    }
    // use entries for a choice list
    const char **list = new const char *[entries.size()];

    int entry;
    for (entry = 0; entry < entries.size(); ++entry)
    {
        list[entry] = entries[entry]->c_str();
    }
    p_DataBaseShape_->setValue(entries.size(), list, 0);
    delete[] list;
    // destroy coString in entries
    for (entry = 0; entry < entries.size(); ++entry)
    {
        delete entries[entry];
    }
}

void
ControlSCA::dynamicFullList(std::vector<Candidate *> &FinalCandidates)
{
    int cand;
    std::vector<string *> entries;
    if (full_knob_found_ != 0)
    {
        entries.push_back(new string("Embossing not in DB"));
    }
    else
    {
        entries.push_back(new string("Embossing available in DB"));
    }
    for (cand = 0; cand < FinalCandidates.size(); ++cand)
    {
        if (isAcceptable(*FinalCandidates[cand], FLT_MAX * 1.0e-9))
        {
            addToFullEntries(entries, *FinalCandidates[cand]);
        }
    }
    // use entries for a choice list
    const char **list = new const char *[entries.size()];

    int entry;
    for (entry = 0; entry < entries.size(); ++entry)
    {
        list[entry] = entries[entry]->c_str();
    }
    p_DataBaseStrategy_->enable();
    p_DataBaseStrategy_->setValue(entries.size(), list, 0);
    delete[] list;
    // destroy coString in entries
    for (entry = 0; entry < entries.size(); ++entry)
    {
        delete entries[entry];
    }
}

float
ControlSCA::BumpHeight(std::vector<Candidate *> &FinalCandidates)
{
    float minDiff = FLT_MAX;
    Candidate *best = NULL;
    for (int i = 0; i < FinalCandidates.size(); ++i)
    {
        if (minDiff > FinalCandidates[i]->getDiff())
        {
            best = FinalCandidates[i];
            minDiff = FinalCandidates[i]->getDiff();
        }
    }
    if (best)
    {
        string mssg(best->getPath());
        if (mssg.length() == 0)
        {
            mssg = "./";
        }
        else if (mssg.c_str()[mssg.length() - 1] != '/')
        {
            mssg += '/';
        }

        // now read the 1/4 knob
        string conn = mssg.c_str();
        conn += "topology.k";
        string disp = mssg.c_str();
        disp += "defgeo";
        string disp_exp = mssg.c_str();
        disp_exp += "defgeo_exp";
        string thick = mssg.c_str();
        thick += "movie100.s30";

        ifstream emb_conn(conn.c_str());
        ifstream emb_displ(disp.c_str());
        ifstream emb_displ_exp(disp_exp.c_str());
        ifstream emb_thick(thick.c_str());
        if (!emb_conn.rdbuf()->is_open())
        {
            //sendWarning("ControlSCA: Could not open connectivity embossing file");
            return 0.0;
        }
        if (!emb_displ.rdbuf()->is_open())
        {
            //sendWarning("ControlSCA: Could not open displacements embossing file (impl)");
            return 0.0;
        }
        if (!emb_displ_exp.rdbuf()->is_open())
        {
            //sendWarning("ControlSCA: Could not open displacements embossing file (expl)");
            return 0.0;
        }
        if (!emb_thick.rdbuf()->is_open())
        {
            //sendWarning("ControlSCA: Could not open thickness embossing file");
            return 0.0;
        }
        vector<int> epl, ecl;
        vector<float> exc;
        vector<float> eyc;
        vector<float> ezc;
        vector<float> dicke;
        if (DatabaseUtils::readEmbossingResults(
                emb_conn, emb_displ, emb_displ_exp, emb_thick,
                epl, ecl, exc, eyc, ezc, dicke) != 0)
        {
            //sendWarning("ControlSCA: Could not successfully read embossing results");
            return 0.0;
        }
        // now calculate max and min in ezc
        vector<float>::iterator min, max;
        min = std::min_element(ezc.begin(), ezc.end());
        max = std::max_element(ezc.begin(), ezc.end());
        if (min != ezc.end() && max != ezc.end())
        {
            return ((*max) - (*min));
        }
    }
    return 0.0;
}
