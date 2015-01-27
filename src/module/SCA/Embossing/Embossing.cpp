/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Embossing.h"
#include <alg/DeleteUnusedPoints.h>
#include <alg/MagmaUtils.h>
#include <do/coDoText.h>
#include <DatabaseUtils.h>
#include <iomanip>
#include <sys/types.h>
#include <unistd.h>
#include <sys/wait.h>
#include <string>

using std::setw;

Embossing::Embossing(int argc, char *argv[])
    : coModule(argc, argv, "SCA Embossing")
{
    p_PraegeParam_ = addInputPort("PraegeParam", "coDoText", "Embossing parameters");
    p_domain_ = addInputPort("Domain", "coDoPolygons", "Basic cell domain");
    p_points_ = addInputPort("NoppenPoints", "coDoPoints", "Knob positions");
    p_colors_ = addInputPort("NoppenColors", "coDoFloat", "Design rules");
    p_wait_ = addInputPort("WaitForLSDyna", "coDoText", "Wait for LS-DYNA simulation");

    p_VernetzteGrundZelle_ = addOutputPort("VernetzteGrundZelle", "coDoPolygons", "Vernetze Grundzelle");
    p_permitTraction_ = addOutputPort("permitTraction", "coDoText", "permit traction");
    p_image_ = addOutputPort("Image", "coDoText", "Image");

    permitTraction_ = false;

    trueData_ = false;

    const char *sca_path = getenv("SCA_PATH");
    char currdir[1024];
    if (sca_path)
    {
        simDir_ = sca_path;
    }
    else if (getcwd(currdir, 1024))
    {
        simDir_ = currdir;
    }

    int len = simDir_.length();
    if (len > 0 && simDir_[len - 1] != '/')
    {
        simDir_ += '/';
    }
    simDir_ += "VISUALISATION/";
}

Embossing::~Embossing()
{
}

int
Embossing::readIntSlider(istringstream &strText, int *addr)
{
    std::vector<char> Value(maxLen_);
    strText >> &Value[0];
    strText >> &Value[0];
    if (!(strText >> *addr))
    {
        Covise::sendWarning("Error on reading IntSlider from istringstream");
        return -1;
    }
    return 0;
}

int
Embossing::readFloatSlider(istringstream &strText, float *addr)
{
    std::vector<char> Value(maxLen_);
    strText >> &Value[0];
    strText >> &Value[0];
    if (!(strText >> *addr))
    {
        Covise::sendWarning("Error on reading FloatSlider from istringstream");
        return -1;
    }
    return 0;
}

int
Embossing::readChoice(istringstream &strText, int *addr)
{
    std::vector<char> Value(maxLen_);
    strText >> &Value[0];
    if (!(strText >> *addr))
    {
        Covise::sendWarning("Error on reading IntSlider from istringstream");
        return -1;
    }
    strText >> &Value[0];
    return 0;
}

void
Embossing::setBooleanFalse()
{
    readNoppenHoehe_ = false;
    readAusrundungsRadius_ = false;
    readAbnutzungsRadius_ = false;
    readNoppenWinkel_ = false;
    readNoppenForm_ = false;
    readLaenge1_ = false;
    readLaenge2_ = false;
    readGummiHaerte_ = false;
    readAnpressDruck_ = false;
    readAnzahlReplikationenX_ = false;
    readAnzahlReplikationenY_ = false;
    readPresentationMode_ = false;
}

bool
Embossing::gotDummies()
{
    const coDistributedObject *is_done = p_wait_->getCurrentObject();
    if (!is_done || !is_done->isType("DOTEXT"))
    {
        return true;
    }
    coDoText *isDone = (coDoText *)is_done;
    char *text;
    int siz = isDone->getTextLength();
    isDone->getAddress(&text);
    if (siz == 0 || strncmp(text, "Done", 4) != 0)
    {
        return true;
    }

    const coDistributedObject *inPol = p_domain_->getCurrentObject();
    const coDistributedObject *inPoints = p_points_->getCurrentObject();
    if (inPol == NULL || !inPol->objectOk())
    {
        sendWarning("Got NULL pointer or polygon is not OK");
        return true;
    }
    if (inPoints == NULL || !inPoints->objectOk())
    {
        sendWarning("Got NULL pointer or points are not OK");
        return true;
    }
    if (4 > dynamic_cast<const coDoPolygons *>(inPol)->getNumPoints())
    {
        return true;
    }
    float *xc, *yc, *zc;
    no_points_ = dynamic_cast<const coDoPoints *>(inPoints)->getNumPoints();
    if (no_points_ <= 0)
    {
        return true;
    }
    dynamic_cast<const coDoPoints *>(inPoints)->getAddresses(&xc, &yc, &zc);
    int point;
    xp_.clear();
    yp_.clear();
    for (point = 0; point < no_points_; ++point)
    {
        xp_.push_back(xc[point]);
        yp_.push_back(yc[point]);
    }

    int *vl, *pl;
    dynamic_cast<const coDoPolygons *>(inPol)->getAddresses(&xc, &yc, &zc, &vl, &pl);
    width_ = xc[2];
    height_ = yc[2];

    return false;
}

#include <float.h>

#include "ReadASCIIDyna.h"

void
Embossing::outputMssg()
{
    std::string tractionMssg;
    if (permitTraction_)
    {
        tractionMssg = "Go";
    }
    else
    {
        tractionMssg = "Do not go";
    }
    coDoText *Mssg = new coDoText(p_permitTraction_->getObjName(),
                                  tractionMssg.length() + 1);
    char *text;
    Mssg->getAddress(&text);
    strcpy(text, tractionMssg.c_str());
    p_permitTraction_->setCurrentObject(Mssg);
}

int
Embossing::compute(const char *port)
{
    (void)port; // silence compiler

    barrelDiameter_ = 0.0;

    if (gotDummies())
    {
        outputDummies();
        return SUCCESS;
    }

    setBooleanFalse();
    // get the text object
    const coDistributedObject *inObj = p_PraegeParam_->getCurrentObject();
    if (inObj == NULL || !inObj->objectOk())
    {
        sendWarning("Got NULL pointer or object is not OK");
        permitTraction_ = false;
        return FAIL;
    }
    if (!inObj->isType("DOTEXT"))
    {
        sendWarning("Only coDoText is acceptable for input");
        permitTraction_ = false;
        return FAIL;
    }
    const coDoText *theText = dynamic_cast<const coDoText *>(inObj);
    int size = theText->getTextLength();
    if (size == 0)
    {
        outputDummies();
        return SUCCESS;
    }

    // be pessimistic
    permitTraction_ = false;

    char *text;
    theText->getAddress(&text);
    istringstream strText;
    strText.str(text);
    maxLen_ = strlen(text) + 1;
    std::vector<char> name(maxLen_);
    while (strText >> &name[0])
    {
        if (strcmp(&name[0], "noppenHoehe") == 0)
        {
            if (readFloatSlider(strText, &noppenHoehe_) != 0)
            {
                sendWarning("Could not read noppenHoehe");
                return FAIL;
            }
            readNoppenHoehe_ = true;
        }
        else if (strcmp(&name[0], "ausrundungsRadius") == 0)
        {
            if (readFloatSlider(strText, &ausrundungsRadius_) != 0)
            {
                sendWarning("Could not read rundZellenBreite");
                return FAIL;
            }
            readAusrundungsRadius_ = true;
        }
        else if (strcmp(&name[0], "abnutzungsRadius") == 0)
        {
            if (readFloatSlider(strText, &abnutzungsRadius_) != 0)
            {
                sendWarning("Could not read abnutzungsRadius");
                return FAIL;
            }
            readAbnutzungsRadius_ = true;
        }
        else if (strcmp(&name[0], "noppenWinkel") == 0)
        {
            if (readFloatSlider(strText, &noppenWinkel_) != 0)
            {
                sendWarning("Could not read noppenWinkel");
                return FAIL;
            }
            readNoppenWinkel_ = true;
        }
        else if (strcmp(&name[0], "noppenForm") == 0)
        {
            if (readChoice(strText, &noppenForm_) != 0)
            {
                sendWarning("Could not read noppenForm");
                return FAIL;
            }
            if (noppenForm_ == 3) // Circle
            {
                --noppenForm_;
            }
            readNoppenForm_ = true;
        }
        else if (strcmp(&name[0], "laenge1") == 0)
        {
            if (readFloatSlider(strText, &laenge1_) != 0)
            {
                sendWarning("Could not read laenge1");
                return FAIL;
            }
            readLaenge1_ = true;
        }
        else if (strcmp(&name[0], "laenge2") == 0)
        {
            if (readFloatSlider(strText, &laenge2_) != 0)
            {
                sendWarning("Could not read laenge2");
                return FAIL;
            }
            readLaenge2_ = true;
        }
        else if (strcmp(&name[0], "tissueTyp") == 0)
        {
            if (readChoice(strText, &tissueTyp_) != 0)
            {
                sendWarning("Could not read tissueTyp");
                return FAIL;
            }
            readTissueTyp_ = true;
        }
        else if (strcmp(&name[0], "gummiHaerte") == 0)
        {
            if (readFloatSlider(strText, &gummiHaerte_) != 0)
            {
                sendWarning("Could not read gummiHaerte");
                return FAIL;
            }
            readGummiHaerte_ = true;
        }
        else if (strcmp(&name[0], "anpressDruck") == 0)
        {
            if (readFloatSlider(strText, &anpressDruck_) != 0)
            {
                sendWarning("Could not read anpressDruck");
                return FAIL;
            }
            readAnpressDruck_ = true;
        }
        else if (strcmp(&name[0], "anzahlReplikationenX") == 0)
        {
            if (readIntSlider(strText, &anzahlReplikationenX_) != 0)
            {
                sendWarning("Could not read anzahlReplikationenX");
                return FAIL;
            }
            readAnzahlReplikationenX_ = true;
        }
        else if (strcmp(&name[0], "anzahlReplikationenY") == 0)
        {
            if (readIntSlider(strText, &anzahlReplikationenY_) != 0)
            {
                sendWarning("Could not read anzahlReplikationenY");
                return FAIL;
            }
            readAnzahlReplikationenY_ = true;
        }
        else if (strcmp(&name[0], "presentationMode") == 0)
        {
            if (readChoice(strText, &presentationMode_) != 0)
            {
                sendWarning("Could not read presentationMode");
                return FAIL;
            }
            readPresentationMode_ = true;
        }
        else if (strcmp(&name[0], "barrelDiameter") == 0)
        {
            strText >> barrelDiameter_;
            strText >> barrelDiameter_;
            strText >> barrelDiameter_;
            if (barrelDiameter_ == 0.0)
            {
                sendError("barrelDiameter is 0");
            }
        }
    }
    if (checkReadFlags() != 0)
    {
        return FAIL;
    }

    // related to reusing previous meshing
    bool skip_launch_ansys = false;
    if (trueData_ && SameData() && SameDesign() && FileExists())
    {
        skip_launch_ansys = true;
    }
    else
    {
        trueData_ = false;
    }

    std::string embConnFile;
    std::string embDispFile;
    std::string embDispExpFile;
    std::string embThickFile;

    if (loadFileNames(embConnFile, embDispFile, embDispExpFile, embThickFile) != 0)
    {
        sendWarning("Could not decide/find results directory or files");
        return FAIL;
    }

    // now read the 1/4 knob
    ifstream emb_conn(embConnFile.c_str());
    ifstream emb_displ(embDispFile.c_str());
    ifstream emb_displ_exp(embDispExpFile.c_str());
    ifstream emb_thick(embThickFile.c_str());
    //cerr << "Pfad: "<< embConnFile.c_str() <<endl;

    if (!emb_conn.rdbuf()->is_open())
    {
        sendWarning("Could not open connectivity embossing file");
        return FAIL;
    }
    if (!emb_displ.rdbuf()->is_open())
    {
        sendWarning("Could not open displacements embossing file (impl)");
        return FAIL;
    }
    if (!emb_displ_exp.rdbuf()->is_open())
    {
        sendWarning("Could not open displacements embossing file (expl)");
        return FAIL;
    }
    if (!emb_thick.rdbuf()->is_open())
    {
        sendWarning("Could not open thickness embossing file");
        return FAIL;
    }

    vector<int> epl, ecl;
    vector<float> exc;
    vector<float> eyc;
    vector<float> ezc;
    vector<float> dicke;
    if (1)
    {
        if (DatabaseUtils::readEmbossingResults(emb_conn, emb_displ, emb_displ_exp, emb_thick,
                                                epl, ecl, exc, eyc, ezc, dicke) != 0)
        {
            sendWarning("Could not successfully read embossing results");
            return FAIL;
        }
    }
    else
    {
        vector<float> nxc;
        vector<float> nyc;
        vector<float> nzc;
        if (noppenForm_ == 2)
        {
            ReadASCIIDyna::ellipticKnob(noppenHoehe_, ausrundungsRadius_,
                                        abnutzungsRadius_, noppenWinkel_, laenge1_, laenge2_, 3, 4, 4,
                                        epl, ecl, exc, eyc, ezc, nxc, nyc, nzc);
        }
        else if (noppenForm_ == 1)
        {
            ReadASCIIDyna::rectangularKnob(noppenHoehe_, ausrundungsRadius_,
                                           abnutzungsRadius_, noppenWinkel_, laenge1_, laenge2_, 3, 4, 4,
                                           epl, ecl, exc, eyc, ezc, nxc, nyc, nzc);
        }
        int point;
        for (point = 0; point < exc.size(); ++point)
        {
            dicke.push_back(0.1);
        }
    }

    const coDistributedObject *colors = p_colors_->getCurrentObject();
    if (!colors->isType("USTSDT"))
    {
        sendWarning("Expected a scalar field for checking design rules");
        return FAIL;
    }
    coDoFloat *data_colors = (coDoFloat *)colors;
    int color_count;
    float *design_rules = NULL;
    data_colors->getAddress(&design_rules);
    for (color_count = 0; color_count < data_colors->getNumPoints(); ++color_count)
    {
        if (design_rules[color_count] != 0)
        {
            sendWarning("Actual design violates some design rules");
            return FAIL;
        }
    }

    // Mirror2 will modify the arrays, but we shall
    // need them below for the calculation of the thickness.
    // That is why we make copies now.
    vector<float> txc, tyc;
    int node;
    for (node = 0; node < exc.size(); ++node)
    {
        txc.push_back(exc[node]);
        tyc.push_back(eyc[node]);
    }
    vector<int> tcl, tpl;
    int vertex;
    for (vertex = 0; vertex < ecl.size(); ++vertex)
    {
        tcl.push_back(ecl[vertex]);
    }
    int elem;
    for (elem = 0; elem < epl.size(); ++elem)
    {
        tpl.push_back(epl[elem]);
    }
    Mirror2(exc, eyc, ezc, ecl, epl, X);
    Mirror2(exc, eyc, ezc, ecl, epl, Y);
    Mirror2(txc, tyc, dicke, tcl, tpl, X);
    Mirror2(txc, tyc, dicke, tcl, tpl, Y);

    std::string auxName = p_VernetzteGrundZelle_->getObjName();
    auxName += "_Aux";
    std::string auxNameThick = p_VernetzteGrundZelle_->getObjName();
    auxNameThick += "_Aux_Thick";

    coDoPolygons *auxCell = NULL, *auxThickCell = NULL;
    {
        float *x, *y, *z;
        int *v, *p;
        auxCell = new coDoPolygons(auxName, exc.size(), ecl.size(), epl.size());
        auxCell->getAddresses(&x, &y, &z, &v, &p);
        copy(exc.begin(), exc.end(), x);
        copy(eyc.begin(), eyc.end(), y);
        copy(ezc.begin(), ezc.end(), z);
        copy(ecl.begin(), ecl.end(), v);
        copy(epl.begin(), epl.end(), p);

        auxThickCell = new coDoPolygons(auxNameThick, txc.size(), tcl.size(), tpl.size());
        auxThickCell->getAddresses(&x, &y, &z, &v, &p);
        copy(txc.begin(), txc.end(), x);
        copy(tyc.begin(), tyc.end(), y);
        copy(dicke.begin(), dicke.end(), z);
        copy(tcl.begin(), tcl.end(), v);
        copy(tpl.begin(), tpl.end(), p);
    }

    std::string auxFullName = p_VernetzteGrundZelle_->getObjName();
    auxFullName += "_Aux_Full";
    std::string auxThickFullName = p_VernetzteGrundZelle_->getObjName();
    auxThickFullName += "_Aux_Thick_Full";

    coDistributedObject *basicCell = checkUSG(auxCell, auxFullName);
    coDistributedObject *basicThickCell = checkUSG(auxThickCell, auxThickFullName);
    delete auxCell;
    delete auxThickCell;

    // this is OK as long as thickness is not necessary...
    // In order to get the correct thickness after mirroring
    // and checkUSG we use an old
    // indian trick -> think that the thickness is a Z-coordinate
    coDoPolygons *basicPolys = dynamic_cast<coDoPolygons *>(basicCell);
    coDoPolygons *basicThickPolys = dynamic_cast<coDoPolygons *>(basicThickCell);
    if (inObj->getAttribute("VERNETZUNG"))
    {
        if (basicPolys->getNumPoints()
            != basicThickPolys->getNumPoints())
        {
            sendWarning("The number of points collected from 'thickness' is not correct");
            return FAIL;
        }

        if (!skip_launch_ansys)
        {
            if (ANSYSInputAndLaunch(dynamic_cast<coDoPolygons *>(basicCell),
                                    dynamic_cast<coDoPolygons *>(basicThickCell)) != 0)
            {
                sendWarning("Could not create ANSYS Input");
                delete basicCell;
                delete basicThickCell;
                return FAIL;
            }
        }
    }
    else // show a bump
    {
        // copy basicPolys
        int no_of_polygons = basicPolys->getNumPolygons();
        int no_of_vertices = basicPolys->getNumVertices();
        int no_of_points = basicPolys->getNumPoints();
        coDoPolygons *bump = new coDoPolygons(p_VernetzteGrundZelle_->getObjName(),
                                              no_of_points, no_of_vertices, no_of_polygons);
        float *x_c, *y_c, *z_c;
        int *v_l, *p_l;
        basicPolys->getAddresses(&x_c, &y_c, &z_c, &v_l, &p_l);
        float *x_n, *y_n, *z_n;
        int *v_n, *p_n;
        bump->getAddresses(&x_n, &y_n, &z_n, &v_n, &p_n);
        memcpy(x_n, x_c, no_of_points * sizeof(float));
        memcpy(y_n, y_c, no_of_points * sizeof(float));
        memcpy(z_n, z_c, no_of_points * sizeof(float));
        memcpy(v_n, v_l, no_of_vertices * sizeof(int));
        memcpy(p_n, p_l, no_of_polygons * sizeof(int));
        // set "dummy" TRANSFORM attribute
        char buf[64];
        string Transform = "Transform: ";
        sprintf(buf, "%d\n", 3); // translation
        Transform += buf;
        bump->addAttribute("vertexOrder", "2");
        bump->addAttribute("TRANSFORM", Transform.c_str());
        p_VernetzteGrundZelle_->setCurrentObject(bump);
    }

    delete basicCell;
    delete basicThickCell;

    if (inObj->getAttribute("VERNETZUNG") && readAndOutputMesh() != 0)
    {
        sendWarning("Could not create polygons out of the ANSYS results");
        return FAIL;
    }
    else if (inObj->getAttribute("PRAEGUNG"))
    {
        permitTraction_ = false;
        outputMssg();
    }

    std::string imageName;
    getImageName(imageName);
    coDoText *txtObj = new coDoText(p_image_->getObjName(),
                                    imageName.length() + 1);
    txtObj->getAddress(&text);
    strcpy(text, imageName.c_str());

    // wir wollen einen "Beleg" von dieser Vernetzung
    // trueData_ = true;
    if (inObj->getAttribute("VERNETZUNG"))
    {
        keepInputData();
        loadDesign();
    }

    return SUCCESS;
}

void
Embossing::keepInputData()
{
    trueData_ = true;
    l_noppenHoehe_ = noppenHoehe_;
    l_ausrundungsRadius_ = ausrundungsRadius_;
    l_abnutzungsRadius_ = abnutzungsRadius_;
    l_noppenWinkel_ = noppenWinkel_;
    l_noppenForm_ = noppenForm_;
    l_laenge1_ = laenge1_;
    l_laenge2_ = laenge2_;
    l_tissueTyp_ = tissueTyp_;
    l_gummiHaerte_ = gummiHaerte_;
    l_anpressDruck_ = anpressDruck_;
}

#include <sys/types.h>
#include <sys/stat.h>

bool
Embossing::FileExists()
{
    struct stat buf;
    std::string path = simDir_;
    path += "zugmesh.db";
    if (stat(path.c_str(), &buf) != 0)
    {
        return false;
    }
    return true;
}

bool
Embossing::SameDesign()
{
    if (l_width_ != width_ || l_height_ != height_)
    {
        return false;
    }
    if (l_xp_.size() != xp_.size() || l_yp_.size() != yp_.size())
    {
        return false;
    }
    int punkt;
    for (punkt = 0; punkt < xp_.size(); ++punkt)
    {
        if (l_xp_[punkt] != xp_[punkt] || l_yp_[punkt] != yp_[punkt])
        {
            return false;
        }
    }

    return true;
}

void
Embossing::loadDesign()
{
    l_width_ = width_;
    l_height_ = height_;
    l_xp_.clear();
    l_yp_.clear();
    int punkt;
    for (punkt = 0; punkt < xp_.size(); ++punkt)
    {
        l_xp_[punkt] = xp_[punkt];
        l_yp_[punkt] = yp_[punkt];
    }
}

bool
Embossing::SameData()
{
    return (l_noppenHoehe_ == noppenHoehe_ && l_ausrundungsRadius_ == ausrundungsRadius_ && l_abnutzungsRadius_ == abnutzungsRadius_ && l_noppenWinkel_ == noppenWinkel_ && l_noppenForm_ == noppenForm_ && l_laenge1_ == laenge1_ && l_laenge2_ == laenge2_ && l_tissueTyp_ == tissueTyp_ && l_gummiHaerte_ == gummiHaerte_ && l_anpressDruck_ == anpressDruck_);
}

void
Embossing::getImageName(std::string &imageName)
{
    const char *sca_path = getenv("SCA_PATH");
    if (sca_path)
    {
        imageName = sca_path;
        imageName += "/TEXTURES/";
    }
    imageName += ReadASCIIDyna::tissueTypes[tissueTyp_ - 1];
    imageName += ".tif";
}

static int
pr_exit(int status)
{
    char buf[64];
    if (WIFEXITED(status) && WEXITSTATUS(status) != 0)
    {
        Covise::sendWarning("ANSYS abnormal termination, exit status %d", status);
        return -1;
    }
    else if (WIFSIGNALED(status))
    {
        Covise::sendWarning("ANSYS abnormal termination, signal number %d", WTERMSIG(status));
#ifdef WCOREDUMP
        if (WCOREDUMP(status))
        {
            Covise::sendWarning("core file generated");
        }
#endif
        return -1;
    }
    else if (WIFSTOPPED(status))
    {
        std::string ANSYS_Stopped("ANSYS process stopped, signal number ");
        sprintf(buf, "%d", WSTOPSIG(status));
        return -1; // well, this may be too radical....
    }
    return 0;
}

int
Embossing::ANSYSInputAndLaunch(coDoPolygons *erhebung, coDoPolygons *thickObj)
{
    // create xpoints.dat ypoints.dat files
    std::string out_xpoints(simDir_);
    out_xpoints += "xpoints.dat";
    ofstream XPoints(out_xpoints.c_str());
    int point;
    for (point = 0; point < xp_.size(); ++point)
    {
        XPoints << xp_[point] << endl;
    }
    XPoints.close();
    std::string out_ypoints(simDir_);
    out_ypoints += "ypoints.dat";
    ofstream YPoints(out_ypoints.c_str());
    for (point = 0; point < yp_.size(); ++point)
    {
        YPoints << yp_[point] << endl;
    }
    YPoints.close();

    std::string out_qparams(simDir_);
    out_qparams += "qparams.log";
    ofstream outParam(out_qparams.c_str());
    int numpoint = erhebung->getNumPoints();
    int numvert = erhebung->getNumVertices();
    int numelems = erhebung->getNumPolygons();

    if (numvert != 4 * numelems)
    {
        sendWarning("Not all elements are quads?: numvert != 4*numelems");
        return -1;
    }

    int *vl, *pl;
    float *xc, *yc, *zc, *thick;

    thickObj->getAddresses(&xc, &yc, &thick, &vl, &pl);
    erhebung->getAddresses(&xc, &yc, &zc, &vl, &pl);

    vector<float> zcv(zc, zc + numpoint);

    // the hard experience shows that zc may have to be corrected.
    // First get domain-line points and work out average zc, then
    // assign that average to those points.
    {
        vector<int> el(pl, pl + numelems);
        vector<int> num_conn(numelems, 4);
        vector<int> cl(vl, vl + numvert);
        vector<int> start_neigh, number_neigh, neighbours;
        MagmaUtils::NodeNeighbours(el, num_conn, cl, numpoint,
                                   start_neigh, number_neigh, neighbours);
        vector<int> elem_start_neigh, elem_number_neigh, elem_neighbours;
        vector<MagmaUtils::Edge> edge_neighbours;
        MagmaUtils::CellNeighbours(el, num_conn, cl, start_neigh, number_neigh, neighbours,
                                   elem_start_neigh, elem_number_neigh, elem_neighbours,
                                   edge_neighbours);
        vector<int> border_els;
        vector<MagmaUtils::Edge> border_edges;
        MagmaUtils::DomainLines(el, num_conn, cl, elem_start_neigh, elem_number_neigh,
                                edge_neighbours, border_els, border_edges);
        // start averaging...
        float zc_border_nodes = 0.0;
        int i = 0;
        for (i = 0; i < border_edges.size(); ++i)
        {
            zc_border_nodes += zc[border_edges[i].first];
            zc_border_nodes += zc[border_edges[i].second];
        }
        zc_border_nodes /= border_edges.size();
        zc_border_nodes *= 0.5;
        for (i = 0; i < border_edges.size(); ++i)
        {
            zcv[border_edges[i].first] = zc_border_nodes;
            zcv[border_edges[i].second] = zc_border_nodes;
        }
    }
    outParam << "width = " << width_ << endl;
    outParam << "height = " << height_ << endl;
    outParam << "numpoint = " << numpoint << endl;
    outParam << "numelems = " << numelems << endl;
    outParam << "numbumps = " << xp_.size() << endl;
    outParam << "PapProp = '";
    outParam << ReadASCIIDyna::tissueTypes[tissueTyp_ - 1] << "'" << endl;

    // coords
    std::string out_coords(simDir_);
    out_coords += "coords.dat";
    ofstream outCoords(out_coords.c_str());
    outCoords.precision(5);
    outCoords.setf(ios::scientific, ios::floatfield);
    int node;
    float loc_conv = 1.0;
    if (1)
    {
        loc_conv = ReadASCIIDyna::CONVERSION_FACTOR;
    }
    for (node = 0; node < numpoint; ++node)
    {
        outCoords << std::setw(13) << loc_conv * xc[node]
                  << std::setw(13) << loc_conv * yc[node]
                  << std::setw(13) << loc_conv * zcv[node]
                  << std::setw(13) << loc_conv * thick[node] << endl;
    }

    // conn
    std::string out_conn(simDir_);
    out_conn += "connec.dat";
    ofstream outConn(out_conn.c_str());
    int elem;
    for (elem = 0; elem < numelems; ++elem)
    {
        int base = 4 * elem;
        outConn << std::setw(10) << vl[base + 0] + 1 << std::setw(10) << vl[base + 1] + 1 << std::setw(10) << vl[base + 2] + 1 << std::setw(10) << vl[base + 3] + 1 << endl;
    }
    // now launch ANSYS
    return LaunchANSYS();
}

int
Embossing::LaunchANSYS()
{
    pid_t ANSYS_pid;
    sendInfo("Starting ANSYS for paper visualisation, please wait");
    if ((ANSYS_pid = fork()) < 0)
    {
        sendWarning("fork error");
        return -1;
    }
    else if (ANSYS_pid == 0) // child
    {
        // first duplicate "driver.log" onto the standard input...
        if (chdir(simDir_.c_str()) != 0)
        {
            sendWarning("Could not change directory for simulation");
            exit(-1);
        }
        int driverFD = open("driver.log", O_RDONLY);
        if (driverFD < 0)
        {
            sendWarning("Open driver.log failed");
            exit(-1);
        }
        int dup2Out = dup2(driverFD, STDIN_FILENO);
        if (dup2Out < 0)
        {
            sendWarning("could not duplicate file descriptor of driver.log");
            exit(-1);
        }
#ifdef _UMLENKE_OUTPUT_
        int ANSYSOutput = open("output.out", O_WRONLY);
        if (ANSYSOutput < 0)
        {
            sendWarning("Open output.out failed");
            exit(-1);
        }
        dup2Out = dup2(driverFD, STDOUT_FILENO);
        if (dup2Out < 0)
        {
            sendWarning("could not duplicate file descriptor of output.out");
            exit(-1);
        }
#endif
        // ... and launch ansys
        vector<string> arguments;
        ReadASCIIDyna::SCA_Calls("ANSYS", arguments);
        if (arguments.size() == 0)
        {
            sendWarning("Could not read ANSYS command line from covise.config");
            exit(-1);
        }
        const char **arglist = new const char *[arguments.size() + 1];
        int arg;
        for (arg = 0; arg < arguments.size(); ++arg)
        {
            arglist[arg] = arguments[arg].c_str();
            // sendWarning(arglist[arg]);
            cerr << arglist[arg] << endl;
        }
        arglist[arg] = NULL;
        //if(execlp("ansys60","ansys60","-p","ANSYSRF",NULL)<0){
        if (execvp(arglist[0], const_cast<char **>(arglist)) < 0)
        {
            sendWarning("ANSYS execution failed");
            delete[] arglist;
            exit(-1);
        }
    }
    int status;
    if (waitpid(ANSYS_pid, &status, 0) < 0) // parent
    {
        sendWarning("waitpid failed");
        return -1;
    }
    sendInfo("ANSYS computation finished!!!");
    // check status
    return pr_exit(status);
}

int
Embossing::createANSYSMesh()
{
    // create params file
    ofstream outParam("params.log");
    outParam << "bheight = " << noppenHoehe_ << endl;
    outParam << "cradius = " << ausrundungsRadius_ << endl;
    outParam << "iradius = " << 2 * (noppenHoehe_ + ausrundungsRadius_) << endl;
    outParam << "egroesse = " << noppenHoehe_ / 3.0 << endl;
    outParam << "width = " << width_ << endl;
    outParam << "height = " << height_ << endl;
    outParam << "numpoint = " << xp_.size() << endl;
    outParam.close();

    // create xpoints.dat ypoints.dat files
    std::string out_xpoints(simDir_);
    out_xpoints += "xpoints.dat";
    ofstream XPoints(out_xpoints.c_str());
    int point;
    for (point = 0; point < xp_.size(); ++point)
    {
        XPoints << xp_[point] << endl;
    }
    XPoints.close();
    std::string out_ypoints(simDir_);
    out_ypoints += "ypoints.dat";
    ofstream YPoints(out_ypoints.c_str());
    for (point = 0; point < yp_.size(); ++point)
    {
        YPoints << yp_[point] << endl;
    }
    YPoints.close();

    // start ANSYS process
    return LaunchANSYS();
}

int
Embossing::readAndOutputMesh()
{
    // outputDummies();
    // coordinates in zug_mesh.dat
    std::string zugMesh(simDir_);
    zugMesh += "zug_mesh.dat";
    ifstream zug_mesh(zugMesh.c_str());
    float no_points;
    zug_mesh >> no_points;
    int point = 0;
    float x, y, z;
    vector<float> xc;
    vector<float> yc;
    vector<float> zc;
    while (zug_mesh >> x >> y >> z)
    {
        xc.push_back(x);
        yc.push_back(y);
        zc.push_back(z);
        ++point;
    }
    zug_mesh.close();
    if (point != int(no_points))
    {
        sendWarning("The number of points declared in the mesh file is wrong");
        return -1;
    }
    // connectivity in zug_conn.dat
    std::string zugConn(simDir_);
    zugConn += "zug_conn.dat";
    ifstream zug_conn(zugConn.c_str());
    float no_polygons;
    zug_conn >> no_polygons;
    float vectices[4];
    vector<int> cl;
    vector<int> pl;
    while (zug_conn >> vectices[0] >> vectices[1] >> vectices[2] >> vectices[3])
    {
        int vertex;
        pl.push_back(cl.size());
        for (vertex = 0; vertex < 4; ++vertex)
        {
            if (vectices[vertex] > 0)
            {
                cl.push_back(int(vectices[vertex]) - 1);
            }
        }
    }
    if (pl.size() != int(no_polygons))
    {
        sendWarning("The number of polygons declared in the mesh file is wrong");
        return -1;
    }
    coDoPolygons *basicCell = NULL;
    {
        basicCell = new coDoPolygons(p_VernetzteGrundZelle_->getObjName(), xc.size(),
                                     cl.size(), pl.size());
        float *x, *y, *z;
        int *c, *p;
        basicCell->getAddresses(&x, &y, &z, &c, &p);
        copy(xc.begin(), xc.end(), x);
        copy(yc.begin(), yc.end(), y);
        copy(zc.begin(), zc.end(), z);
        copy(cl.begin(), cl.end(), c);
        copy(pl.begin(), pl.end(), p);
    }
    basicCell->addAttribute("vertexOrder", "2");
    std::string Transform;
    setTransformAttribute(Transform);
    basicCell->addAttribute("TRANSFORM", Transform.c_str());
    if (presentationMode_ == 2)
    {
        char buf[64];
        sprintf(buf, "barrelDiameter %f\n", barrelDiameter_);
        basicCell->addAttribute("POLY_TO_CYL", buf);
    }
    p_VernetzteGrundZelle_->setCurrentObject(basicCell);

    permitTraction_ = true;
    outputMssg();

    return 0;
}

void
Embossing::setTransformAttribute(std::string &Transform)
{
    char buf[64];
    Transform = "Transform: ";
    sprintf(buf, "%d\n", 7); // tile
    Transform += buf;

    Transform += "TilingPlane ";
    sprintf(buf, "%d\n", 1); // XY
    Transform += buf;

    Transform += "flipTile ";
    sprintf(buf, "%d\n", 1); // flip
    Transform += buf;

    Transform += "TilingMin ";
    sprintf(buf, "%d %d\n", 0, 0); // min. repl.
    Transform += buf;

    Transform += "TilingMax ";
    sprintf(buf, "%d %d\n", anzahlReplikationenX_, anzahlReplikationenY_);
    Transform += buf;
}

int
Embossing::checkReadFlags()
{
    if (!readNoppenHoehe_)
    {
        sendWarning("Could not read NoppenHoehe");
        return -1;
    }
    if (!readAusrundungsRadius_)
    {
        sendWarning("Could not read AusrundungsRadius");
        return -1;
    }
    if (!readAbnutzungsRadius_)
    {
        sendWarning("Could not read AbnutzungsRadius");
        return -1;
    }
    if (!readNoppenWinkel_)
    {
        sendWarning("Could not read NoppenWinkel");
        return -1;
    }
    if (!readNoppenForm_)
    {
        sendWarning("Could not read NoppenForm");
        return -1;
    }
    if (!readLaenge1_)
    {
        sendWarning("Could not read Laenge1_");
        return -1;
    }
    if (!readLaenge2_)
    {
        sendWarning("Could not read Laenge2_");
        return -1;
    }
    if (!readTissueTyp_)
    {
        sendWarning("Could not read TissueTyp");
        return -1;
    }
    if (!readGummiHaerte_)
    {
        sendWarning("Could not read GummiHaerte");
        return -1;
    }
    if (!readAnpressDruck_)
    {
        sendWarning("Could not read AnpressDruck");
        return -1;
    }
    if (!readAnzahlReplikationenX_)
    {
        sendWarning("Could not read AnzahlReplikationenX");
        return -1;
    }
    if (!readAnzahlReplikationenY_)
    {
        sendWarning("Could not read AnzahlReplikationenY");
        return -1;
    }
    return 0;
}

void
Embossing::outputDummies()
{
    coDoPolygons *dummy = new coDoPolygons(p_VernetzteGrundZelle_->getObjName(), 0, 0, 0);
    char buf[64];
    std::string Transform = "Transform: ";
    sprintf(buf, "%d\n", 7); // tile
    Transform += buf;
    dummy->addAttribute("TRANSFORM", Transform.c_str());
    p_VernetzteGrundZelle_->setCurrentObject(dummy);

    outputMssg();
    p_image_->setCurrentObject(new coDoText(p_image_->getObjName(), 0));
}

#include "ResultDataBase.h"
#include "ResultFloatParam.h"
#include "ResultEnumParam.h"
#include "ReadASCIIDyna.h"
#include <float.h>

int
Embossing::loadFileNames(std::string &conn, std::string &disp, std::string &disp_exp,
                         std::string &thick)
{
    const char *sca_path_c = getenv("SCA_PATH");
    if (!sca_path_c)
    {
        sendWarning("Please, set SCA_PATH environment variable");
        return -1;
    }
    string sca_path(sca_path_c);
    sca_path += "/DATABASE/";
    ResultDataBase dataB(sca_path.c_str());
    std::vector<ResultParam *> list; // prepare a list with knob parameter values:
    // noppenHoehe, ausrundungsRadius, abnutzungsRadius, noppenWinkel,
    // noppenForm, laenge1, laenge2, tissueTyp, gummiHaerte, anpressDruck
    ResultFloatParam v0("noppenHoehe", noppenHoehe_, 3);
    ResultFloatParam v1("ausrundungsRadius", ausrundungsRadius_, 3);
    ResultFloatParam v2("abnutzungsRadius", abnutzungsRadius_, 3);
    ResultFloatParam v3("noppenWinkel", noppenWinkel_, 3);
    // char *labs[] = { "Raute", "Ellipse" };
    ResultEnumParam v4("noppenForm", 2, ReadASCIIDyna::noppenFormChoices, noppenForm_ - 1);
    ResultFloatParam v5("laenge1", laenge1_, 3);
    ResultFloatParam v6("laenge2", laenge2_, 3);
    ResultEnumParam v7("tissueTyp", 6, ReadASCIIDyna::tissueTypes, tissueTyp_ - 1);
    ResultFloatParam v8("gummiHaerte", gummiHaerte_, 3);
    ResultFloatParam v9("anpressDruck", anpressDruck_, 3);
    list.push_back(&v0);
    list.push_back(&v1);
    list.push_back(&v2);
    list.push_back(&v3);
    list.push_back(&v4);
    list.push_back(&v5);
    list.push_back(&v6);
    list.push_back(&v7);
    list.push_back(&v8);
    list.push_back(&v9);

    float diff = 0.0;
    std::vector<Candidate *> FinalCandidates;
    const char *path = dataB.searchForResult(diff, list, FinalCandidates, 0.01);
    conn = path;
    conn += "topology.k";
    disp = path;
    disp += "defgeo";
    disp_exp = path;
    disp_exp += "defgeo_exp";
    thick = path;
    thick += "movie100.s30";
    int candidate;
    for (candidate = 0; candidate < FinalCandidates.size(); ++candidate)
    {
        delete FinalCandidates[candidate];
    }

    return 0;
}

/*
static int
AddDisplacements(vector<float>& exc,vector<float>& eyc,vector<float>& ezc,
vector<float>& pxc, vector<float>& pyc, vector<float>& pzc,
std::vector<int>& mark)
{
   int node,count=0;
   for(node=0;node<mark.size();++node)
   {
      if(mark[node] != 0)
      {
         ++count;
      }
   }
   if(count != exc.size())
   {
      Covise::sendWarning("Displacement arrays and mark array have an incompatible state");
      return -1;
   }
   for(node=0,count=0;node<mark.size();++node)
   {
      if(mark[node] != 0)
      {
         exc[count] += pxc[node];
         eyc[count] += pyc[node];
         ezc[count] += pzc[node];
         ++count;
      }
   }
   float z_level=0.0;
   float xy_rand=-FLT_MAX;
   int punkt;
   for(punkt=0;punkt<ezc.size();++punkt)
   {
      float x = exc[punkt];
      float y = eyc[punkt];
      float z = ezc[punkt];
      if(x+y > xy_rand)
      {
         xy_rand = x+y;
         z_level = z;
      }
   }
   // shift z
   for(punkt=0;punkt<ezc.size();++punkt)
   {
      ezc[punkt] -= z_level;
   }
   return 0;
}


int
Embossing::readEmbossingResults(ifstream& emb_conn,ifstream& emb_displ,
ifstream& emb_displ_exp,
ifstream& emb_thick,
vector<int>& epl,vector<int>& cpl,
vector<float>& exc,
vector<float>& eyc,
vector<float>& ezc,vector<float>& dicke)
{
   vector<float> pxc;
   vector<float> pyc;
   vector<float> pzc;

   // std::vector<float> pdicke;
   float one_thickness;
   while(emb_thick >> one_thickness)
   {
      if(one_thickness == 0.0)
      {
         continue;
      }
      dicke.push_back(one_thickness);
   }

   char buffer[1024];
   bool found=false;
   while(emb_conn >> buffer)
   {
      if(strcmp(buffer,"*NODE")==0)
      {
         found=true;
         break;
      }
   }
   if(!found)
   {
      sendWarning("Could not found node coordinate section");
      return -1;
   }
   char retChar;
   emb_conn.get(retChar);
   if(retChar != '\r')
   {
      if(retChar != '\n')
      {
         emb_conn.putback(retChar);
      }
   }
   else
   {
      emb_conn.get(retChar);                      // read new line
   }
   while(emb_conn.getline(buffer,1024))
   {
      int node;
      if(sscanf(buffer,"%d",&node)!=1)
      {
         // we are done with the node definition
         break;
      }
      char numbers[64];
      strncpy(numbers,buffer+8,16);
      numbers[16] = ' ';
      strncpy(numbers+16+1,buffer+8+16,16);
      numbers[16+1+16] = ' ';
      strncpy(numbers+16+1+16+1,buffer+8+16+16,16);
      numbers[16+1+16+1+16] = '\0';
      istringstream floats(numbers);
      float x,y,z;
      floats.setf(floats.flags() | ios::uppercase);
      if(!(floats>>x>>y>>z))
      {
         sendWarning("Could not read coordinates for a node");
         return -1;
      }
      pxc.push_back(x);
      pyc.push_back(y);
      pzc.push_back(z);
   }

   // now follows the connectivity
   if(strncmp(buffer,"*ELEMENT_SHELL",14)!=0)
   {
      found=false;
      while(emb_conn >> buffer)
      {
         if(strcmp(buffer,"*ELEMENT_SHELL")==0)
         {
            found=true;
            break;
         }
      }
      if(!found)
      {
         sendWarning("Could not find element shell section");
         return -1;
      }
   }
   emb_conn.get(retChar);
   if(retChar != '\r')
   {
      if(retChar != '\n')
      {
         emb_conn.putback(retChar);
      }
   }
   else
   {
      emb_conn.get(retChar);                      // read new line
   }
   while(emb_conn.getline(buffer,1024))
   {
      int element,material;
      int n0,n1,n2,n3;
      if(sscanf(buffer,"%d %d %d %d %d %d",&element,&material,
         &n0,&n1,&n2,&n3)!=6)
      {
         // we are done with the shell section
         break;
      }
      if(material != 2)
      {
         continue;                                // this is not paper
      }
      epl.push_back(cpl.size());
      cpl.push_back(n0-1);
      cpl.push_back(n1-1);
      cpl.push_back(n2-1);
      cpl.push_back(n3-1);
   }
   // eliminate unused nodes
std::vector<int> mark;
   mark.reserve(pxc.size(),1);
   int paperNodes=MarkCoords(cpl,mark);
   if(paperNodes != dicke.size())
   {
      sendWarning("Mesh data is not compatible with thickness file");
      return -1;
   }

   // read displacements
   if(readDisplacements(emb_displ,
      mark,exc,eyc,ezc,true,dicke.size())!=0)
   {
      return -1;
   }
   if(readDisplacements(emb_displ_exp,mark,exc,eyc,ezc,false,-1)!=0)
   {
      return -1;
   }
   // add displacements and positions
   if(AddDisplacements(exc,eyc,ezc,pxc,pyc,pzc,mark) !=0)
   {
      return -1;
   }

   return 0;
}


inline float
CrazyFormat(char number[8])
{
   char normal[16];
   int pos;
   normal[0] = number[0];
   for(pos=1;pos<6;++pos)
   {
      normal[pos] = (number[pos] == ' ') ? '0':number[pos];
   }
   if(number[6] == '-' || number[6] == '+' || number[6] == ' ')
   {
      normal[6] = 'e';
      normal[7] = number[6];
      if(normal[7] == ' ')
      {
         normal[7] = ' ';
      }
      normal[8] = number[7];
      if(normal[8] == ' ')
      {
         normal[6] = '\0';
      }
   }
   else
   {
      normal[6] = '\0';
   }
   float ret;
   sscanf(normal,"%f",&ret);
   return ret;
}


int
Embossing::readDisplacements(ifstream& emb_displ,
std::vector<int>& mark,
vector<float>& exc,
vector<float>& eyc,
vector<float>& ezc,
bool newFormat,
int dickeSize)
{
   // jump over first reasults
   char buf[1024];
   int count=0;
   while(emb_displ.getline(buf,1024))
   {
      ++count;
   }
   if(   !newFormat && count%(1+mark.size()) != 0)
   {
      cerr << "Explicit " << count << ' '<< mark.size() << endl;
   }
   if(newFormat && count%(1+dickeSize) != 0)
   {
      cerr << "Implicit " << count << ' '<< 1+dickeSize << endl;
   }
   if(   (!newFormat && count%(1+mark.size()) != 0)
      || (newFormat && count%(1+dickeSize) != 0))
   {
      sendWarning("The displacements file has a wrong number of lines");
      return -1;
   }
   emb_displ.clear();
   emb_displ.seekg(0,ios::beg);                   // rewind
   int recount;
   int lastLines;
   if(newFormat)
   {
      lastLines = dickeSize;
   }
   else
   {
      lastLines = mark.size();
   }
   for(recount=0;recount<count - lastLines;++recount)
   {
      emb_displ.getline(buf,1024);
   }
   // now read displacements of marked nodes
   int node;
   for(node = 0,count=0;node < lastLines;++node)
   {
      emb_displ.getline(buf,1024);
      if(!newFormat && mark[node]==0)
      {
         continue;
      }
      // check node number
      char number[16];
      strncpy(number,buf,8);
      number[8]='\0';
      int checkNode;
      sscanf(number,"%d",&checkNode);
      if(!newFormat)
      {
         if(checkNode != node+1)
         {
            sendWarning("Node numbers in displacements file are not correct");
            return -1;
         }
      }
      // read X position
      strncpy(number,buf+8,8);
      number[8]='\0';
      if(newFormat)
      {
         exc.push_back(CrazyFormat(number));
      }
      else
      {
         exc[count] += CrazyFormat(number);
      }
      // read Y position
      strncpy(number,buf+8+8,8);
      number[8]='\0';
      if(newFormat)
      {
         eyc.push_back(CrazyFormat(number));
      }
      else
      {
         eyc[count] += CrazyFormat(number);
      }
      // read Z position
      strncpy(number,buf+8+8+8,8);
      number[8]='\0';
      if(newFormat)
      {
         ezc.push_back(CrazyFormat(number));
      }
      else
      {
         ezc[count] += CrazyFormat(number);
      }
      ++count;
   }
   return 0;
}


// mark has already the correct size
int
Embossing::MarkCoords(vector<int>& cpl,std::vector<int>& mark)
{
   int numini = mark.size();
   memset(&mark[0],0,numini*sizeof(int));
   int vert;
   for(vert=0;vert<cpl.size();++vert)
   {
      if(cpl[vert] >= numini)
      {
         sendWarning("MarkCoords: connectivity refers non-existing nodes");
         return -1;                               // error
      }
      if(cpl[vert] < 0)
      {
         sendWarning("MarkCoords: connectivity refers node 0 or negative");
         return -1;                               // error
      }
      ++mark[cpl[vert]];
   }
   std::vector<int> compressedPoints;
   int point,count=0;
   for(point=0;point<numini;++point)
   {
      if(mark[point] != 0)
      {
         compressedPoints.push_back(count);
         ++count;
      }
      else
      {
         compressedPoints.push_back(-1);
      }
   }
   for(vert=0;vert<cpl.size();++vert)
   {
      cpl[vert] = compressedPoints[cpl[vert]];
      if(cpl[vert]<0)
      {
         sendWarning("MarkCoords: This is a bug");
         return -1;
      }
   }
   return count;
}

*/
void
Embossing::Mirror2(vector<float> &exc, vector<float> &eyc, vector<float> &ezc,
                   vector<int> &ecl, vector<int> &epl, Direction d)
{
    int numpoints = exc.size();
    int point;
    for (point = 0; point < numpoints; ++point)
    {
        float tmpx = exc[point];
        float tmpy = eyc[point];
        float tmpz = ezc[point];
        exc.push_back(tmpx);
        eyc.push_back(tmpy);
        ezc.push_back(tmpz);
    }
    if (d == X)
    {
        for (point = 0; point < numpoints; ++point)
        {
            exc[numpoints + point] *= -1.0;
        }
    }
    else
    {
        for (point = 0; point < numpoints; ++point)
        {
            eyc[numpoints + point] *= -1.0;
        }
    }
    int vertex;
    int numvertices = ecl.size();
    // eeeeeps, there is a problem with the orientation!!!!
    for (vertex = 0; vertex < numvertices; ++vertex)
    {
        int base = vertex % 2;
        int shift = 1 - 2 * base;
        ecl.push_back(ecl[vertex + shift] + numpoints);
    }
    int count, element;
    int numelems = epl.size();
    for (element = 0, count = numvertices; element < numelems; ++element, count += 4)
    {
        epl.push_back(count);
    }
}

MODULE_MAIN(SCA, Embossing)
