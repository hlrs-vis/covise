/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Design.h"
#include <string>
#include <alg/DeleteUnusedPoints.h>
#include <do/coDoText.h>
#include <do/coDoSet.h>

Design::Design(int argc, char *argv[])
    : coModule(argc, argv, "SCA Design")
    , xknob_(0.0)
    , yknob_(0.0)
{
    p_tolerance_ = addFloatParam("tolerance", "relative tolerance");
    p_tolerance_->setValue(1.0e-6);
    p_designParam_ = addInputPort("DesignParam", "coDoText", "Design parameters");
    p_Knob_ = addInputPort("Knob", "coDoPolygons", "The knob");
    p_grundZelle_ = addOutputPort("Grundzelle", "coDoPolygons", "Basic cell geometry");
    p_noppenPositionen_ = addOutputPort("NoppenPositionen", "coDoPoints", "Knob positions");
    p_noppenColors_ = addOutputPort("NoppenColors", "coDoFloat", "Noppen colors for design rules");
    p_show_grundZelle_ = addOutputPort("ShowGrundzelle", "coDoPolygons", "Show basic cell geometry");
    p_show_noppenPositionen_ = addOutputPort("ShowKnobs", "coDoPolygons", "Show knob positions");
    p_show_noppenColors_ = addOutputPort("ShowNoppenColors", "coDoFloat", "Show noppen colors for design rules");
    p_show_fuesse_ = addOutputPort("ShowKnobProfiles", "coDoLines", "Show knob profiles");
    p_show_phys_fuesse_ = addOutputPort("ShowPhysKnobProfiles", "coDoLines", "Show physical knob profiles");
    p_cutX_ = addOutputPort("cutX", "coDoText", "CutGeomerty X params");
    p_cutY_ = addOutputPort("cutY", "coDoText", "CutGeomerty Y params");
}

Design::~Design()
{
}

int
Design::readIntSlider(istringstream &strText, int *addr)
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
Design::readFloatSlider(istringstream &strText, float *addr)
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
Design::readIntScalar(istringstream &strText, int *addr)
{
    if (!(strText >> *addr))
    {
        Covise::sendWarning("Error on reading IntScalar from istringstream");
        return -1;
    }
    return 0;
}

int
Design::readChoice(istringstream &strText, int *addr)
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

int
Design::readFloatVector(istringstream &strText, int len, float *addr)
{
    std::vector<char> Value(maxLen_);
    strText >> &Value[0];
    int comp;
    for (comp = 0; comp < len; ++comp)
    {
        if (!(strText >> *addr))
        {
            Covise::sendWarning("Error on reading FloatVector from istringstream");
            return -1;
        }
        ++addr;
    }
    return 0;
}

// not used!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
void
Design::Periodicity(vector<float> &xc, vector<float> &yc, vector<float> &zc,
                    float width, float height, float deltax, float deltay)
{
    // loop over points and check translated points...
    int point;
    int no_points = xc.size();
    for (point = 0; point < no_points; ++point)
    {
        int i, j;
        for (i = -1; i <= 1; ++i)
        {
            float newX = xc[point] + i * deltax;
            bool Xborder = true;
            if (!(fabs(newX) < p_tolerance_->getValue() * width
                  || fabs(newX - width) < p_tolerance_->getValue() * width))
            {
                Xborder = false;
            }
            for (j = -1; j <= 1; ++j)
            {
                bool Yborder = true;
                if (i == 0 && j == 0)
                {
                    continue;
                }
                float newY = yc[point] + j * deltay;
                if (!(fabs(newY) < p_tolerance_->getValue() * height
                      || fabs(newY - height) < p_tolerance_->getValue() * height))
                {
                    Yborder = false;
                }
                if (Xborder || Yborder)
                {
                    xc.push_back(newX);
                    yc.push_back(newY);
                    zc.push_back(0.0);
                }
            }
        }
    }
}

int
Design::compute(const char *port)
{
    (void)port; // silence compiler

    readGrundZellenHoehe_ = readGrundZellenBreite_ = readAnzahlLinien_ = false;
    readAnzahlPunkteProLinie_ = readVersatz_ = false;
    readDAnzahlReplikationenX_ = readDAnzahlReplikationenY_ = false;
    readNoppenHoehe_ = readAusrundungsRadius_ = readAbnutzungsRadius_ = false;
    readNoppenWinkel_ = readNoppenForm_ = readLaenge1_ = readLaenge2_ = false;
    readTissueTyp_ = readGummiHaerte_ = readAnpressDruck_ = false;
    readFree_or_param_ = false;
    readNum_points_ = false;
    int noppen_count;
    for (noppen_count = 0; noppen_count < MAX_POINTS; ++noppen_count)
    {
        readFreie_noppen_[noppen_count] = false;
    }
    // get the text object
    const coDistributedObject *inObj = p_designParam_->getCurrentObject();
    if (inObj == NULL || !inObj->objectOk())
    {
        sendWarning("Got NULL pointer or object is not OK");
        return FAIL;
    }
    if (!inObj->isType("DOTEXT"))
    {
        sendWarning("Only coDoText is acceptable for input");
        return FAIL;
    }
    const coDoText *theText = dynamic_cast<const coDoText *>(inObj);
    int size = theText->getTextLength();
    if (size == 0)
    {
        outputDummies();
        return SUCCESS;
    }
    char *text;
    theText->getAddress(&text);

    istringstream strText;
    strText.str(string(text));
    maxLen_ = strlen(text) + 1;
    std::vector<char> name(maxLen_);

    while (strText >> &name[0])
    {
        if (strcmp(&name[0], "grundZellenHoehe") == 0)
        {
            if (readFloatSlider(strText, &grundZellenHoehe_) != 0)
            {
                sendWarning("Could not read grundZellenHoehe");
                return FAIL;
            }
            readGrundZellenHoehe_ = true;
        }
        else if (strcmp(&name[0], "grundZellenBreite") == 0)
        {
            if (readFloatSlider(strText, &grundZellenBreite_) != 0)
            {
                sendWarning("Could not read grundZellenBreite");
                return FAIL;
            }
            readGrundZellenBreite_ = true;
        }
        else if (strcmp(&name[0], "anzahlLinien") == 0)
        {
            if (readIntSlider(strText, &anzahlLinien_) != 0)
            {
                sendWarning("Could not read anzahlLinien");
                return FAIL;
            }
            readAnzahlLinien_ = true;
        }
        else if (strcmp(&name[0], "anzahlPunkteProLinie") == 0)
        {
            if (readIntSlider(strText, &anzahlPunkteProLinie_) != 0)
            {
                sendWarning("Could not read anzahlPunkteProLinie");
                return FAIL;
            }
            readAnzahlPunkteProLinie_ = true;
        }
        else if (strcmp(&name[0], "versatz") == 0)
        {
            int i_versatz;
            if (readIntSlider(strText, &i_versatz) != 0)
            {
                sendWarning("Could not read versatz");
                return FAIL;
            }
            versatz_ = 100.0 / (i_versatz + 1.0);
            readVersatz_ = true;
        }
        else if (strcmp(&name[0], "DanzahlReplikationenX") == 0)
        {
            if (readIntSlider(strText, &DanzahlReplikationenX_) != 0)
            {
                sendWarning("Could not read DanzahlReplikationenX");
                return FAIL;
            }
            readDAnzahlReplikationenX_ = true;
        }
        else if (strcmp(&name[0], "DanzahlReplikationenY") == 0)
        {
            if (readIntSlider(strText, &DanzahlReplikationenY_) != 0)
            {
                sendWarning("Could not read DanzahlReplikationenY");
                return FAIL;
            }
            readDAnzahlReplikationenY_ = true;
        }
        else if (strcmp(&name[0], "noppenHoehe") == 0)
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
        // free or parametric
        else if (strcmp(&name[0], "free_or_param") == 0)
        {
            if (readChoice(strText, &free_or_param_) != 0)
            {
                sendWarning("Could not read free_or_param");
                return FAIL;
            }
            readFree_or_param_ = true;
        }
        // free case
        else if (strncmp(&name[0], "Noppen_", strlen("Noppen_")) == 0)
        {
            // get the number!!!
            const char *strnumber = &name[0] + strlen("Noppen_");
            int number = atoi(strnumber) - 1;
            if (readFloatVector(strText, 2, freie_noppen_[number]) != 0)
            {
                sendWarning("Could not read freie_noppen_");
                return FAIL;
            }
            readFreie_noppen_[number] = true;
        }
        else if (strcmp(&name[0], "NumPoints") == 0)
        {
            if (readIntScalar(strText, &num_points_) != 0)
            {
                sendWarning("Could not read num_points");
                return FAIL;
            }
            readNum_points_ = true;
        }
    }
    if (checkReadFlags() != 0)
    {
        return FAIL;
    }
    // params for cutX and cutY
    char buf[128];
    sprintf(buf, "distance %g\nnormal ", -grundZellenBreite_);
    strcat(buf, "-1.0 0.0 0.0");
    coDoText *doTextX = new coDoText(p_cutX_->getObjName(), strlen(buf) + 1);
    char *addr;
    doTextX->getAddress(&addr);
    strcpy(addr, buf);
    p_cutX_->setCurrentObject(doTextX);

    sprintf(buf, "distance %g \nnormal 0.0 -1.0 0.0", -grundZellenHoehe_);
    coDoText *doTextY = new coDoText(p_cutY_->getObjName(), strlen(buf) + 1);
    doTextY->getAddress(&addr);
    strcpy(addr, buf);
    p_cutY_->setCurrentObject(doTextY);

    // make polygon for the sheet
    coDoPolygons *polys = NULL;
    {
        float x_c[4], y_c[4], z_c[4];
        x_c[0] = 0.0;
        x_c[1] = grundZellenBreite_;
        x_c[2] = grundZellenBreite_;
        x_c[3] = 0.0;
        y_c[0] = 0.0;
        y_c[1] = 0.0;
        y_c[2] = grundZellenHoehe_;
        y_c[3] = grundZellenHoehe_;
        z_c[0] = 0.0;
        z_c[1] = 0.0;
        z_c[2] = 0.0;
        z_c[3] = 0.0;
        int vl[4];
        vl[0] = 0;
        vl[1] = 1;
        vl[2] = 2;
        vl[3] = 3;
        int zero = 0;
        polys = new coDoPolygons(p_grundZelle_->getObjName(),
                                 4, x_c, y_c, z_c, 4, vl, 1, &zero);
    }
    polys->addAttribute("vertexOrder", "2");
    polys->addAttribute("COLOR", "White");
    p_grundZelle_->setCurrentObject(polys);

    // set TRANSFORM attribute
    std::string transfAttr;
    setTransformAttribute(transfAttr);

    // reuse sheet polygon in case of SHOW_SCA_DESIGN
    if (inObj->getAttribute("SHOW_SCA_DESIGN"))
    {
        polys->incRefCount();
        coDistributedObject *setList[2];
        setList[0] = polys;
        setList[1] = NULL;
        coDoSet *set_polys = new coDoSet(p_show_grundZelle_->getObjName(), setList);
        set_polys->addAttribute("TRANSFORM", transfAttr.c_str());
        p_show_grundZelle_->setCurrentObject(set_polys);
    }
    else
    {
        coDistributedObject *dummy = new coDoPolygons(p_show_grundZelle_->getObjName(), 0, 0, 0);
        dummy->addAttribute("TRANSFORM", transfAttr.c_str());
        p_show_grundZelle_->setCurrentObject(dummy);
    }
    // now the points
    vector<float> x_c;
    vector<float> y_c;
    vector<float> z_c;
    if (free_or_param_ > 2)
    {
        if (anzahlLinien_ == 0)
        {
            x_c.push_back(0.0);
            y_c.push_back(0.0);
            z_c.push_back(0.0);
        }
        else
        {
            int line;
            int point;
            float delta_y = float(grundZellenHoehe_) / (anzahlLinien_ * anzahlPunkteProLinie_);
            float delta_x = float(grundZellenBreite_) / anzahlPunkteProLinie_;
            float tolerance = p_tolerance_->getValue();
            for (line = 0; line < anzahlLinien_; ++line)
            {
                float shiftx = line * versatz_ * delta_x / 100.0;
                shiftx = fmod(shiftx, delta_x);
                float shifty = line * versatz_ * delta_y / 100.0;
                shifty = fmod(shifty, delta_y);
                // if shiftx is almost delta_x, make it equal 0.0
                if (delta_x - shiftx < tolerance * delta_x
                    || delta_y - shifty < tolerance * delta_y)
                {
                    shiftx = 0.0;
                    shifty = 0.0;
                }
                float base_y = shifty + (float(grundZellenHoehe_) / anzahlLinien_) * line;
                for (point = 0; point < anzahlPunkteProLinie_ + 1; ++point)
                {
                    float xcoord = delta_x * point + shiftx;
                    if (fabs(xcoord - grundZellenBreite_) < tolerance * grundZellenBreite_)
                    {
                        xcoord = float(grundZellenBreite_);
                    }
                    float ycoord = base_y + delta_y * point;
                    if (fabs(ycoord - grundZellenHoehe_) < tolerance * grundZellenHoehe_)
                    {
                        ycoord = float(grundZellenHoehe_);
                    }
                    if (xcoord <= float(grundZellenBreite_) && ycoord <= float(grundZellenHoehe_))
                    {
                        x_c.push_back(xcoord);
                        y_c.push_back(ycoord);
                        z_c.push_back(0.0);
                    }
                }
            }
        }
    }
    else if (free_or_param_ == 2) // param
    {
        float tolerance = p_tolerance_->getValue();
        num_points_ = 0;
        for (noppen_count = 0; noppen_count < MAX_POINTS; ++noppen_count)
        {
            if (freie_noppen_[noppen_count][0] < 0.0)
            {
                continue;
            }
            if (freie_noppen_[noppen_count][0] > grundZellenBreite_)
            {
                continue;
            }
            if (freie_noppen_[noppen_count][1] < 0.0)
            {
                continue;
            }
            if (freie_noppen_[noppen_count][1] > grundZellenHoehe_)
            {
                continue;
            }
            x_c.push_back(freie_noppen_[noppen_count][0]);
            y_c.push_back(freie_noppen_[noppen_count][1]);
            z_c.push_back(0.0);
            if (x_c[num_points_] < tolerance)
            {
                x_c[num_points_] = 0.0;
            }
            if (grundZellenBreite_ - x_c[num_points_] < tolerance)
            {
                x_c[num_points_] = grundZellenBreite_;
            }
            if (y_c[num_points_] < tolerance)
            {
                y_c[num_points_] = 0.0;
            }
            if (grundZellenHoehe_ - y_c[num_points_] < tolerance)
            {
                y_c[num_points_] = grundZellenHoehe_;
            }
            ++num_points_;
        }
    }
    else if (!inObj->getAttribute("KNOB_SELECT"))
    {
        sendWarning("Choose between free or parametric design");
        return FAIL;
    }

    // check design rules: input x_c, y_c arrays and
    //                     basic cell dimensions
    //                            grundZellenBreite_, grundZellenHoehe_
    vector<float> colors;
    for (noppen_count = 0; noppen_count < x_c.size(); ++noppen_count)
    {
        colors.push_back(0.0);
    }

    // get knob dimensions (xknob, yknob)
    float xknob, yknob;
    xknob = laenge1_ * 0.5 + 1.0 + noppenHoehe_ * tan(noppenWinkel_ * M_PI / 180.0);
    yknob = laenge2_ * 0.5 + 1.0 + noppenHoehe_ * tan(noppenWinkel_ * M_PI / 180.0);

    // check LS-Dyna feet
    if (DesignRules(x_c, y_c, colors, inObj->getAttribute("KNOB_SELECT") != NULL,
                    xknob, yknob))
    {
        sendWarning("Could not check design rules");
        return FAIL;
    }

    coDoPoints *points = new coDoPoints(p_noppenPositionen_->getObjName(),
                                        x_c.size());
    {
        float *x, *y, *z;
        points->getAddresses(&x, &y, &z);
        copy(x_c.begin(), x_c.end(), x);
        copy(y_c.begin(), y_c.end(), y);
        copy(z_c.begin(), z_c.end(), z);
    }
    p_noppenPositionen_->setCurrentObject(points);
    coDoFloat *data_colors = new coDoFloat(
        p_noppenColors_->getObjName(),
        colors.size());
    {
        float *col;
        data_colors->getAddress(&col);
        copy(colors.begin(), colors.end(), col);
    }
    p_noppenColors_->setCurrentObject(data_colors);

    if (inObj->getAttribute("SHOW_SCA_DESIGN"))
    {
        coDistributedObject *setList[2];
        // we do not show a symbolic design any more... but the real tool!!!
        // points->incRefCount();
        // setList[0] = points;
        MakeTheKnobs(x_c, y_c, z_c);

        data_colors->incRefCount();
        setList[0] = data_colors;
        setList[1] = NULL;
        coDoSet *setColors = new coDoSet(p_show_noppenColors_->getObjName(),
                                         setList);
        // p_show_noppenPositionen_->setCurrentObject(set_points);
        p_show_noppenColors_->setCurrentObject(setColors);
        ShowFuesse(x_c, y_c);
        ShowPhysFuesse(x_c, y_c);
    }
    else
    {
        setTransformAttribute(transfAttr);
        coDoPolygons *show_noppenPositionen = new coDoPolygons(p_show_noppenPositionen_->getObjName(), 0, 0, 0);
        show_noppenPositionen->addAttribute("TRANSFORM", transfAttr.c_str());
        p_show_noppenPositionen_->setCurrentObject(show_noppenPositionen);

        p_show_noppenColors_->setCurrentObject(new coDoFloat(
            p_show_noppenColors_->getObjName(), 0));
        p_show_fuesse_->setCurrentObject(new coDoLines(p_show_fuesse_->getObjName(),
                                                       0, 0, 0));
        p_show_phys_fuesse_->setCurrentObject(new coDoLines(p_show_phys_fuesse_->getObjName(),
                                                            0, 0, 0));
    }
    return SUCCESS;
}

void
AddRectangle(std::vector<int> &pl, std::vector<int> &vl,
             std::vector<float> &xfuss, std::vector<float> &yfuss, std::vector<float> &zfuss,
             float xc, float yc, float xknob, float yknob, float levitation)
{
    pl.push_back(vl.size());
    // first point
    vl.push_back(xfuss.size());
    xfuss.push_back(xc + xknob);
    yfuss.push_back(yc + yknob);
    zfuss.push_back(levitation);
    // second point
    vl.push_back(xfuss.size());
    xfuss.push_back(xc - xknob);
    yfuss.push_back(yc + yknob);
    zfuss.push_back(levitation);
    // third point
    vl.push_back(xfuss.size());
    xfuss.push_back(xc - xknob);
    yfuss.push_back(yc - yknob);
    zfuss.push_back(levitation);
    // fourth point
    vl.push_back(xfuss.size());
    xfuss.push_back(xc + xknob);
    yfuss.push_back(yc - yknob);
    zfuss.push_back(levitation);
    // first point again
    vl.push_back(xfuss.size());
    xfuss.push_back(xc + xknob);
    yfuss.push_back(yc + yknob);
    zfuss.push_back(levitation);
}

void
AddEllipse(std::vector<int> &pl, std::vector<int> &vl,
           std::vector<float> &xfuss, std::vector<float> &yfuss, std::vector<float> &zfuss,
           float xc, float yc, float xknob, float yknob, float levitation)
{
    const int quarterDiv = 8;
    pl.push_back(vl.size());
    int angle;
    for (angle = 0; angle < 4 * quarterDiv; ++angle)
    {
        float rad = 0.5 * M_PI * angle / float(quarterDiv);
        vl.push_back(xfuss.size());
        xfuss.push_back(xc + xknob * cos(rad));
        yfuss.push_back(yc + yknob * sin(rad));
        zfuss.push_back(levitation);
    }
    vl.push_back(xfuss.size());
    xfuss.push_back(xc + xknob);
    yfuss.push_back(yc);
    zfuss.push_back(levitation);
}

void
Design::ShowFuesse(vector<float> &x_c, vector<float> &y_c)
{
    std::vector<int> pl;
    std::vector<int> vl;
    std::vector<float> xfuss;
    std::vector<float> yfuss;
    std::vector<float> zfuss;
    int point;
    // fill arrays
    // decide symbol levitation
    float levitation = 0.05 * noppenHoehe_;

    if (noppenForm_ == 1)
    {
        for (point = 0; point < x_c.size(); ++point)
        {
            AddRectangle(pl, vl, xfuss, yfuss, zfuss, x_c[point], y_c[point], xknob_, yknob_, levitation);
        }
    }
    else
    {
        for (point = 0; point < x_c.size(); ++point)
        {
            AddEllipse(pl, vl, xfuss, yfuss, zfuss, x_c[point], y_c[point], xknob_, yknob_, levitation);
        }
    }
    // create covise object
    std::string setElemName = p_show_fuesse_->getObjName();
    setElemName += "_0";
    coDistributedObject *setList[2];
    setList[1] = NULL;
    coDoLines *polys = new coDoLines(setElemName.c_str(),
                                     xfuss.size(), &xfuss[0],
                                     &yfuss[0], &zfuss[0],
                                     vl.size(), &vl[0],
                                     pl.size(), &pl[0]);
    setList[0] = polys;
    coDoSet *setFuesse = new coDoSet(p_show_fuesse_->getObjName(), setList);
    // polys->addAttribute("vertexOrder","2");
    polys->addAttribute("NO_BBOX", "");
    polys->addAttribute("COLOR", "blue");
    p_show_fuesse_->setCurrentObject(setFuesse);
}

void
Design::ShowPhysFuesse(vector<float> &x_c, vector<float> &y_c)
{
    std::vector<int> pl;
    std::vector<int> vl;
    std::vector<float> xfuss;
    std::vector<float> yfuss;
    std::vector<float> zfuss;
    int point;
    // fill arrays
    // decide symbol levitation
    float levitation = 0.05 * noppenHoehe_;

    float xknob, yknob;
    xknob = laenge1_ * 0.5 + 1.0 + noppenHoehe_ * tan(noppenWinkel_ * M_PI / 180.0);
    yknob = laenge2_ * 0.5 + 1.0 + noppenHoehe_ * tan(noppenWinkel_ * M_PI / 180.0);
    if (noppenForm_ == 1)
    {
        for (point = 0; point < x_c.size(); ++point)
        {
            AddRectangle(pl, vl, xfuss, yfuss, zfuss, x_c[point], y_c[point], xknob, yknob, levitation);
        }
    }
    else
    {
        for (point = 0; point < x_c.size(); ++point)
        {
            AddEllipse(pl, vl, xfuss, yfuss, zfuss, x_c[point], y_c[point], xknob, yknob, levitation);
        }
    }
    // create covise object
    std::string setElemName = p_show_phys_fuesse_->getObjName();
    setElemName += "_0";
    coDistributedObject *setList[2];
    setList[1] = NULL;
    coDoLines *polys = new coDoLines(setElemName.c_str(),
                                     xfuss.size(), &xfuss[0],
                                     &yfuss[0], &zfuss[0],
                                     vl.size(), &vl[0],
                                     pl.size(), &pl[0]);
    setList[0] = polys;
    coDoSet *setFuesse = new coDoSet(p_show_phys_fuesse_->getObjName(), setList);
    // polys->addAttribute("vertexOrder","2");
    polys->addAttribute("NO_BBOX", "");
    polys->addAttribute("COLOR", "cyan");
    p_show_phys_fuesse_->setCurrentObject(setFuesse);
}

int
Design::getKnobSize(float *xsize, float *ysize)
{
    const coDistributedObject *in_knob = p_Knob_->getCurrentObject();
    if (!in_knob || !in_knob->isType("POLYGN"))
    {
        sendWarning("got a NULL pointer to the knob or a wrong type");
        return -1;
    }
    coDoPolygons *inKnob = (coDoPolygons *)in_knob;
    int no_points = inKnob->getNumPoints();
    *xsize = 0;
    *ysize = 0;
    if (no_points <= 0)
    {
        return 0;
    }
    float *x_c;
    float *y_c;
    float *z_c;
    int *v_l;
    int *l_l;
    inKnob->getAddresses(&x_c, &y_c, &z_c, &v_l, &l_l);
    int point;
    for (point = 0; point < no_points; ++point)
    {
        if (x_c[point] > *xsize)
        {
            *xsize = x_c[point];
        }
        if (y_c[point] > *ysize)
        {
            *ysize = y_c[point];
        }
    }
    return 0;
}

#include "ResultDataBase.h"
#include "ResultFloatParam.h"
#include "ResultEnumParam.h"
#include "ReadASCIIDyna.h"
#include <float.h>

int
Design::DesignRules(vector<float> &x_c, vector<float> &y_c, vector<float> &colors,
                    bool knob_select, float xknob, float yknob)
{
    (void)knob_select;
    /*   
      if(noppenForm_==3){// try first elliptic here
         int point1,point2;
         for(point1=0;point1<x_c.size();++point1){
            Ellipse elipse1(xknob,yknob,x_c[point1],y_c[point1]);
            for(point2=point1+1;point2<x_c.size();++point2){
               Ellipse elipse2(xknob,yknob,x_c[point2],y_c[point2]);
               if(EllipseIntersect(xknob,yknob,x_c[point1],y_c[point1],
                                   xknob,yknob,x_c[point2],y_c[point2])
                 ){
                  ++colors[point1];
   ++colors[point2];
   }
   }
   }
   }
   */
    if (noppenForm_ == 1) // rectangular or elliptic
    {
        int point1, point2;
        for (point1 = 0; point1 < x_c.size(); ++point1)
        {
            for (point2 = point1 + 1; point2 < x_c.size(); ++point2)
            {
                float dx = x_c[point1] - x_c[point2];
                float dy = y_c[point1] - y_c[point2];
                if (fabs(dx) <= 2 * xknob && fabs(dy) <= 2 * yknob)
                {
                    ++colors[point1];
                    ++colors[point2];
                }
            }
            // we also check distances to domain limits
            if (x_c[point1] > 0.0 && x_c[point1] < xknob)
            {
                sendWarning("One point is too close to the left side");
                ++colors[point1];
            }
            if (x_c[point1] < grundZellenBreite_ && grundZellenBreite_ - x_c[point1] < xknob)
            {
                sendWarning("One point is too close to the right side");
                ++colors[point1];
            }
            if (y_c[point1] > 0.0 && y_c[point1] < yknob)
            {
                sendWarning("One point is too close to the lower side");
                ++colors[point1];
            }
            if (y_c[point1] < grundZellenHoehe_ && grundZellenHoehe_ - y_c[point1] < yknob)
            {
                sendWarning("One point is too close to the upper side");
                ++colors[point1];
            }
        }
    }
    else if (noppenForm_ == 2 || noppenForm_ == 3) // ellipse
    {
        int point1, point2;
        for (point1 = 0; point1 < x_c.size(); ++point1)
        {
            for (point2 = point1 + 1; point2 < x_c.size(); ++point2)
            {
                float dx = (x_c[point1] - x_c[point2]) / xknob;
                float dy = (y_c[point1] - y_c[point2]) / yknob;
                if (dx * dx + dy * dy <= 4)
                {
                    ++colors[point1];
                    ++colors[point2];
                }
            }
            // we also check distances to domain limits
            if (x_c[point1] > 0.0 && x_c[point1] < xknob)
            {
                sendWarning("One point is too close to the left side");
                ++colors[point1];
            }
            if (x_c[point1] < readGrundZellenBreite_ && grundZellenBreite_ - x_c[point1] < xknob)
            {
                sendWarning("One point is too close to the right side");
                ++colors[point1];
            }
            if (y_c[point1] > 0.0 && y_c[point1] < yknob)
            {
                sendWarning("One point is too close to the lower side");
                ++colors[point1];
            }
            if (y_c[point1] < grundZellenHoehe_ && grundZellenHoehe_ - y_c[point1] < yknob)
            {
                sendWarning("One point is too close to the upper side");
                ++colors[point1];
            }
        }
    }
    return 0;
}

void
Design::setTransformAttribute(std::string &Transform)
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
    sprintf(buf, "%d %d\n", DanzahlReplikationenX_, DanzahlReplikationenY_);
    Transform += buf;
}

int
Design::checkReadFlags()
{
    if (!readGrundZellenHoehe_)
    {
        sendWarning("Could not read GrundZellenHoehe");
        return -1;
    }
    if (!readGrundZellenBreite_)
    {
        sendWarning("Could not read GrundZellenBreite");
        return -1;
    }
    if (!readAnzahlLinien_)
    {
        sendWarning("Could not read AnzahlLinien");
        return -1;
    }
    if (!readAnzahlPunkteProLinie_)
    {
        sendWarning("Could not read AnzahlPunkteProLinie");
        return -1;
    }
    if (!readVersatz_)
    {
        sendWarning("Could not read Versatz");
        return -1;
    }
    // knob stuff
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
    // free or param
    if (!readFree_or_param_)
    {
        sendWarning("Could not read free_or_param");
        return -1;
    }
    // free case
    int noppen_count;
    for (noppen_count = 0; noppen_count < MAX_POINTS; ++noppen_count)
    {
        if (!readFreie_noppen_[noppen_count])
        {
            sendWarning("Could not read freie_noppen_");
            return -1;
        }
    }
    if (!readNum_points_)
    {
        sendWarning("Could not read num_points");
        return -1;
    }
    return 0;
}

void
Design::outputDummies()
{
    p_grundZelle_->setCurrentObject(new coDoPolygons(p_grundZelle_->getObjName(), 0, 0, 0));
    p_noppenPositionen_->setCurrentObject(new coDoPoints(p_noppenPositionen_->getObjName(), 0));
    std::string transfAttr;
    setTransformAttribute(transfAttr);
    coDistributedObject *dummy = new coDoPolygons(p_show_grundZelle_->getObjName(), 0, 0, 0);
    dummy->addAttribute("TRANSFORM", transfAttr.c_str());
    p_show_phys_fuesse_->setCurrentObject(new coDoLines(p_show_phys_fuesse_->getObjName(),
                                                        0, 0, 0));
    p_show_grundZelle_->setCurrentObject(dummy);
    p_show_noppenPositionen_->setCurrentObject(new coDoPolygons(p_show_noppenPositionen_->getObjName(), 0, 0, 0));
    p_noppenColors_->setCurrentObject(new coDoFloat(
        p_noppenColors_->getObjName(), 0));
}

void
Design::MakeTheKnobs(const vector<float> &x_t,
                     const vector<float> &y_t,
                     const vector<float> &z_t)
{
    // get the input knob and replicate it with translations
    const coDistributedObject *in_knob = p_Knob_->getCurrentObject();
    if (!in_knob || !in_knob->isType("POLYGN"))
    {
        sendWarning("got a NULL pointer to the knob or a wrong type");
        return;
    }
#ifndef _MERGE_
    coDoPolygons *inKnob = (coDoPolygons *)in_knob;
#else
    coDoPolygons *inKnobNotMerged = (coDoPolygons *)in_knob;
    // merge nodes
    string temporaryObject = p_show_noppenPositionen_->getObjName();
    temporaryObject += "_Temporal";
    coDoPolygons *inKnob = (coDoPolygons *)(checkUSG(inKnobNotMerged, temporaryObject.c_str()));
#endif

    int no_points = inKnob->getNumPoints();
    int no_vert = inKnob->getNumVertices();
    int no_poly = inKnob->getNumPolygons();
    float *x_c;
    float *y_c;
    float *z_c;
    int *v_l;
    int *l_l;
    inKnob->getAddresses(&x_c, &y_c, &z_c, &v_l, &l_l);

    int copy_i;
    vector<float> xc, yc, zc;
    vector<int> vl, ll;
    for (copy_i = 0; copy_i < x_t.size(); ++copy_i)
    {
        int poly;
        for (poly = 0; poly < no_poly; ++poly)
        {
            ll.push_back(l_l[poly] + vl.size());
        }
        int vert;
        for (vert = 0; vert < no_vert; ++vert)
        {
            vl.push_back(v_l[vert] + xc.size());
        }
        int point;
        for (point = 0; point < no_points; ++point)
        {
            xc.push_back(x_c[point] + x_t[copy_i]);
            yc.push_back(y_c[point] + y_t[copy_i]);
            zc.push_back(z_c[point] + z_t[copy_i]);
        }
    }
    coDoPolygons *replications = new coDoPolygons(
        p_show_noppenPositionen_->getObjName(),
        xc.size(), vl.size(), ll.size());
    {
        float *x_c, *y_c, *z_c;
        int *v_l, *l_l;
        replications->getAddresses(&x_c, &y_c, &z_c, &v_l, &l_l);
        copy(xc.begin(), xc.end(), x_c);
        copy(yc.begin(), yc.end(), y_c);
        copy(zc.begin(), zc.end(), z_c);
        copy(vl.begin(), vl.end(), v_l);
        copy(ll.begin(), ll.end(), l_l);
    }
#ifdef _MERGE_
    inKnob->destroy();
    delete inKnob;
#endif
    p_show_noppenPositionen_->setCurrentObject(replications);
}

MODULE_MAIN(SCA, Design)
