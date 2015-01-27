/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "KnobShape.h"
#include "ReadASCIIDyna.h"
#include <alg/DeleteUnusedPoints.h>
#include <do/coDoText.h>
#include <do/coDoSet.h>

#include "ResultDataBase.h"
#include "ResultFloatParam.h"
#include "ResultEnumParam.h"

KnobShape::~KnobShape()
{
}

int
KnobShape::compute(const char *port)
{
    (void)port; // silence compiler

    if (p_upDiv_->getValue() < 2 || p_downDiv_->getValue() < 2)
    {
        sendWarning("The number of divisons may not be lower than 2");
        return FAIL;
    }
    if (p_tolerance_->getValue() <= 0.0 || p_tolerance_->getValue() > 1.0)
    {
        sendWarning("The relative tolerance is a number greater than 0 and smaller than 1");
        return FAIL;
    }
    readNoppenHoehe_ = readAusrundungsRadius_ = readAbnutzungsRadius_ = false;
    readNoppenWinkel_ = readNoppenForm_ = readLaenge1_ = readLaenge2_ = false;
    readTissueTyp_ = readGummiHaerte_ = readAnpressDruck_ = false;

    // get the text object
    const coDistributedObject *inObj = p_knobParam_->getCurrentObject();
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
        p_showPoly_->setCurrentObject(new coDoPolygons(p_showPoly_->getObjName(), 0, 0, 0));
        p_showNormals_->setCurrentObject(new coDoVec3(p_showNormals_->getObjName(), 0));
        p_handOutPoly_->setCurrentObject(new coDoPolygons(p_handOutPoly_->getObjName(), 0, 0, 0));
        return SUCCESS;
    }
    char *text;
    theText->getAddress(&text);
    istringstream strText;
    strText.str(string(text));
    int maxLen = strlen(text) + 1;
    vector<char> name(maxLen);
    while (strText >> &name[0])
    {
        if (strcmp(&name[0], "noppenHoehe") == 0)
        {
            if (ReadASCIIDyna::readFloatSlider(strText, &noppenHoehe_, maxLen) != 0)
            {
                sendWarning("Could not read noppenHoehe");
                return FAIL;
            }
            readNoppenHoehe_ = true;
        }
        else if (strcmp(&name[0], "ausrundungsRadius") == 0)
        {
            if (ReadASCIIDyna::readFloatSlider(strText, &ausrundungsRadius_, maxLen) != 0)
            {
                sendWarning("Could not read rundZellenBreite");
                return FAIL;
            }
            readAusrundungsRadius_ = true;
        }
        else if (strcmp(&name[0], "abnutzungsRadius") == 0)
        {
            if (ReadASCIIDyna::readFloatSlider(strText, &abnutzungsRadius_, maxLen) != 0)
            {
                sendWarning("Could not read abnutzungsRadius");
                return FAIL;
            }
            readAbnutzungsRadius_ = true;
        }
        else if (strcmp(&name[0], "noppenWinkel") == 0)
        {
            if (ReadASCIIDyna::readFloatSlider(strText, &noppenWinkel_, maxLen) != 0)
            {
                sendWarning("Could not read noppenWinkel");
                return FAIL;
            }
            readNoppenWinkel_ = true;
        }
        else if (strcmp(&name[0], "noppenForm") == 0)
        {
            if (ReadASCIIDyna::readChoice(strText, &noppenForm_, maxLen) != 0)
            {
                sendWarning("Could not read noppenForm");
                return FAIL;
            }
            if (noppenForm_ == 3)
            {
                noppenForm_ = 2;
            }
            readNoppenForm_ = true;
        }
        else if (strcmp(&name[0], "laenge1") == 0)
        {
            if (ReadASCIIDyna::readFloatSlider(strText, &laenge1_, maxLen) != 0)
            {
                sendWarning("Could not read laenge1");
                return FAIL;
            }
            readLaenge1_ = true;
        }
        else if (strcmp(&name[0], "laenge2") == 0)
        {
            if (ReadASCIIDyna::readFloatSlider(strText, &laenge2_, maxLen) != 0)
            {
                sendWarning("Could not read laenge2");
                return FAIL;
            }
            readLaenge2_ = true;
        }
        else if (strcmp(&name[0], "tissueTyp") == 0)
        {
            if (ReadASCIIDyna::readChoice(strText, &tissueTyp_, maxLen) != 0)
            {
                sendWarning("Could not read tissueTyp");
                return FAIL;
            }
            readTissueTyp_ = true;
        }
        else if (strcmp(&name[0], "gummiHaerte") == 0)
        {
            if (ReadASCIIDyna::readFloatSlider(strText, &gummiHaerte_, maxLen) != 0)
            {
                sendWarning("Could not read gummiHaerte");
                return FAIL;
            }
            readGummiHaerte_ = true;
        }
        else if (strcmp(&name[0], "anpressDruck") == 0)
        {
            if (ReadASCIIDyna::readFloatSlider(strText, &anpressDruck_, maxLen) != 0)
            {
                sendWarning("Could not read anpressDruck");
                return FAIL;
            }
            readAnpressDruck_ = true;
        }
    }
    if (checkReadFlags() != 0)
    {
        return FAIL;
    }

    /*
      // now we have to get the path where the knob is
      coString getPath;
      int knob_not_found = checkKnobPath(getPath);
      if(knob_not_found){
         sendWarning("Could not find knob");
         return FAIL;
      }

      // now get file names
      coString embConnFile(getPath.getValue());
   coString embDispFile(getPath.getValue());
   coString embThickFile(getPath.getValue());

   ReadASCIIDyna::loadFileNames(embConnFile,embDispFile,embThickFile);

   // open files
   ifstream emb_conn(embConnFile.getValue());
   ifstream emb_displ(embDispFile.getValue());
   ifstream emb_thick(embThickFile.getValue());

   if(!emb_conn.rdbuf()->is_open()){
   sendWarning("Could not open connectivity embossing file");
   return FAIL;
   }
   if(!emb_displ.rdbuf()->is_open()){
   sendWarning("Could not open displacements embossing file");
   return FAIL;
   }
   if(!emb_thick.rdbuf()->is_open()){
   sendWarning("Could not open thickness embossing file");
   return FAIL;
   }
   */

    vector<int> epl, ecl;
    vector<float> exc;
    vector<float> eyc;
    vector<float> ezc;
    vector<float> dicke;
    vector<float> nxc;
    vector<float> nyc;
    vector<float> nzc;
    if (noppenForm_ == 2)
    {
        ReadASCIIDyna::ellipticKnob(noppenHoehe_, ausrundungsRadius_,
                                    abnutzungsRadius_, noppenWinkel_, laenge1_, laenge2_, 6,
                                    p_upDiv_->getValue(), p_downDiv_->getValue(),
                                    epl, ecl, exc, eyc, ezc, nxc, nyc, nzc);
    }
    else if (noppenForm_ == 1)
    {
        ReadASCIIDyna::rectangularKnob(noppenHoehe_, ausrundungsRadius_,
                                       abnutzungsRadius_, noppenWinkel_, laenge1_, laenge2_, 3,
                                       p_upDiv_->getValue(), p_downDiv_->getValue(),
                                       epl, ecl, exc, eyc, ezc, nxc, nyc, nzc);
    }
    vector<int> keepEcl(ecl);
    vector<int> keepEpl(epl);
    ReadASCIIDyna::Mirror2(nxc, nyc, nzc, keepEcl, keepEpl, ReadASCIIDyna::X);
    ReadASCIIDyna::Mirror2(nxc, nyc, nzc, keepEcl, keepEpl, ReadASCIIDyna::Y);

    ReadASCIIDyna::Mirror2(exc, eyc, ezc, ecl, epl, ReadASCIIDyna::X);
    ReadASCIIDyna::Mirror2(exc, eyc, ezc, ecl, epl, ReadASCIIDyna::Y);
    // merge nodes on the YZ, ZX planes in the elliptic case...
    if (noppenForm_ == 2) // elliptic
    {
        ReadASCIIDyna::MergeNodes(exc, eyc, ezc, ecl, epl, 1e-5);
    }
    // coDoPolygons *auxCell =
    coDistributedObject *basicCell = NULL;
    {
        float *x, *y, *z;
        int *c, *p;
        basicCell = new coDoPolygons(p_handOutPoly_->getObjName(), exc.size(),
                                     ecl.size(), epl.size());
        ((coDoPolygons *)basicCell)->getAddresses(&x, &y, &z, &c, &p);
        copy(exc.begin(), exc.end(), x);
        copy(eyc.begin(), eyc.end(), y);
        copy(ezc.begin(), ezc.end(), z);
        copy(ecl.begin(), ecl.end(), c);
        copy(epl.begin(), epl.end(), p);
    }
    p_handOutPoly_->setCurrentObject(basicCell);

    /*
      coString auxName = p_handOutPoly_->getObjName();
      auxName += "_Aux";
   */

    // coDistributedObject *basicCell = checkUSG(auxCell,p_handOutPoly_->getObjName());
    // delete auxCell;
    basicCell->addAttribute("vertexOrder", "2");

    p_handOutPoly_->setCurrentObject(basicCell);
    if (p_knobParam_->getCurrentObject()->getAttribute("KNOB_SELECT"))
    {
        coDistributedObject *setList[2];
        basicCell->incRefCount();
        setList[0] = basicCell;
        setList[1] = NULL;
        p_showPoly_->setCurrentObject(new coDoSet(p_showPoly_->getObjName(), setList));
        // now the normals
        string auxNormName = p_showNormals_->getObjName();
        auxNormName += "_Aux";
        coDistributedObject *theNormals = NULL;
        {
            theNormals = new coDoVec3(auxNormName.c_str(), nxc.size());
            float *x, *y, *z;
            ((coDoVec3 *)theNormals)->getAddresses(&x, &y, &z);
            copy(nxc.begin(), nxc.end(), x);
            copy(nyc.begin(), nyc.end(), y);
            copy(nzc.begin(), nzc.end(), z);
        }
        coDistributedObject *setNormalList[2];
        setNormalList[0] = theNormals;
        setNormalList[1] = NULL;
        p_showNormals_->setCurrentObject(new coDoSet(p_showNormals_->getObjName(), setNormalList));
    }
    else
    {
        p_showPoly_->setCurrentObject(new coDoPolygons(p_showPoly_->getObjName(), 0, 0, 0));
        p_showNormals_->setCurrentObject(new coDoVec3(p_showNormals_->getObjName(), 0));
    }

    return SUCCESS;
}

int
KnobShape::checkReadFlags()
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
    /*
      if(!readTissueTyp_){
         sendWarning("Could not read TissueTyp");
         return -1;
      }
      if(!readGummiHaerte_){
         sendWarning("Could not read GummiHaerte");
         return -1;
      }
      if(!readAnpressDruck_){
         sendWarning("Could not read AnpressDruck");
   return -1;
   }
   */
    return 0;
}

KnobShape::KnobShape(int argc, char *argv[])
    : coModule(argc, argv, "find and show knob")
{
    p_knobParam_ = addInputPort("KnobParam", "coDoText", "Knob parameters");
    p_showPoly_ = addOutputPort("TheKnob", "coDoPolygons", "The knob");
    p_showNormals_ = addOutputPort("TheNormals", "coDoVec3|DO_Unstructured_V3D_Normals", "The normals");
    p_handOutPoly_ = addOutputPort("TheSameKnob", "coDoPolygons", "The knob for modules");

    //
    p_upDiv_ = addInt32Param("AbrasionDiv", "Abrasion line divisons");
    p_upDiv_->setValue(4);
    p_downDiv_ = addInt32Param("RoundingDiv", "Rounding line divisons");
    p_downDiv_->setValue(4);
    p_tolerance_ = addFloatParam("Tolerance", "Relative tolerance");
    p_tolerance_->setValue(0.01);
}

// 0 -> exact hit
// 1 -> approx.
// 2 -> not found
int
KnobShape::checkKnobPath(string &getPath)
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
    ResultFloatParam v0("noppenHoehe", noppenHoehe_, 3);
    ResultFloatParam v1("ausrundungsRadius", ausrundungsRadius_, 3);
    ResultFloatParam v2("abnutzungsRadius", abnutzungsRadius_, 3);
    ResultFloatParam v3("noppenWinkel", noppenWinkel_, 3);
    // char *labs[] = { "Raute", "Ellipse" };
    ResultEnumParam v4("noppenForm", 3, ReadASCIIDyna::noppenFormChoices, noppenForm_ - 1);
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
    const char *path = dataB.searchForResult(diff, list, FinalCandidates,
                                             p_tolerance_->getValue());

    if (path == NULL)
    {
        sendWarning("Could not find knob with the given parameters");
        return 2;
    }
    getPath = path;
    int candidate;
    for (candidate = 0; candidate < FinalCandidates.size(); ++candidate)
    {
        delete FinalCandidates[candidate];
    }
    return (diff != 0.0);
}

MODULE_MAIN(SCA, KnobShape)
