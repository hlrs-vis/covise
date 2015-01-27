/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description:                                                        ++
// ++             Implementation of class InvAnnoFlag                     ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date:  22.10.2001                                                   ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <covise/covise.h>

//#include <X11/keysymdef.h>
#include <X11/keysym.h>

#include "InvAnnotationFlag.h"

#include <Inventor/nodes/SoLineSet.h>
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoRotationXYZ.h>
#include <Inventor/nodes/SoCone.h>
#include <Inventor/nodes/SoMaterial.h>
#include <Inventor/nodes/SoFont.h>
#include <Inventor/nodes/SoScale.h>

#include <Inventor/SoPickedPoint.h>

#include "ModuleInfo.h"
#include <util/string_util.h>

//
// Default Constructor
//
InvAnnoFlag::InvAnnoFlag()
    : InvActiveNode()
    , iNorm_(0, -1, 0)
    , alive_(true)
    , num_(0)
    , hostname_(ModuleInfo->getHostname())
    , scale_(1.0)
{
    // create flag
    transl_ = new SoTranslation;
    getSwitch()->addChild(transl_);

    rota_ = new SoRotation;
    getSwitch()->addChild(rota_);

    reScale_ = new SoScale;
    getSwitch()->addChild(reScale_);

    getSwitch()->addChild(makeArrow());
}

//
// construct a flag with a given instance number
//
InvAnnoFlag::InvAnnoFlag(const int &num)
    : InvActiveNode()
    , iNorm_(0, -1, 0)
    , alive_(true)
    , num_(0)
    , hostname_(ModuleInfo->getHostname())
    , scale_(1.0)
{
    num_ = num;

    // create flag
    transl_ = new SoTranslation;
    getSwitch()->addChild(transl_);

    rota_ = new SoRotation;
    getSwitch()->addChild(rota_);

    reScale_ = new SoScale;
    getSwitch()->addChild(reScale_);

    getSwitch()->addChild(makeArrow());

    char name[64];
    char chNum[8];
    sprintf(chNum, "%d", num);
    strcpy(name, "ANNOTATION-");
    strcat(name, chNum);

    getSeparator()->setName(SbName(name));
}

//
// parse serialize string to construct obj
//
InvAnnoFlag::InvAnnoFlag(const SerializeString &str)
    : InvActiveNode()
    , iNorm_(0, -1, 0)
    , alive_(true)
    , num_(0)
{
    std::string tok("<InvAnnoFlag");
    size_t r = str.find_first_of(tok.c_str());

    // if we didn't find the initial token: everything is done.
    // Somebody else will solve the problem
    if (r == (size_t)std::string::npos)
    {
        alive_ = false;
        return;
    }

    int errCnt(0);

    // try to get all other information
    std::string hostn = parseToken(str, "<H");
    if (hostn.empty())
        errCnt++;
    hostname_ = hostn;

    std::string inst = parseToken(str, "<IN");
    if (inst.empty())
        errCnt++;
    int ret = sscanf(inst.c_str(), "%d", &num_);
    if (ret != 1)
    {
        fprintf(stderr, "InvAnnoFlag::InvAnnoFlag: sscanf1 failed\n");
    }

    std::string text = parseToken(str, "<T");
    if (text.empty())
        errCnt++;

    std::string tmpStr = parseToken(str, "<P");
    if (tmpStr.empty())
        errCnt++;
    float xPos, yPos, zPos;
    ret = sscanf(tmpStr.c_str(), "%f|%f|%f", &xPos, &yPos, &zPos);
    if (ret != 3)
    {
        fprintf(stderr, "InvAnnoFlag::InvAnnoFlag: sscanf2 failed\n");
    }

    tmpStr = parseToken(str, "<RR");
    if (tmpStr.empty())
        errCnt++;
    float q[4];
    ret = sscanf(tmpStr.c_str(), "%f|%f|%f|%f", &q[0], &q[1], &q[2], &q[3]);
    if (ret != 4)
    {
        fprintf(stderr, "InvAnnoFlag::InvAnnoFlag: sscanf3 failed\n");
    }

    tmpStr = parseToken(str, "<SC");
    if (tmpStr.empty())
        errCnt++;
    float scl;
    ret = sscanf(tmpStr.c_str(), "%f", &scl);
    if (ret != 1)
    {
        fprintf(stderr, "InvAnnoFlag::InvAnnoFlag: sscanf4 failed\n");
    }
    scale_ = scl;

    // if one part of the serialize str is missing we are created as a
    // kind of zombie and we will die soon
    if (errCnt > 0)
    {
        alive_ = false;
        return;
    }

    // create flag
    transl_ = new SoTranslation;
    getSwitch()->addChild(transl_);

    rota_ = new SoRotation;
    getSwitch()->addChild(rota_);

    reScale_ = new SoScale;
    getSwitch()->addChild(reScale_);

    getSwitch()->addChild(makeArrow());

    // set group node information
    char name[64];
    char chNum[8];
    sprintf(chNum, "%d", num_);
    strcpy(name, "ANNOTATION-");
    strcat(name, chNum);

    getSeparator()->setName(SbName(name));

    // set all other data
    textField_.append(text);
    // instead of blanks the textfield contains tags - remove it here
    textField_.untag();

    SbVec3f point(xPos, yPos, zPos);
    transl_->translation = point;

    SbRotation rot(q[0], q[1], q[2], q[3]);
    rota_->rotation = rot;
}

std::string
InvAnnoFlag::parseToken(const std::string &str, const char *tok) const
{
    char chFin[3];
    strcpy(chFin, "\\>\0");
    std::string ret;

    size_t r = str.find_first_of(tok);
    // we return an empty string if tok is not found
    if (r == (size_t)std::string::npos)
        return ret;
    // we have to add the length of the token we dont want it in retval
    size_t rs = r + strlen(tok);
    r = str.find_first_of(chFin, r);
    ret = strip(str.substr(rs, (r - rs)));

    return ret;
}

bool
InvAnnoFlag::isAlive()
{
    return alive_;
}

bool
    InvAnnoFlag::
    operator==(const SerializeString &str) const
{
    // a flag equal a serString if the serString descrobes a flag
    // and has the same <H ..\> and <IN .. \> entry
    std::string tok("<InvAnnoFlag");
    size_t r = str.find_first_of(tok.c_str());

    // if we didn't find the initial token: everything is done.
    if (r == (size_t)std::string::npos)
    {
        return false;
    }

    bool ret(true);
    std::string hostn = parseToken(str, "<H");
    if (hostn.empty())
        return false;
    ret = ret && (hostname_ == hostn);

    std::string inst = parseToken(str, "<IN");
    if (inst.empty())
        return false;
    int inm;
    int retval = sscanf(inst.c_str(), "%d", &inm);
    if (retval != 1)
    {
        fprintf(stderr, "InvAnnoFlag::operator==: sscanf failed\n");
    }
    ret = ret && (num_ == inm);

    std::string text = parseToken(str, "<T");
    if (text.empty())
        return false;
    ret = ret && (textField_.getTaggedText() == text);

    return ret;
}

//
// Destructor
//
InvAnnoFlag::~InvAnnoFlag()
{
}

SerializeString
InvAnnoFlag::serialize() const
{
    std::string fin("\\>");

    // we use a tagged string
    SerializeString str("<InvAnnoFlag");
    str = str + std::string("<H") + hostname_ + fin; // <H hostname />

    char tmp[32];
    sprintf(tmp, "%d", num_);

    // <IN Instance />
    str = str + std::string("<IN") + std::string(tmp) + fin;

    SbVec3f point = transl_->translation.getValue();

    sprintf(tmp, "%#f", point[0]);
    std::string xPos(tmp);

    sprintf(tmp, "%#f", point[1]);
    std::string yPos(tmp);

    sprintf(tmp, "%#f", point[2]);
    std::string zPos(tmp);
    std::string spc("|");

    str = str + std::string("<P") + xPos + spc + yPos + spc + zPos + fin;

    SbRotation rot = rota_->rotation.getValue();
    float q[4];
    rot.getValue(q[0], q[1], q[2], q[3]);

    std::string rotStr;
    int i;
    for (i = 0; i < 4; ++i)
    {
        sprintf(tmp, "%#f", q[i]);
        rotStr = rotStr + std::string(tmp) + spc;
    }
    str = str + std::string("<RR") + rotStr + fin;

    sprintf(tmp, "%#f", scale_);
    std::string sclStr(tmp);
    str = str + std::string("<SC") + sclStr + fin;

    str = str + std::string("<T") + textField_.getTaggedText() + fin;

    str = str + fin;

    return str;
}

SoGroup *
InvAnnoFlag::makeArrow()
{
    static float vertexPos[2][3] = {
        { 0, 0, 0 },
        { 0, -0.7, 0 }
    };

    SoGroup *arrow = new SoGroup;

    SoLineSet *line = new SoLineSet;

    SoCoordinate3 *coords = new SoCoordinate3;
    coords->point.setValues(0, 2, vertexPos);

    arrow->addChild(coords);

    arrow->addChild(line);

    SoTranslation *transl = new SoTranslation;
    transl->translation.setValue(0, -0.15, 0);
    arrow->addChild(transl);

    SoCone *tip = new SoCone;
    tip->bottomRadius = 0.1;
    tip->height = 0.3;

    arrow->addChild(tip);

    // Choose a font
    SoFont *myFont = new SoFont;
    myFont->name.setValue("Times-Roman");
    myFont->size.setValue(12.0);
    arrow->addChild(myFont);

    // Add Text2 for TEXT, translated to proper location.
    SoSeparator *textSep = new SoSeparator;

    SoTranslation *textTranslate = new SoTranslation;
    textTranslate->translation.setValue(0, -0.71, 0);

    annoText_ = new SoText2;

    textSep->addChild(textTranslate);
    textSep->addChild(annoText_);

    arrow->addChild(textSep);

    return arrow;
}

void
InvAnnoFlag::setPickedPoint(const SoPickedPoint *pick, const SbVec3f &camPos)
{

    SbVec3f point = pick->getPoint();
    SbVec3f normal = pick->getNormal();

    if (camPos.dot(normal) < 0)
    {
        normal = -normal;
    }

    //    cerr << "  picked POINT " << point[0] << " : " << point[1] << " : " << point[2] << endl;

    SbRotation rota(iNorm_, normal);

    transl_->translation = point;
    rota_->rotation = rota;
}

void
InvAnnoFlag::setText()
{

    // include text in the scene-graph
    int i;
    int numLines = textField_.getNumLines();
    char **tsr = new char *[numLines];

    for (i = 0; i < numLines; ++i)
    {
        std::string line = textField_.getLine(i);
        tsr[i] = new char[1 + line.size()];
        strcpy(tsr[i], line.c_str());
    }

    if (annoText_)
    {
        annoText_->string.setValue("");
        annoText_->string.setValues(0, numLines, (const char **)tsr);
    }
}

void
InvAnnoFlag::setText(const char *cstr)
{
    // append text to the formating field
    textField_.append(cstr);

    setText();
}

void
InvAnnoFlag::setBackSpace(const int &n)
{
    textField_.backSpace(n);

    // include it in the scene-graph
    int i;
    int numLines = textField_.getNumLines();
    char **tsr = new char *[numLines];

    for (i = 0; i < numLines; ++i)
    {
        std::string line = textField_.getLine(i);
        tsr[i] = new char[1 + line.size()];
        strcpy(tsr[i], line.c_str());
    }

    if (annoText_)
    {
        annoText_->string.setValue("");
        annoText_->string.setValues(0, numLines, (const char **)tsr);
    }
}

void
InvAnnoFlag::clearText()
{
    textField_.clear();

    if (annoText_)
    {
        annoText_->string.setValue("");
    }
}

void
InvAnnoFlag::setTextLine(const std::string &str)
{
    if (annoText_)
    {
        annoText_->string.setValue(str.c_str());
    }
}

void
InvAnnoFlag::selectionCB(void *me, SoPath *p)
{
    InvAnnoFlag *mee = static_cast<InvAnnoFlag *>(me);

    InvActiveNode::selectionCB(me, p);
    std::string auf("please enter annotation!");
    mee->setTextLine(auf);
    //    mee->setKbActive();
}

void
InvAnnoFlag::deSelectionCB(void *me, SoPath *p)
{
    InvActiveNode::deSelectionCB(me, p);
    InvAnnoFlag *mee = static_cast<InvAnnoFlag *>(me);
    //    mee->setKbInactive();
    mee->textField_.clear();
    //    mee->setText();
}

int
InvAnnoFlag::getInstance()
{
    return num_;
}

void
InvAnnoFlag::reScale(const float &s)
{
    scale_ *= s;
    reScale_->scaleFactor.setValue(scale_, scale_, scale_);
}
