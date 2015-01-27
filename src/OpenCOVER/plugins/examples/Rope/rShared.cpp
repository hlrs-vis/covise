/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#include "rShared.h"
using namespace osg;
using namespace opencover;

rShared::rShared()
{
    this->setLenFactor(1.0);
    this->setLengthOfTwist(0.0);
    this->elemID = -1;
    this->elemName = NULL;
    this->ropeLength = 1000.0;
    this->segHeight = 10.0;
    this->numSegments = 8;
    this->orientation = osg::Vec3(0, 0, 1);
}

rShared::~rShared()
{
    if (this->elemName)
        free(this->elemName);
}

osg::Vec4 rShared::getColTr(void)
{
    osg::Vec4 color;
    if (daddy)
        color = daddy->getColTr();
    else
        color = Vec4(this->colTr->getRed(), this->colTr->getGreen(), this->colTr->getBlue(), 1.0);
    return color;
}

void rShared::initShared(rShared *daddy, char *name, int id, coTUIFrame *frame, coTUIComboBox *box)
{
    char buf[256];

    this->daddy = daddy;
    this->elemID = id;
    if (this->elemName)
        free(this->elemName);
    this->elemName = strdup(name);
    sprintf(buf, "%s_%d", name, id);
    this->elemIDName = strdup(buf);
    this->coverGroup = new osg::Group();
    this->coverGroup->setName(name);
    this->coverFrame = frame;
    this->coverBox = box;
    if (this->daddy)
    { // Wenn Papa, dann da dran haengen
        char buf[512];

        this->daddy->coverGroup->addChild(this->coverGroup.get());
        this->depth = daddy->depth + 1;
        sprintf(buf, "%s/%s", this->daddy->getIDNamePath(), this->getIDName());
        this->elemIDNamePath = strdup(buf);
        this->ropeLength = daddy->ropeLength; // gesetzt wird das im rope ...
    }
    else
    { // ... sonst an den COVER
        cover->getObjectsRoot()->addChild(this->coverGroup.get());
        this->depth = 0;
        this->elemIDNamePath = strdup(this->getIDName());
    }
    if (this->coverBox)
        this->coverBox->addEntry(this->elemIDNamePath);
}

float rShared::setLenFactor(float val)
{
    if (val >= 0.0 && val <= 10.0)
        this->elemLenFactor = val;
    return this->getLenFactor();
}

float rShared::getLenFactor()
{
    return this->elemLenFactor;
}

float rShared::getLength()
{
    float len = this->getLenFactor();

    if (this->daddy)
        len *= this->daddy->getLength();

    return len;
}

float rShared::setLengthOfTwist(float lengthOfTwist)
{
    if (lengthOfTwist != 0.0)
        this->setStateLengthOfTwist_On();
    else
        this->setStateLengthOfTwist_Off();
    this->elemLengthOfTwist = lengthOfTwist;
    return this->getLengthOfTwist();
}

float rShared::getLengthOfTwist(void)
{
    return this->elemLengthOfTwist;
}

float rShared::setPosAngle(float angle)
{
    this->elemAngle = angle;
    return this->getPosAngle();
}

float rShared::getPosAngle(void)
{
    return this->elemAngle;
}

float rShared::setPosRadius(float R)
{
    if (R >= 0.0)
        this->elemR = R;
    return this->getPosRadius();
}

float rShared::getPosRadius(void)
{
    return this->elemR;
}

void rShared::setOrientation(osg::Vec3 O)
{
    if (daddy)
        daddy->setOrientation(O);
    else
        this->orientation = osg::Vec3(0, 0, 1);
    // rei, TODO ... muss derzeit statisch sein, sonst knallts ..
}

osg::Vec3 rShared::getOrientation(void)
{
    if (daddy)
        return daddy->getOrientation();
    else
        return this->orientation;
}

// Diese Werte werden immer nur beim Seil-Objekt gespeichert
void rShared::setNumSegments(int val)
{
    if (this->daddy)
    {
        this->daddy->setNumSegments(val);
    }
    else
    {
        if (val >= 3 && val <= 128)
        {
            this->numSegments = val;
        }
    }
}

int rShared::getNumSegments(void)
{
    if (this->daddy)
        return this->daddy->getNumSegments();
    else
        return this->numSegments;
}

void rShared::setSegHeight(float val)
{
    if (daddy)
    {
        daddy->setSegHeight(val);
    }
    else
    {
        if (val >= 1.0 && val <= 100.0)
        {
            this->segHeight = val;
        }
    }
}

float rShared::getSegHeight(void)
{
    if (daddy)
        return daddy->getSegHeight();
    else
        return this->segHeight;
}

int rShared::getID(void)
{
    return this->elemID;
}

char *rShared::getName(void)
{
    return this->elemName;
}

char *rShared::getIDName(void)
{
    return this->elemIDName;
}

char *rShared::getIDNamePath(void)
{
    return this->elemIDNamePath;
}

float rShared::getRopeLength(void)
{
    if (daddy)
        return daddy->getRopeLength();
    else
        return this->ropeLength;
}

void rShared::setRopeLength(float val)
{
    this->ropeLength = ((val > 100.0 && val < 5000.0) ? val : 1000.0);
}

float rShared::getElemLength(void)
{
    return this->getRopeLength() * this->getLength();
}

void rShared::setStateLengthOfTwist_On(void)
{
    this->elemStateLengthOfTwist = true;
}

void rShared::setStateLengthOfTwist_Off(void)
{
    this->elemStateLengthOfTwist = false;
}

bool rShared::getStateLengthOfTwist(void)
{
    return this->elemStateLengthOfTwist;
}

char *rShared::identStr(void)
{
    static char buf[256];
    char *p = buf;
    int len = this->depth + 1;

    memset(buf, 0, sizeof(buf));
    *p = '\n';
    while (len--)
    {
        *++p = ' ';
        *++p = ' ';
    }
    return buf;
}

int rShared::getElemID(void)
{
    return this->elemID;
}

bool rShared::isCore(void)
{
    return this->elemCore;
}
void rShared::setCore(bool state)
{
    this->elemCore = state;
}

float rShared::rad2grad(float rad)
{
    return (rad / (2.0 * M_PI) * 360.0);
}

float rShared::grad2rad(float grad)
{
    return (grad / 360.0 * (2.0 * M_PI));
}
