/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#define USE_MATH_DEFINES
#include <math.h>
#include <QDir>
#include <config/coConfig.h>
#include <device/VRTracker.h>
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/coVRTui.h>
#include <OpenVRUI/coTrackerButtonInteraction.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <osg/PolygonMode>
#include "strandgroup.h"
#include <osg/LineWidth>
#include <stdio.h>
using namespace osg;

Strandgroup::Strandgroup(rShared *daddy, int id, float lot, coTUIFrame *frame, coTUIComboBox *box)
{
    int i;

    this->setLenFactor(1.0); // default
    this->numStrands = 0;
    this->setLengthOfTwist(lot);
    this->setCore(lot == 0.0 ? true : false);
    this->LenSliderL = NULL;
    this->LOTSliderL = NULL;
    this->colorL = NULL;
    this->myL1 = NULL;
    this->sonL1 = NULL;
    this->sonL2 = NULL;
    this->myLenSlider = NULL;
    this->myLOTSlider = NULL;
    this->sonLenSlider = NULL;
    this->sonColorButton = NULL;
    for (i = 0; i < this->maxStrands; i++)
    {
        this->sgLenSliderL[i] = NULL;
        this->sgLenSlider[i] = NULL;
        this->sgColorButton[i] = NULL;
    }

    this->initShared(daddy, (char *)"Strandgroup", id, frame, box);
}

//Strandgroup Destructor delets all functions and selections, which have been made
Strandgroup::~Strandgroup()
{
    int i;

    this->delManipulation();
    for (i = 0; i < this->numStrands; i++)
        delete this->Strands[i];
}

void Strandgroup::createGeom()
{
    int i;

    fprintf(stderr, " Strandgroup::createGeom: %s\n", this->getName());
    for (i = 0; i < this->numStrands; i++)
        this->Strands[i]->createGeom();
}

int Strandgroup::addStrand(float R, float angle, coTUIFrame *frame, coTUIComboBox *box)
{
    if (this->numStrands < this->maxStrands)
    {
        this->Strands[this->numStrands] = new Strand(this, this->numStrands, R, angle, frame, box);
        return this->numStrands++;
    }
    return -1;
}

xercesc::DOMElement *Strandgroup::Save(xercesc::DOMDocument &document)
{
    char tmp[256];
    int i;

    xercesc::DOMElement *element = document.createElement(xercesc::XMLString::transcode(this->getName()));

    element->appendChild(document.createTextNode(xercesc::XMLString::transcode(this->identStr())));
    element->setAttribute(xercesc::XMLString::transcode("depth"), xercesc::XMLString::transcode(this->getName()));
    sprintf(tmp, "%f", this->getLengthOfTwist());
    element->setAttribute(xercesc::XMLString::transcode("LengthOfTwist"), xercesc::XMLString::transcode(tmp));
    for (i = 0; i < this->numStrands; i++)
    {
        element->appendChild(this->Strands[i]->Save(document));
        element->appendChild(document.createTextNode(xercesc::XMLString::transcode(this->identStr())));
    }

    return element;
}

void Strandgroup::Load(xercesc::DOMElement *rNode)
{
    int i;
    char *p;
    float posR;
    float posAngle;
    int len;
    int sID;

    xercesc::DOMNodeList *nodeList = rNode->getChildNodes();
    len = nodeList->getLength();
    fprintf(stderr, "    Strandgroup::Load : NumberOfNodes=%d\n", len);
    for (i = 0; i < len; ++i)
    {
        fprintf(stderr, "    Strandgroup::Load : nodeNo=%d\n", i);

        xercesc::DOMElement *node = dynamic_cast<xercesc::DOMElement *>(nodeList->item(i));
        if (node)
        {
            posR = posAngle = 1.0; // damit es net gar so arg knallen koennte ... ;(
            if ((p = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("posRadius")))) != NULL)
                posR = atof(p);
            if ((p = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("posAngle")))) != NULL)
                posAngle = this->grad2rad(atof(p));
            sID = this->addStrand(posR, posAngle, this->coverFrame, this->coverBox);
            this->Strands[sID]->Load(node);
        }
    }
}

void Strandgroup::addManipulation(coTUIFrame *frame)
{
    int i;
    int y = 0;

    // my ... ich ;)
    // son ... nur die Ebene drunter
    // all ... alles drunter

    // Kopfzeile pinseln ...
    y = 1;
    fprintf(stderr, "Strandgroup::addManipulation\n");
    LenSliderL = new coTUILabel("Length factor", frame->getID());
    LenSliderL->setPos(2, y);
    LOTSliderL = new coTUILabel("Length of twist", frame->getID());
    LOTSliderL->setPos(4, y);
    colorL = new coTUILabel("Color", frame->getID());
    colorL->setPos(6, y);

    // Erste Zeile ...
    y++;
    myL1 = new coTUILabel("Strandgroup", frame->getID());
    myL1->setPos(0, y);
    myLenSlider = new coTUIFloatSlider("numHSlider", frame->getID());
    myLenSlider->setEventListener(this);
    myLenSlider->setPos(2, y);
    myLenSlider->setMin(0.0);
    myLenSlider->setMax(3.0);
    myLenSlider->setValue(this->getLenFactor());
    myLenSlider->setSize(6, 2);
    if (!this->isCore())
    {
        myLOTSlider = new coTUIFloatSlider("numHSlider", frame->getID());
        myLOTSlider->setEventListener(this);
        myLOTSlider->setPos(4, y);
        myLOTSlider->setMin(-2000);
        myLOTSlider->setMax(2000);
        myLOTSlider->setValue(this->getLengthOfTwist());
        myLOTSlider->setSize(5, 0);
    }

    y++;
    sonL1 = new coTUILabel("Strand", frame->getID());
    sonL1->setPos(0, y);
    sonL2 = new coTUILabel("all", frame->getID());
    sonL2->setPos(1, y);
    sonLenSlider = new coTUIFloatSlider("numHSlider", frame->getID());
    sonLenSlider->setEventListener(this);
    sonLenSlider->setPos(2, y);
    sonLenSlider->setMin(0.0);
    sonLenSlider->setMax(3.0);
    sonLenSlider->setValue(1.0);
    sonLenSlider->setSize(6, 2);
    sonColorButton = new coTUIButton("color", frame->getID());
    sonColorButton->setEventListener(this);
    sonColorButton->setPos(6, y);

    for (i = 0, y++; i < this->numStrands; i++)
    {
        char buf[256];

        sprintf(buf, "No %2d", i);
        sgLenSliderL[i] = new coTUILabel(buf, frame->getID());
        sgLenSliderL[i]->setPos(1, y + i);
        sgLenSlider[i] = new coTUIFloatSlider("numHSlider", frame->getID());
        sgLenSlider[i]->setEventListener(this);
        sgLenSlider[i]->setPos(2, y + i);
        sgLenSlider[i]->setMin(0.0);
        sgLenSlider[i]->setMax(3.0);
        sgLenSlider[i]->setValue(this->Strands[i]->getLenFactor());
        sgLenSlider[i]->setSize(6, 2);
        sgColorButton[i] = new coTUIButton("color", frame->getID());
        sgColorButton[i]->setEventListener(this);
        sgColorButton[i]->setPos(6, y + i);
    }
}

void Strandgroup::delManipulation(void)
{
    int i;

    if (LenSliderL)
    {
        delete LenSliderL;
        LenSliderL = NULL;
    }
    if (LOTSliderL)
    {
        delete LOTSliderL;
        LOTSliderL = NULL;
    }
    if (colorL)
    {
        delete colorL;
        colorL = NULL;
    }
    if (myL1)
    {
        delete myL1;
        myL1 = NULL;
    }
    if (myLenSlider)
    {
        delete myLenSlider;
        myLenSlider = NULL;
    }
    if (myLOTSlider)
    {
        delete myLOTSlider;
        myLOTSlider = NULL;
    }
    if (sonL1)
    {
        delete sonL1;
        sonL1 = NULL;
    }
    if (sonL2)
    {
        delete sonL2;
        sonL2 = NULL;
    }
    if (sonLenSlider)
    {
        delete sonLenSlider;
        sonLenSlider = NULL;
    }
    if (sonColorButton)
    {
        delete sonColorButton;
        sonColorButton = NULL;
    }
    for (i = 0; i < this->numStrands; i++)
    {
        if (sgLenSliderL[i])
        {
            delete sgLenSliderL[i];
            sgLenSliderL[i] = NULL;
        }
        if (sgLenSlider[i])
        {
            delete sgLenSlider[i];
            sgLenSlider[i] = NULL;
        }
        if (sgColorButton[i])
        {
            delete sgColorButton[i];
            sgColorButton[i] = NULL;
        }
    }
}

void Strandgroup::tabletPressEvent(coTUIElement *tabUI)
{
    int i;

    fprintf(stderr, "Strandgroup::tabletPressEvent\n");
    if (tabUI == this->sonColorButton)
    {
        this->setColor(this->getColTr());
    }
    for (i = 0; i < this->numStrands && this->sgColorButton[i]; i++)
    {
        if (tabUI == this->sgColorButton[i])
        {
            this->Strands[i]->setColor(this->getColTr());
        }
    }
}

void Strandgroup::tabletEvent(coTUIElement *tabUI)
{
    int i;

    fprintf(stderr, "Strandgroup::tabletEvent\n");
    if (tabUI == this->myLenSlider)
    {
        this->setLenFactor(this->myLenSlider->getValue());
        this->createGeom();
    }
    else if (tabUI == this->sonLenSlider)
    {
        float val = this->sonLenSlider->getValue();
        for (i = 0; i < this->numStrands; i++)
        {
            this->Strands[i]->setLenFactor(val);
            this->sgLenSlider[i]->setValue(val);
        }
        this->createGeom();
    }
    else if (tabUI == this->myLOTSlider)
    {
        this->setLengthOfTwist(this->myLOTSlider->getValue());
        this->createGeom();
    }
    for (i = 0; i < this->numStrands; i++)
    {
        if (tabUI == this->sgLenSlider[i])
        {
            this->Strands[i]->setLenFactor(this->sgLenSlider[i]->getValue());
            this->Strands[i]->createGeom();
        }
    }
}

void Strandgroup::setColor(osg::Vec4 color)
{
    int i;

    for (i = 0; i < this->numStrands; i++)
        this->Strands[i]->setColor(color);
}

void Strandgroup::recurseSetLenFactor(float val)
{
    int i;

    this->setLenFactor(val);
    for (i = 0; i < this->numStrands; i++)
        this->Strands[i]->recurseSetLenFactor(val);
}
