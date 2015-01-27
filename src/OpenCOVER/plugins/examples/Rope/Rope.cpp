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
#include "Rope.h"
#include <osg/LineWidth>
#include <stdio.h>
using namespace osg;

Rope::Rope(int id, coTUIFrame *frame, coTUIComboBox *box)
{
    int i;

    this->numStrandgroups = 0;
    this->ropeL11 = NULL;
    this->ropeL12 = NULL;
    this->ropeL13 = NULL;
    this->ropeL14 = NULL;
    this->ropeL15 = NULL;
    this->ropeL16 = NULL;
    this->ropeL17 = NULL;
    this->ropeL18 = NULL;
    this->ropeL19 = NULL;
    this->ropePosR = NULL;
    this->ropePosA = NULL;
    this->ropeNumS = NULL;
    this->ropeSHeight = NULL;
    this->ropeOx = NULL;
    this->ropeOy = NULL;
    this->ropeOz = NULL;
    this->LenSliderL = NULL;
    this->LOTSliderL = NULL;
    this->colorL = NULL;
    this->myL1 = NULL;
    this->sonL1 = NULL;
    this->sonL2 = NULL;
    this->myLenSlider = NULL;
    this->sonLenSlider = NULL;
    this->sonLOTSlider = NULL;
    this->sonColorButton = NULL;
    for (i = 0; i < this->maxStrandgroups; i++)
    {
        this->sgLenSliderL[i] = NULL;
        this->sgLenSlider[i] = NULL;
        this->sgLOTSlider[i] = NULL;
        this->sgColorButton[i] = NULL;
    }
    this->initShared(NULL, (char *)"Rope", id, frame, box);
    this->setLengthOfTwist(0.0);
    this->setPosAngle(0.0);
    this->setPosRadius(0.0);
    this->setOrientation(osg::Vec3(0, 0, 1));
    this->setSegHeight(10);
    this->setNumSegments(8);
}

Rope::~Rope(void)
{
    int i;

    this->delManipulation();
    for (i = 0; i < this->numStrandgroups; i++)
        delete this->Strandgroups[i];
}

//Constructor of Rope
bool Rope::init()
{
    return true;
}

void Rope::createGeom()
{
    int i;

    fprintf(stderr, "Rope::createGeom: %s\n", this->getName());
    for (i = 0; i < this->numStrandgroups; i++)
        this->Strandgroups[i]->createGeom();
}

void Rope::AlbertRope(void)
{
    int sgID, sID, wgID, i, j;
    int numW = 4;
    int numS = 3;
    float R = 3.5;
    float lotW = 142;
    float posRw = sqrt(2.0 * 2.0 * R * 2.0 * R) / 2.0;
    float R2 = posRw + R;
    float lotS = 142;
    float posRs = sqrt(2.0 * R2 * 2.0 * R2 - R2 * R2) / 3.0 * 2.0 - 0.5;

    fprintf(stderr, "Rope::AlbertRope(R=%f, posRw=%f, posRs=%f\n", R, posRw, posRs);
    this->setRopeLength(1000.0);

    sgID = this->addStrandgroup(lotW, this->coverFrame, this->coverBox);
    for (i = 0; i < numS; i++)
    {
        sID = this->Strandgroups[sgID]->addStrand(posRs, 2.0 * M_PI / (float)numS * (float)i, this->coverFrame, this->coverBox);
        // outer strands - core wire
        wgID = this->Strandgroups[sgID]->Strands[sID]->addWiregroup(lotS, this->coverFrame, this->coverBox);
        for (j = 0; j < numW; j++)
            this->Strandgroups[sgID]->Strands[sID]->Wiregroups[wgID]->addWire(posRw, (float)j * (2.0 * M_PI / (float)numW), R, this->coverFrame, this->coverBox);
    }
}

void Rope::TestRope1(void)
{
    int sgID, sID, wgID, i, j;

    fprintf(stderr, "Rope::TestRope1");
    // das Testseil von Uwe ... mit zu kleinen Draehten ...
    this->setRopeLength(200.0);

    // core strand
    sgID = this->addStrandgroup(0.0, this->coverFrame, this->coverBox);
    sID = this->Strandgroups[sgID]->addStrand(0, 0.0, this->coverFrame, this->coverBox);
    // core strand - core wire
    wgID = this->Strandgroups[sgID]->Strands[sID]->addWiregroup(0.0, this->coverFrame, this->coverBox);
    this->Strandgroups[sgID]->Strands[sID]->Wiregroups[wgID]->addWire(0, 0, 6, this->coverFrame, this->coverBox);
    // core strand - outer wires
    wgID = this->Strandgroups[0]->Strands[0]->addWiregroup(100, this->coverFrame, this->coverBox);
    for (i = 0; i < 6; i++)
        this->Strandgroups[sgID]->Strands[sID]->Wiregroups[wgID]->addWire(20, (float)i * (2.0 * M_PI / 6.0), 6, this->coverFrame, this->coverBox);

    // outer strands
    sgID = this->addStrandgroup(600, this->coverFrame, this->coverBox);
    for (i = 0; i < 6; i++)
    {
        sID = this->Strandgroups[sgID]->addStrand(60, 2.0 * M_PI / 6.0 * (float)i, this->coverFrame, this->coverBox);
        // outer strands - core wire
        wgID = this->Strandgroups[sgID]->Strands[sID]->addWiregroup(0.0, this->coverFrame, this->coverBox);
        this->Strandgroups[sgID]->Strands[sID]->Wiregroups[wgID]->addWire(0, 0, 6, this->coverFrame, this->coverBox);
        wgID = this->Strandgroups[sgID]->Strands[sID]->addWiregroup(100, this->coverFrame, this->coverBox);
        for (j = 0; j < 6; j++)
            this->Strandgroups[sgID]->Strands[sID]->Wiregroups[wgID]->addWire(20, (float)j * (2.0 * M_PI / 6.0), 6, this->coverFrame, this->coverBox);
    }
}

int Rope::addStrandgroup(float lot, coTUIFrame *frame, coTUIComboBox *box)
{
    if (this->numStrandgroups < this->maxStrandgroups)
    {
        this->Strandgroups[this->numStrandgroups] = new Strandgroup(this, this->numStrandgroups, lot, frame, box);
        return this->numStrandgroups++;
    }
    return -1;
}

xercesc::DOMElement *Rope::Save(xercesc::DOMDocument &document)
{
    char tmp[256];
    int i;
    osg::Vec3 orient;

    xercesc::DOMElement *element = document.createElement(xercesc::XMLString::transcode(this->getName()));
    element->appendChild(document.createTextNode(xercesc::XMLString::transcode(this->identStr())));
    element->setAttribute(xercesc::XMLString::transcode("depth"), xercesc::XMLString::transcode(this->getName()));
    sprintf(tmp, "%f", this->getRopeLength());
    element->setAttribute(xercesc::XMLString::transcode("length"), xercesc::XMLString::transcode(tmp));
    sprintf(tmp, "%f", this->getPosRadius());
    element->setAttribute(xercesc::XMLString::transcode("posRadius"), xercesc::XMLString::transcode(tmp));
    sprintf(tmp, "%f", this->getPosAngle());
    element->setAttribute(xercesc::XMLString::transcode("posAngle"), xercesc::XMLString::transcode(tmp));
    orient = this->getOrientation();
    sprintf(tmp, "%f,%f,%f", orient.x(), orient.y(), orient.z());
    element->setAttribute(xercesc::XMLString::transcode("orientation"), xercesc::XMLString::transcode(tmp));
    sprintf(tmp, "%d", this->getNumSegments());
    element->setAttribute(xercesc::XMLString::transcode("numSegments"), xercesc::XMLString::transcode(tmp));
    sprintf(tmp, "%f", this->getSegHeight());
    element->setAttribute(xercesc::XMLString::transcode("segHeight"), xercesc::XMLString::transcode(tmp));

    for (i = 0; i < this->numStrandgroups; i++)
    {
        element->appendChild(this->Strandgroups[i]->Save(document));
        element->appendChild(document.createTextNode(xercesc::XMLString::transcode(this->identStr())));
    }
    return element;
}

void Rope::Load(xercesc::DOMElement *rNode)
{
    int i;
    char *p;
    float lot;
    int len;
    int sgID;
    float x, y, z;

    if (rNode == NULL)
    {
        this->TestRope1();
        return;
    }
    xercesc::DOMNodeList *nodeList = rNode->getChildNodes();
    len = nodeList->getLength();
    fprintf(stderr, "  Rope::Load : NumberOfNodes=%d\n", len);
    if ((p = xercesc::XMLString::transcode(rNode->getAttribute(xercesc::XMLString::transcode("length")))) != NULL)
        this->setRopeLength(atof(p));
    if ((p = xercesc::XMLString::transcode(rNode->getAttribute(xercesc::XMLString::transcode("posRadius")))) != NULL)
        this->setPosRadius(atof(p));
    if ((p = xercesc::XMLString::transcode(rNode->getAttribute(xercesc::XMLString::transcode("orientation")))) != NULL)
    {
        if (3 == sscanf(p, "%f,%f,%f", &x, &y, &z))
        {
            if (x != 0.0 || y != 0.0 || z != 0.0)
            {
                this->setOrientation(osg::Vec3(x, y, z));
            }
        }
    }
    if ((p = xercesc::XMLString::transcode(rNode->getAttribute(xercesc::XMLString::transcode("numSegments")))) != NULL)
        this->setNumSegments(atoi(p));
    if ((p = xercesc::XMLString::transcode(rNode->getAttribute(xercesc::XMLString::transcode("segHeight")))) != NULL)
        this->setSegHeight(atof(p));

    for (i = 0; i < len; ++i)
    {
        fprintf(stderr, "  Rope::Load : nodeNo=%d\n", i);

        xercesc::DOMElement *node = dynamic_cast<xercesc::DOMElement *>(nodeList->item(i));
        if (node)
        {
            lot = 1.0; // bad default
            if ((p = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("LengthOfTwist")))) != NULL)
                lot = atof(p);
            sgID = this->addStrandgroup(lot, this->coverFrame, this->coverBox);
            this->Strandgroups[sgID]->Load(node);
        }
        else
            fprintf(stderr, "  Rope::Load : nodeNo=%d NOT FOUND\n", i);
    }
}

void Rope::addManipulation(coTUIFrame *frame)
{
    int i;
    int y = 0;

    // my ... ich ;)
    // son ... nur die Ebene drunter
    // all ... alles drunter

    // x, y, z, numS, segheight
    // posR, posAngle, LOT
    // Kopfzeile pinseln ...
    y = 1;
    ropeL13 = new coTUILabel("posR", frame->getID());
    ropeL13->setPos(2, y);
    ropeL14 = new coTUILabel("posA", frame->getID());
    ropeL14->setPos(3, y);
    ropeL15 = new coTUILabel("numS", frame->getID());
    ropeL15->setPos(4, y);
    ropeL16 = new coTUILabel("sHeight", frame->getID());
    ropeL16->setPos(5, y);
    //ropeL17 = new coTUILabel("x", frame->getID());
    //ropeL17->setPos(6,y);
    //ropeL18 = new coTUILabel("y", frame->getID());
    //ropeL18->setPos(7,y);
    //ropeL19 = new coTUILabel("z", frame->getID());
    //ropeL19->setPos(8,y);

    y++;
    ropeL11 = new coTUILabel("Rope", frame->getID());
    ropeL11->setPos(0, y);
    ropeL12 = new coTUILabel("global", frame->getID());
    ropeL12->setPos(1, y);
    ropePosR = new coTUIEditFloatField("a", frame->getID(), 0);
    ropePosR->setPos(2, y);
    ropePosR->setEventListener(this);
    ropePosR->setValue(this->getPosRadius());
    ropePosR->setSize(7, 1);
    ropePosA = new coTUIEditFloatField("a", frame->getID(), 0);
    ropePosA->setPos(3, y);
    ropePosA->setEventListener(this);
    ropePosA->setValue(this->rad2grad(this->getPosAngle()));
    ropePosA->setSize(7, 1);
    ropeNumS = new coTUIEditIntField("a", frame->getID(), 0);
    ropeNumS->setPos(4, y);
    ropeNumS->setEventListener(this);
    ropeNumS->setValue(this->getNumSegments());
    ropeNumS->setSize(4, 0);
    ropeSHeight = new coTUIEditFloatField("a", frame->getID(), 0);
    ropeSHeight->setPos(5, y);
    ropeSHeight->setEventListener(this);
    ropeSHeight->setValue(this->getSegHeight());
    ropeSHeight->setSize(3, 0);
    //osg::Vec3 orient = this->getOrientation();
    //ropeOx = new coTUIEditFloatField("a", frame->getID(), 0);
    //ropeOx->setPos(6, y);
    //ropeOx->setEventListener(this);
    //ropeOx->setValue(orient.x());
    //ropeOx->setSize(4, 1);
    //ropeOy = new coTUIEditFloatField("a", frame->getID(), 0);
    //ropeOy->setPos(7, y);
    //ropeOy->setEventListener(this);
    //ropeOy->setValue(orient.y());
    //ropeOy->setSize(4, 1);
    //ropeOz = new coTUIEditFloatField("a", frame->getID(), 0);
    //ropeOz->setPos(8, y);
    //ropeOz->setEventListener(this);
    //ropeOz->setValue(orient.z());
    //ropeOz->setSize(4, 1);

    y++;
    y++;
    fprintf(stderr, "Rope::addManipulation\n");
    LenSliderL = new coTUILabel("Length factor", frame->getID());
    LenSliderL->setPos(2, y);
    LOTSliderL = new coTUILabel("Length of twist", frame->getID());
    LOTSliderL->setPos(4, y);
    colorL = new coTUILabel("Color", frame->getID());
    colorL->setPos(6, y);

    // Erste Zeile ...
    y++;
    myL1 = new coTUILabel("Rope", frame->getID());
    myL1->setPos(0, y);
    myLenSlider = new coTUIFloatSlider("numHSlider", frame->getID());
    myLenSlider->setEventListener(this);
    myLenSlider->setPos(2, y);
    myLenSlider->setMin(0.0);
    myLenSlider->setMax(3.0);
    myLenSlider->setValue(this->getLenFactor());
    myLenSlider->setSize(6, 2);

    y++;
    sonL1 = new coTUILabel("Strandgroup", frame->getID());
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
    // nur falls wir irgendwo in der Gruppe mindestens eine Schlaglaenge haben
    // basteln wir den "Schlaglitze-alles-setz"-mach-Knopf
    for (i = 0; i < this->numStrandgroups; i++)
    {
        if (!this->Strandgroups[i]->isCore())
        {
            sonLOTSlider = new coTUIFloatSlider("numHSlider", frame->getID());
            sonLOTSlider->setEventListener(this);
            sonLOTSlider->setPos(4, y);
            sonLOTSlider->setMin(-2000);
            sonLOTSlider->setMax(2000);
            sonLOTSlider->setValue(this->Strandgroups[i]->getLengthOfTwist());
            sonLOTSlider->setSize(5, 0);
            break;
        }
    }
    sonColorButton = new coTUIButton("color", frame->getID());
    sonColorButton->setEventListener(this);
    sonColorButton->setPos(6, y);

    for (i = 0, y++; i < this->numStrandgroups; i++)
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
        sgLenSlider[i]->setValue(this->Strandgroups[i]->getLenFactor());
        sgLenSlider[i]->setSize(6, 2);
        if (this->Strandgroups[i]->isCore() == false)
        { // Nur beim "nicht-Kern"
            sgLOTSlider[i] = new coTUIFloatSlider("numHSlider", frame->getID());
            sgLOTSlider[i]->setEventListener(this);
            sgLOTSlider[i]->setPos(4, y + i);
            sgLOTSlider[i]->setMin(-2000);
            sgLOTSlider[i]->setMax(+2000);
            sgLOTSlider[i]->setValue(this->Strandgroups[i]->getLengthOfTwist());
            sgLOTSlider[i]->setSize(5, 0);
        }
        sgColorButton[i] = new coTUIButton("color", frame->getID());
        sgColorButton[i]->setEventListener(this);
        sgColorButton[i]->setPos(6, y + i);
    }
}

void Rope::delManipulation(void)
{
    int i;

    if (this->ropeL11)
    {
        delete this->ropeL11;
        this->ropeL11 = NULL;
    }
    if (this->ropeL12)
    {
        delete this->ropeL12;
        this->ropeL12 = NULL;
    }
    if (this->ropeL13)
    {
        delete this->ropeL13;
        this->ropeL13 = NULL;
    }
    if (this->ropeL14)
    {
        delete this->ropeL14;
        this->ropeL14 = NULL;
    }
    if (this->ropeL15)
    {
        delete this->ropeL15;
        this->ropeL15 = NULL;
    }
    if (this->ropeL16)
    {
        delete this->ropeL16;
        this->ropeL16 = NULL;
    }
    if (this->ropeL17)
    {
        delete this->ropeL17;
        this->ropeL17 = NULL;
    }
    if (this->ropeL18)
    {
        delete this->ropeL18;
        this->ropeL18 = NULL;
    }
    if (this->ropeL19)
    {
        delete this->ropeL19;
        this->ropeL19 = NULL;
    }
    if (this->ropePosR)
    {
        delete this->ropePosR;
        this->ropePosR = NULL;
    }
    if (this->ropePosA)
    {
        delete this->ropePosA;
        this->ropePosA = NULL;
    }
    if (this->ropeNumS)
    {
        delete this->ropeNumS;
        this->ropeNumS = NULL;
    }
    if (this->ropeSHeight)
    {
        delete this->ropeSHeight;
        this->ropeSHeight = NULL;
    }
    if (this->ropeOx)
    {
        delete this->ropeOx;
        this->ropeOx = NULL;
    }
    if (this->ropeOy)
    {
        delete this->ropeOy;
        this->ropeOy = NULL;
    }
    if (this->ropeOz)
    {
        delete this->ropeOz;
        this->ropeOz = NULL;
    }
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
    if (sonLOTSlider)
    {
        delete sonLOTSlider;
        sonLOTSlider = NULL;
    }
    if (sonColorButton)
    {
        delete sonColorButton;
        sonColorButton = NULL;
    }
    for (i = 0; i < this->numStrandgroups; i++)
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
        if (sgLOTSlider[i])
        {
            delete sgLOTSlider[i];
            sgLOTSlider[i] = NULL;
        }
        if (sgColorButton[i])
        {
            delete sgColorButton[i];
            sgColorButton[i] = NULL;
        }
    }
}

void Rope::tabletPressEvent(coTUIElement *tabUI)
{
    int i;

    fprintf(stderr, "Rope::tabletPressEvent\n");
    if (tabUI == this->sonColorButton)
    {
        this->setColor(this->getColTr());
    }
    for (i = 0; i < this->numStrandgroups && this->sgColorButton[i]; i++)
    {
        if (tabUI == this->sgColorButton[i])
        {
            this->Strandgroups[i]->setColor(this->getColTr());
        }
    }
}

void Rope::tabletEvent(coTUIElement *tabUI)
{
    int i;

    fprintf(stderr, "Rope::tabletEvent\n");
    if (tabUI == this->myLenSlider)
    {
        this->setLenFactor(this->myLenSlider->getValue());
        this->createGeom();
    }
    else if (tabUI == this->sonLenSlider)
    {
        float val = this->sonLenSlider->getValue();
        for (i = 0; i < this->numStrandgroups; i++)
        {
            this->Strandgroups[i]->setLenFactor(val);
            this->sgLenSlider[i]->setValue(val);
        }
        this->createGeom();
    }
    else if (tabUI == this->sonLOTSlider)
    {
        int i;
        float val = this->sonLOTSlider->getValue();

        for (i = 0; i < this->numStrandgroups; i++)
        {
            if (!this->Strandgroups[i]->isCore())
            {
                this->Strandgroups[i]->setLengthOfTwist(val);
                this->sgLOTSlider[i]->setValue(val);
                this->Strandgroups[i]->createGeom();
            }
        }
    }
    for (i = 0; i < this->numStrandgroups; i++)
    {
        if (tabUI == this->sgLenSlider[i])
        {
            this->Strandgroups[i]->setLenFactor(this->sgLenSlider[i]->getValue());
            this->Strandgroups[i]->createGeom();
        }
    }
    for (i = 0; i < this->numStrandgroups; i++)
    {
        if (tabUI == this->sgLOTSlider[i])
        {
            this->Strandgroups[i]->setLengthOfTwist((float)(this->sgLOTSlider[i]->getValue()));
            this->Strandgroups[i]->createGeom();
        }
    }
    if (tabUI == this->ropePosR)
    {
        this->setPosRadius(this->ropePosR->getValue());
        this->ropePosR->setValue(this->getPosRadius());
        this->createGeom();
    }
    if (tabUI == this->ropePosA)
    {
        this->setPosAngle(this->grad2rad(this->ropePosA->getValue()));
        this->ropePosA->setValue(this->rad2grad(this->getPosAngle()));
        this->createGeom();
    }
    if (tabUI == this->ropeNumS)
    {
        this->setNumSegments(this->ropeNumS->getValue());
        this->ropeNumS->setValue(this->getNumSegments());
        this->createGeom();
    }
    if (tabUI == this->ropeSHeight)
    {
        this->setSegHeight(this->ropeSHeight->getValue());
        this->ropeSHeight->setValue(this->getSegHeight());
        this->createGeom();
    }
    if (tabUI == this->ropeOx || tabUI == this->ropeOy || tabUI == this->ropeOz)
    {
        osg::Vec3 orient(ropeOx->getValue(), ropeOy->getValue(), ropeOz->getValue());
        this->setOrientation(orient);
        orient = this->getOrientation();
        this->ropeOx->setValue(orient.x());
        this->ropeOy->setValue(orient.y());
        this->ropeOz->setValue(orient.z());
        this->createGeom();
    }
}

void Rope::setColor(osg::Vec4 color)
{
    int i;

    for (i = 0; i < this->numStrandgroups; i++)
        this->Strandgroups[i]->setColor(color);
}

void Rope::recurseSetLenFactor(float val)
{
    int i;

    this->setLenFactor(val);
    for (i = 0; i < this->numStrandgroups; i++)
        this->Strandgroups[i]->recurseSetLenFactor(val);
}
