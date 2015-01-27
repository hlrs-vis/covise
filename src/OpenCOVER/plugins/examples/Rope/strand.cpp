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
#include "Strand.h"
#include <osg/LineWidth>
#include <stdio.h>
using namespace osg;

Strand::Strand(rShared *daddy, int id, float R, float angle, coTUIFrame *frame, coTUIComboBox *box)
{
    int i;

    this->setLenFactor(1.0); // default
    this->numWiregroups = 0;
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
    for (i = 0; i < this->maxWiregroups; i++)
    {
        this->sgLenSliderL[i] = NULL;
        this->sgLenSlider[i] = NULL;
        this->sgLOTSlider[i] = NULL;
        this->sgColorButton[i] = NULL;
    }

    this->initShared(daddy, (char *)"Strand", id, frame, box);
    this->setPosRadius(R);
    this->setPosAngle(angle);
}

//Strand Destructor delets all functions and selections, which have been made
Strand::~Strand()
{
    int i;

    this->delManipulation();
    for (i = 0; i < this->numWiregroups; i++)
        delete this->Wiregroups[i];
}

void Strand::createGeom()
{
    int i;

    fprintf(stderr, " Strand::createGeom: %s\n", this->getName());
    for (i = 0; i < this->numWiregroups; i++)
    {
        this->Wiregroups[i]->createGeom();
    }
}

int Strand::addWiregroup(float lot, coTUIFrame *frame, coTUIComboBox *box)
{
    if (this->numWiregroups < this->maxWiregroups)
    {
        this->Wiregroups[this->numWiregroups] = new Wiregroup(this, this->numWiregroups, lot, frame, box);
        return this->numWiregroups++;
    }
    return -1;
}

xercesc::DOMElement *Strand::Save(xercesc::DOMDocument &document)
{
    char tmp[256];
    int i;

    xercesc::DOMElement *element = document.createElement(xercesc::XMLString::transcode(this->getName()));

    element->appendChild(document.createTextNode(xercesc::XMLString::transcode(this->identStr())));
    element->setAttribute(xercesc::XMLString::transcode("depth"), xercesc::XMLString::transcode(this->getName()));
    sprintf(tmp, "%f", this->getPosRadius());
    element->setAttribute(xercesc::XMLString::transcode("posRadius"), xercesc::XMLString::transcode(tmp));
    sprintf(tmp, "%f", this->rad2grad(this->getPosAngle()));
    element->setAttribute(xercesc::XMLString::transcode("posAngle"), xercesc::XMLString::transcode(tmp));
    for (i = 0; i < this->numWiregroups; i++)
    {
        fprintf(stderr, "call wiregroup[%d]->Save()\n", i);
        element->appendChild(this->Wiregroups[i]->Save(document));
        element->appendChild(document.createTextNode(xercesc::XMLString::transcode(this->identStr())));
    }

    return element;
}

void Strand::Load(xercesc::DOMElement *rNode)
{
    int i;
    char *p;
    float lot;
    int len;
    int wgID;

    xercesc::DOMNodeList *nodeList = rNode->getChildNodes();
    len = nodeList->getLength();
    fprintf(stderr, "      Strand::Load : NumberOfNodes=%d\n", len);
    for (i = 0; i < len; ++i)
    {
        fprintf(stderr, "      Strand::Load : nodeNo=%d\n", i);

        xercesc::DOMElement *node = dynamic_cast<xercesc::DOMElement *>(nodeList->item(i));
        if (node)
        {
            lot = 2000.0; // default ;(
            if ((p = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("LengthOfTwist")))) != NULL)
                lot = atof(p);
            wgID = this->addWiregroup(lot, this->coverFrame, this->coverBox);
            this->Wiregroups[wgID]->Load(node);
        }
    }
}

void Strand::addManipulation(coTUIFrame *frame)
{
    int i;
    int y = 0;

    // my ... ich ;)
    // son ... nur die Ebene drunter
    // all ... alles drunter

    // Kopfzeile pinseln ...
    y = 1;
    fprintf(stderr, "Strand::addManipulation\n");
    LenSliderL = new coTUILabel("Length factor", frame->getID());
    LenSliderL->setPos(2, y);
    LOTSliderL = new coTUILabel("Length of twist", frame->getID());
    LOTSliderL->setPos(4, y);
    colorL = new coTUILabel("Color", frame->getID());
    colorL->setPos(6, y);

    // Erste Zeile ...
    y++;
    myL1 = new coTUILabel("Strand", frame->getID());
    myL1->setPos(0, y);
    myLenSlider = new coTUIFloatSlider("numHSlider", frame->getID());
    myLenSlider->setEventListener(this);
    myLenSlider->setPos(2, y);
    myLenSlider->setMin(0.0);
    myLenSlider->setMax(3.0);
    myLenSlider->setValue(this->getLenFactor());
    myLenSlider->setSize(6, 2);

    y++;
    sonL1 = new coTUILabel("Wiregroup", frame->getID());
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
    // basteln wir den "alles"-mach-Knopf
    for (i = 0; i < this->numWiregroups; i++)
    {
        if (!this->Wiregroups[i]->isCore())
        {
            sonLOTSlider = new coTUIFloatSlider("numHSlider", frame->getID());
            sonLOTSlider->setEventListener(this);
            sonLOTSlider->setPos(4, y);
            sonLOTSlider->setMin(-2000);
            sonLOTSlider->setMax(2000);
            sonLOTSlider->setValue(this->Wiregroups[i]->getLengthOfTwist());
            sonLOTSlider->setSize(5, 0);
            break;
        }
    }
    sonColorButton = new coTUIButton("color", frame->getID());
    sonColorButton->setEventListener(this);
    sonColorButton->setPos(6, y);

    for (i = 0, y++; i < this->numWiregroups; i++)
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
        sgLenSlider[i]->setValue(this->Wiregroups[i]->getLenFactor());
        sgLenSlider[i]->setSize(6, 2);
        if (this->Wiregroups[i]->isCore() == false)
        { // Nur beim "nicht-Kern"
            sgLOTSlider[i] = new coTUIFloatSlider("numHSlider", frame->getID());
            sgLOTSlider[i]->setEventListener(this);
            sgLOTSlider[i]->setPos(4, y + i);
            sgLOTSlider[i]->setMin(-2000);
            sgLOTSlider[i]->setMax(+2000);
            sgLOTSlider[i]->setValue(this->Wiregroups[i]->getLengthOfTwist());
            sgLOTSlider[i]->setSize(5, 0);
        }
        sgColorButton[i] = new coTUIButton("color", frame->getID());
        sgColorButton[i]->setEventListener(this);
        sgColorButton[i]->setPos(6, y + i);
    }
}

void Strand::delManipulation(void)
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
    for (i = 0; i < this->numWiregroups; i++)
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

void Strand::tabletPressEvent(coTUIElement *tabUI)
{
    int i;

    fprintf(stderr, "Strand::tabletPressEvent\n");
    if (tabUI == this->sonColorButton)
    {
        this->setColor(this->getColTr());
    }
    for (i = 0; i < this->numWiregroups && this->sgColorButton[i]; i++)
    {
        if (tabUI == this->sgColorButton[i])
        {
            this->Wiregroups[i]->setColor(this->getColTr());
        }
    }
}

void Strand::tabletEvent(coTUIElement *tabUI)
{
    int i;

    fprintf(stderr, "Strand::tabletEvent\n");
    if (tabUI == this->myLenSlider)
    {
        this->setLenFactor(this->myLenSlider->getValue());
        this->createGeom();
    }
    else if (tabUI == this->sonLenSlider)
    {
        float val = this->sonLenSlider->getValue();
        for (i = 0; i < this->numWiregroups; i++)
        {
            this->Wiregroups[i]->setLenFactor(val);
            this->sgLenSlider[i]->setValue(val);
        }
        this->createGeom();
    }
    else if (tabUI == this->sonLOTSlider)
    {
        int i;
        float val = this->sonLOTSlider->getValue();

        for (i = 0; i < this->numWiregroups; i++)
        {
            if (!this->Wiregroups[i]->isCore())
            {
                this->Wiregroups[i]->setLengthOfTwist(val);
                this->sgLOTSlider[i]->setValue(val);
                this->Wiregroups[i]->createGeom();
            }
        }
    }
    for (i = 0; i < this->numWiregroups; i++)
    {
        if (tabUI == this->sgLenSlider[i])
        {
            this->Wiregroups[i]->setLenFactor(this->sgLenSlider[i]->getValue());
            this->Wiregroups[i]->createGeom();
        }
    }
    for (i = 0; i < this->numWiregroups; i++)
    {
        if (tabUI == this->sgLOTSlider[i])
        {
            this->Wiregroups[i]->setLengthOfTwist((float)(this->sgLOTSlider[i]->getValue()));
            this->Wiregroups[i]->createGeom();
        }
    }
}

void Strand::setColor(osg::Vec4 color)
{
    int i;

    for (i = 0; i < this->numWiregroups; i++)
        this->Wiregroups[i]->setColor(color);
}

void Strand::recurseSetLenFactor(float val)
{
    int i;

    this->setLenFactor(val);
    for (i = 0; i < this->numWiregroups; i++)
        this->Wiregroups[i]->recurseSetLenFactor(val);
}
