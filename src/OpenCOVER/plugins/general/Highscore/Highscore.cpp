/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeTimesteps.cpp
#ifdef _WIN32
#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#include <winsock2.h>
#include <windows.h>
#endif
#include <util/common.h>
#include <vrml97/vrml/config.h>
#include <vrml97/vrml/VrmlNodeType.h>
#include <vrml97/vrml/coEventQueue.h>

#include <vrml97/vrml/MathUtils.h>
#include <vrml97/vrml/System.h>
#include <vrml97/vrml/Viewer.h>
#include <vrml97/vrml/VrmlScene.h>
#include <vrml97/vrml/VrmlSFTime.h>
#include <vrml97/vrml/VrmlNamespace.h>
#include <cover/VRViewer.h>
#include <cover/coVRTui.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRAnimationManager.h>
#include <cover/coVRPluginSupport.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <math.h>

#include <util/byteswap.h>
#include <xercesc/dom/DOM.hpp>
#if XERCES_VERSION_MAJOR < 3
#include <xercesc/dom/DOMWriter.hpp>
#else
#include <xercesc/dom/DOMLSSerializer.hpp>
#endif
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/XMLUni.hpp>

#include "Highscore.h"

void HSEntry::setStartTime(double t)
{
    startTime = t;
}

void HSEntry::setInterimTime(double t)
{
    interimTime = t;
    char number[100];
    sprintf(number, "%03.3f", getInterimTimeDiff());
    InterimLabel->setLabel(number);
}

void HSEntry::setEndTime(double t)
{
    endTime = t;
    char number[100];
    sprintf(number, "%03.3f", getLapTime());
    LapLabel->setLabel(number);
}

double HSEntry::getLapTime()
{
    return endTime - startTime;
}

double HSEntry::getInterimTimeDiff()
{
    return interimTime - startTime;
}

void HSEntry::setName(std::string &n)
{
    name = n;
    NameLabel->setLabel(n.c_str());
}

void HSEntry::setPos(int p)
{
    pos = p;
    char number[100];
    sprintf(number, "%d", pos);
    PosLabel->setLabel(number);
    PosLabel->setPos((pos / 20) * 4 + 0, (pos % 20));
    NameLabel->setPos((pos / 20) * 4 + 1, (pos % 20));
    LapLabel->setPos((pos / 20) * 4 + 2, (pos % 20));
    InterimLabel->setPos((pos / 20) * 4 + 3, (pos % 20));
}

void HSEntry::reset()
{
    startTime = 0;
    endTime = 0;
    interimTime = 0;
}

HSEntry::HSEntry(Highscore *h)
{
    startTime = 0;
    endTime = 0;
    interimTime = 0;
    hs = h;
    PosLabel = new coTUILabel("X", hs->HighscoreTab->getID());
    NameLabel = new coTUILabel("X", hs->HighscoreTab->getID());
    LapLabel = new coTUILabel("X", hs->HighscoreTab->getID());
    InterimLabel = new coTUILabel("X", hs->HighscoreTab->getID());
}

HSEntry::~HSEntry()
{
    delete PosLabel;
    delete NameLabel;
    delete LapLabel;
    delete InterimLabel;
}

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeHighscore(scene);
}

VrmlNodeType *VrmlNodeHighscore::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("Highscore", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class

    t->addEventIn("startTime", VrmlField::SFTIME);
    t->addEventIn("interimTime", VrmlField::SFTIME);
    t->addEventIn("reset", VrmlField::SFTIME);

    return t;
}

VrmlNodeType *VrmlNodeHighscore::nodeType() const
{
    return defineType(0);
}

VrmlNodeHighscore::VrmlNodeHighscore(VrmlScene *scene)
    : VrmlNodeChild(scene)
{
    setModified();
}

void VrmlNodeHighscore::addToScene(VrmlScene *, const char * /*relUrl*/)
{
}

// need copy constructor for new markerName (each instance definitely needs a new marker Name) ...

VrmlNodeHighscore::VrmlNodeHighscore(const VrmlNodeHighscore &n)
    : VrmlNodeChild(n.d_scene)
{
}

VrmlNodeHighscore::~VrmlNodeHighscore()
{
}

VrmlNode *VrmlNodeHighscore::cloneMe() const
{
    return new VrmlNodeHighscore(*this);
}

VrmlNodeHighscore *VrmlNodeHighscore::toHighscore() const
{
    return (VrmlNodeHighscore *)this;
}

// Set the value of one of the node fields.

void VrmlNodeHighscore::setField(const char *fieldName,
                                 const VrmlField &fieldValue)
{
    if (strcmp(fieldName, "startTime") == 0)
    {
        Highscore::instance()->setStartTime(fieldValue.toSFTime()->get());
    }
    if (strcmp(fieldName, "interimTime") == 0)
    {
        Highscore::instance()->setInterimTime(fieldValue.toSFTime()->get());
    }
    if (strcmp(fieldName, "reset") == 0)
    {
        Highscore::instance()->setResetTime(fieldValue.toSFTime()->get());
    }
}

void VrmlNodeHighscore::eventIn(double timeStamp,
                                const char *eventName,
                                const VrmlField *fieldValue)
{
    if (strcmp(eventName, "startTime") == 0)
    {
        Highscore::instance()->setStartTime(fieldValue->toSFTime()->get());
    }
    else if (strcmp(eventName, "interimTime") == 0)
    {
        Highscore::instance()->setInterimTime(fieldValue->toSFTime()->get());
    }
    else if (strcmp(eventName, "reset") == 0)
    {
        Highscore::instance()->setResetTime(fieldValue->toSFTime()->get());
    }
    // Check exposedFields
    else
    {
        VrmlNode::eventIn(timeStamp, eventName, fieldValue);
    }
}

const VrmlField *VrmlNodeHighscore::getField(const char *fieldName) const
{
    cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
}

Highscore *Highscore::myInstance = NULL;

void Highscore::tabletPressEvent(coTUIElement * /* tUIItem */)
{
    // if(tUIItem == updateNow)
    {
        // doUpdate = true;
    }
}

void Highscore::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == DriverName)
    {
        std::string name;
        if (DriverName->getText() != "")
            name = DriverName->getText();
        currentEntry->setName(name);
    }
}

Highscore::Highscore()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    myInstance = this;
}

bool Highscore::init()
{
	XMLCh *t1 = NULL;
    impl = xercesc::DOMImplementationRegistry::getDOMImplementation(xercesc::XMLString::transcode("Core")); xercesc::XMLString::release(&t1);
    fprintf(stderr, "Highscore::Highscore\n");
    HighscoreTab = new coTUITab("Highscore", coVRTui::instance()->mainFolder->getID());
    HighscoreTab->setPos(0, 0);
    HSL = new coTUILabel("Best drivers:", HighscoreTab->getID());
    HSL->setPos(0, 0);
    DL = new coTUILabel("CurrentDriver:", HighscoreTab->getID());
    DL->setPos(0, 22);
    DriverName = new coTUIEditField("NoName", HighscoreTab->getID());
    DriverName->setPos(1, 22);
    DriverName->setEventListener(this);
    VrmlNamespace::addBuiltIn(VrmlNodeHighscore::defineType());
    passedInterim = false;
    currentEntry = new HSEntry(this);
    load();

    return true;
}

// this is called if the plugin is removed at runtime
Highscore::~Highscore()
{
    fprintf(stderr, "Highscore::~Highscore\n");

    while (hsEntries.size() > 0)
    {
        delete hsEntries.back();
        hsEntries.pop_back();
    }
    delete HSL;
    delete HighscoreTab;
    delete DL;
    delete DriverName;
    myInstance = NULL;

    delete impl;
}

void
Highscore::setStartTime(double t)
{
    if (passedInterim) // we finised a lap
    {
        currentEntry->setEndTime(t);
        std::list<HSEntry *>::iterator hs;
        for (hs = hsEntries.begin(); hs != hsEntries.end(); ++hs)
        {
            if ((*hs)->getLapTime() > currentEntry->getLapTime())
            {
                hsEntries.insert(hs, currentEntry);
                currentEntry = new HSEntry(this);
                std::string name;
                if (DriverName->getText() != "")
                    name = DriverName->getText();
                currentEntry->setName(name);
                break;
            }
        }
        if (hs == hsEntries.end())
        {
            hsEntries.push_back(currentEntry);
            currentEntry = new HSEntry(this);
            std::string name;
            if (DriverName->getText() != "")
                name = DriverName->getText();
            currentEntry->setName(name);
        }
        while (hsEntries.size() > 100)
        {
            delete hsEntries.back();
            hsEntries.pop_back();
        }
        int pos = 1;
        for (std::list<HSEntry *>::iterator hs = hsEntries.begin(); hs != hsEntries.end(); ++hs)
        {
            (*hs)->setPos(pos++);
        }
        save(); // save highscore table
    }
    currentEntry->setStartTime(t);
    passedInterim = false;
}

void
Highscore::setInterimTime(double t)
{
    currentEntry->setInterimTime(t);
    passedInterim = true;
}

void
Highscore::setResetTime(double)
{
    currentEntry->reset();
    passedInterim = false;
}

void Highscore::load()
{
	XMLCh *t1 = NULL;
    xercesc::XercesDOMParser *parser = new xercesc::XercesDOMParser();
    parser->setValidationScheme(xercesc::XercesDOMParser::Val_Never);

    try
    {
        parser->parse("highscore.xml");
    }
    catch (...)
    {
        cerr << "error parsing highscore table" << endl;
    }

    xercesc::DOMDocument *xmlDoc = parser->getDocument();
    xercesc::DOMElement *rootElement = NULL;
    if (xmlDoc)
    {
        rootElement = xmlDoc->getDocumentElement();
    }

    if (rootElement)
    {
        xercesc::DOMNodeList *nodeList = rootElement->getChildNodes();
        for (int i = 0; i < nodeList->getLength(); ++i)
        {
            xercesc::DOMElement *node = dynamic_cast<xercesc::DOMElement *>(nodeList->item(i));
            if (!node)
                continue;
            char *pos = xercesc::XMLString::transcode(node->getAttribute(t1 = xercesc::XMLString::transcode("pos"))); xercesc::XMLString::release(&t1);
            char *name = xercesc::XMLString::transcode(node->getAttribute(t1 = xercesc::XMLString::transcode("name"))); xercesc::XMLString::release(&t1);
            char *startTime = xercesc::XMLString::transcode(node->getAttribute(t1 = xercesc::XMLString::transcode("startTime"))); xercesc::XMLString::release(&t1);
            char *interimTime = xercesc::XMLString::transcode(node->getAttribute(t1 = xercesc::XMLString::transcode("interimTime"))); xercesc::XMLString::release(&t1);
            char *endTime = xercesc::XMLString::transcode(node->getAttribute(t1 = xercesc::XMLString::transcode("endTime"))); xercesc::XMLString::release(&t1);
            HSEntry *he = new HSEntry(this);
            double d;
            int posi;
            sscanf(pos, "%d", &posi);
            he->setPos(posi);
            std::string names = name;
            he->setName(names);
            sscanf(startTime, "%lf", &d);
            he->setStartTime(d);
            sscanf(interimTime, "%lf", &d);
            he->setInterimTime(d);
            sscanf(endTime, "%lf", &d);
            he->setEndTime(d);
            hsEntries.push_back(he);
			xercesc::XMLString::release(&pos);
			xercesc::XMLString::release(&name);
			xercesc::XMLString::release(&startTime);
			xercesc::XMLString::release(&interimTime);
			xercesc::XMLString::release(&endTime);
        }
    }
}

void Highscore::save()
{
	XMLCh *t1 = NULL;
	XMLCh *t2 = NULL;
    xercesc::DOMDocument *document = impl->createDocument(0, t1 = xercesc::XMLString::transcode("Highscores"), 0); xercesc::XMLString::release(&t1);

    xercesc::DOMElement *rootElement = document->getDocumentElement();

    for (std::list<HSEntry *>::iterator hs = hsEntries.begin(); hs != hsEntries.end(); ++hs)
    {
        xercesc::DOMElement *hsElement = document->createElement(xercesc::XMLString::transcode("HSEntry"));

        char number[100];
        sprintf(number, "%d", (*hs)->getPos());
        hsElement->setAttribute(t1 = xercesc::XMLString::transcode("pos"), t2 = xercesc::XMLString::transcode(number)); xercesc::XMLString::release(&t1); xercesc::XMLString::release(&t2);
        hsElement->setAttribute(t1 = xercesc::XMLString::transcode("name"), t2 = xercesc::XMLString::transcode((*hs)->getName().c_str())); xercesc::XMLString::release(&t1); xercesc::XMLString::release(&t2);
        sprintf(number, "%lf", (*hs)->getStartTime());
        hsElement->setAttribute(t1 = xercesc::XMLString::transcode("startTime"), t2 = xercesc::XMLString::transcode(number)); xercesc::XMLString::release(&t1); xercesc::XMLString::release(&t2);
        sprintf(number, "%lf", (*hs)->getInterimTime());
        hsElement->setAttribute(t1 = xercesc::XMLString::transcode("interimTime"), t2 = xercesc::XMLString::transcode(number)); xercesc::XMLString::release(&t1); xercesc::XMLString::release(&t2);
        sprintf(number, "%lf", (*hs)->getEndTime());
        hsElement->setAttribute(t1 = xercesc::XMLString::transcode("endTime"), t2 = xercesc::XMLString::transcode(number)); xercesc::XMLString::release(&t1); xercesc::XMLString::release(&t2);
        rootElement->appendChild(hsElement);
    }

#if XERCES_VERSION_MAJOR < 3
    xercesc::DOMWriter *writer = impl->createDOMWriter();
    // set the format-pretty-print feature
    if (writer->canSetFeature(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true))
        writer->setFeature(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true);
    xercesc::XMLFormatTarget *xmlTarget = new xercesc::LocalFileFormatTarget("highscore.xml");
    bool written = writer->writeNode(xmlTarget, *rootElement);
    if (!written)
        fprintf(stderr, "Highscore::save info: Could not open file for writing !\n");

    delete writer;
    delete xmlTarget;
#else

    xercesc::DOMLSSerializer *writer = ((xercesc::DOMImplementationLS *)impl)->createLSSerializer();
    //xercesc::DOMConfiguration* dc = writer->getDomConfig();
    //dc->setParameter(xercesc::XMLUni::fgDOMErrorHandler,errorHandler);
    //dc->setParameter(xercesc::XMLUni::fgDOMWRTDiscardDefaultContent,true);

    xercesc::DOMLSOutput *theOutput = ((xercesc::DOMImplementationLS *)impl)->createLSOutput();
    theOutput->setEncoding(xercesc::XMLString::transcode("utf8")); xercesc::XMLString::release(&t1);

    bool written = writer->writeToURI(rootElement, xercesc::XMLString::transcode("highscore.xml")); xercesc::XMLString::release(&t1);
    if (!written)
        fprintf(stderr, "Material::save info: Could not open file for writing %s!\n", "highscore.xml");
    delete writer;

#endif
    delete document;
}

COVERPLUGIN(Highscore)
