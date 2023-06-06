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
#include "RopePlugin.h"
#include <osg/LineWidth>
#include <stdio.h>
#include <cover/coVRCommunication.h>
using namespace osg;

RopePlugin::RopePlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    this->numRopes = 0;
}

//Constructor of Rope
bool RopePlugin::init()
{
    fprintf(stderr, "RopePlugin::RopePlugin\n");

    rID = sgID = sID = wgID = -1;

    paramTab = new coTUITab("Rope", coVRTui::instance()->mainFolder->getID());
    paramTab->setPos(0, 0);
    TabFolder = new coTUITabFolder("numHTab", paramTab->getID());
    TabFolder->setPos(0, 0);
    Tab1 = new coTUITab("Load/Save", TabFolder->getID());
    Tab1->setPos(0, 0);
    Frame0 = new coTUIFrame("Frame0 of Tab1", Tab1->getID());
    Frame0->setPos(0, 0);
    SaveButton = new coTUIFileBrowserButton("Save", Frame0->getID());
    SaveButton->setMode(coTUIFileBrowserButton::SAVE);
    SaveButton->setEventListener(this);
    SaveButton->setPos(0, 1);
    SaveButton->setFilterList("*.xml");

    LoadButton = new coTUIFileBrowserButton("Load", Frame0->getID());
    coVRCommunication::instance()->setFBData(LoadButton->getVRBData());
    LoadButton->setMode(coTUIFileBrowserButton::OPEN);
    LoadButton->setEventListener(this);
    LoadButton->setPos(2, 1);
    //LoadButton->setFilterList(coVRFileManager::instance()->getFilterList());
    LoadButton->setFilterList("*.xml");

    TestRopeButton = new coTUIButton("Testrope", Frame0->getID());
    TestRopeButton->setEventListener(this);
    TestRopeButton->setPos(0, 3);
    AlbertRopeButton = new coTUIButton("Albert rope", Frame0->getID());
    AlbertRopeButton->setEventListener(this);
    AlbertRopeButton->setPos(2, 3);
    SaveTestRope = new coTUIButton("Save testrope", Frame0->getID());
    SaveTestRope->setEventListener(this);
    SaveTestRope->setPos(0, 4);
    SaveAlbertRope = new coTUIButton("Save albert rope", Frame0->getID());
    SaveAlbertRope->setEventListener(this);
    SaveAlbertRope->setPos(2, 4);
    LoadTestRope = new coTUIButton("Load Testrope", Frame0->getID());
    LoadTestRope->setEventListener(this);
    LoadTestRope->setPos(0, 5);
    LoadAlbertRope = new coTUIButton("Load Albert rope", Frame0->getID());
    LoadAlbertRope->setEventListener(this);
    LoadAlbertRope->setPos(2, 5);

    // Ab hier kommen die Schalter zum Manipulieren ...
    Tab2 = new coTUITab("Settings", TabFolder->getID());
    Tab2->setPos(0, 0);
    Frame1 = new coTUIFrame("Frame1 of Tab2", Tab2->getID());
    Frame1->setPos(0, 0);
    SelComboBox = new coTUIComboBox("numHSlider", Frame1->getID());
    SelComboBox->setEventListener(this);
    SelComboBox->setPos(0, 0);
    ColorTriangle = new coTUIColorTriangle("numHColor", Frame1->getID());
    ColorTriangle->setEventListener(this);
    ColorTriangle->setPos(1, 0);
    Frame2 = new coTUIFrame("Frame2 of Tab2", Tab2->getID());
    Frame2->setPos(0, 1);

    return true;
}

// this is called if the plugin is removed at runtime
// which currently never happens

// Destructor delets all functions and selections, which have been made, while closing plugin
RopePlugin::~RopePlugin()
{
    int i;

    fprintf(stderr, "RopePlugin::~RopePlugin\n");

    for (i = 0; i < this->numRopes; i++)
        delete Ropes[i];
    //while(ropeGroup->getNumParents())
    //ropeGroup->getParent(0)->removeChild(ropeGroup.get());

    delete paramTab;
    delete TabFolder;
    delete Tab1;
    delete Frame0;
    delete SaveButton;
    delete LoadButton;
    delete TestRopeButton;
    delete Tab2;
    delete Frame1;
    delete SelComboBox;
    delete ColorTriangle;
    delete Frame2;
}

// if a tablet event happened, than the program will look which event it was
void RopePlugin::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == SelComboBox)
    {
        adaptManipulation(this->Frame2);
    }
    else if (tUIItem == SaveButton)
    {
        char fn[256];
        char *p;
        p = fn;
        fprintf(stderr, "RopePlugin::tabletEvent Save\n");
        strcpy(fn, SaveButton->getSelectedPath().c_str());
        if (!strncmp(fn, "file://", 7))
            p += 7;
        this->Save(p);
    }
    else if (tUIItem == LoadButton)
    {
        fprintf(stderr, "RopePlugin::tabletPressEvent ... LoadButton\n");
        char fn[256];
        char *p;
        p = fn;
        fprintf(stderr, "RopePlugin::tabletEvent Save\n");
        strcpy(fn, LoadButton->getSelectedPath().c_str());
        if (!strncmp(fn, "file://", 7))
            p += 7;
        this->Load(p);
    }
}
void RopePlugin::adaptManipulation(coTUIFrame *frame)
{
    char buf[256];
    char *p;

    // Falls was altes da ist --> loeschen
    if (this->wgID >= 0)
        this->Ropes[this->rID]->Strandgroups[this->sgID]->Strands[this->sID]->Wiregroups[this->wgID]->delManipulation();
    else if (this->sID >= 0)
        this->Ropes[this->rID]->Strandgroups[this->sgID]->Strands[this->sID]->delManipulation();
    else if (this->sgID >= 0)
        this->Ropes[this->rID]->Strandgroups[this->sgID]->delManipulation();
    else if (this->rID >= 0)
        this->Ropes[this->rID]->delManipulation();
    this->rID = this->sgID = this->sID = this->wgID = -1;
    fprintf(stderr, "SelComboBox: %s\n", this->SelComboBox->getSelectedText().c_str());
    strcpy(buf, this->SelComboBox->getSelectedText().c_str());
    // Haessliche Loesung ... aber tut ...
    p = buf;
    if ((p = this->nextID(p)) != NULL)
    {
        this->rID = atoi(p);
        if ((p = this->nextID(p)) != NULL)
        {
            this->sgID = atoi(p);
            if ((p = this->nextID(p)) != NULL)
            {
                this->sID = atoi(p);
                if ((p = this->nextID(p)) != NULL)
                {
                    this->wgID = atoi(p);
                    this->Ropes[this->rID]->Strandgroups[this->sgID]->Strands[this->sID]->Wiregroups[this->wgID]->addManipulation(frame);
                }
                else
                {
                    this->Ropes[this->rID]->Strandgroups[this->sgID]->Strands[this->sID]->addManipulation(frame);
                }
            }
            else
            {
                this->Ropes[this->rID]->Strandgroups[this->sgID]->addManipulation(frame);
            }
        }
        else
        {
            this->Ropes[this->rID]->addManipulation(frame);
        }
    }
    else
    {
        fprintf(stderr, "Schrott ...\n");
    }
    // Und zu guter letzte stellen wir noch einen Zugriff auf die Farben sicher ...
    this->Ropes[this->rID]->colTr = this->ColorTriangle;
}

char *RopePlugin::nextID(char *p)
{
    while (*p && *p != '_')
        p++;
    return (*p ? ++p : NULL);
}

void RopePlugin::tabletPressEvent(coTUIElement *tUIItem)
{
    if (tUIItem == TestRopeButton)
    {
        int rID;
        rID = this->addRope(this->Frame1, this->SelComboBox);
        this->Ropes[rID]->TestRope1();
        this->Ropes[rID]->createGeom();
    }
    else if (tUIItem == AlbertRopeButton)
    {
        int rID;
        rID = this->addRope(this->Frame1, this->SelComboBox);
        this->Ropes[rID]->AlbertRope();
        this->Ropes[rID]->createGeom();
    }
    else if (tUIItem == LoadAlbertRope)
    {
        this->Load("c:\\temp\\albert_rope.xml");
    }
    else if (tUIItem == SaveAlbertRope)
    {
        this->Save("c:\\temp\\albert_rope.xml");
    }
    else if (tUIItem == LoadTestRope)
    {
        this->Load("c:\\temp\\test_rope.xml");
    }
    else if (tUIItem == SaveTestRope)
    {
        this->Save("c:\\temp\\test_rope.xml");
    }
    else if (tUIItem == SelComboBox)
    {
        adaptManipulation(this->Frame2);
    }
}
void RopePlugin::preFrame()
{
}

COVERPLUGIN(RopePlugin)

int RopePlugin::addRope(coTUIFrame *frame, coTUIComboBox *box)
{
    if (this->numRopes < this->maxRopes)
    {
        this->Ropes[this->numRopes] = new Rope(this->numRopes, frame, box);
        return this->numRopes++;
    }
    return -1;
}

bool RopePlugin::Save(const char *fn)
{
    int i;

    fprintf(stderr, "RopePlugin::Save %s\n", fn);

    xercesc::DOMImplementation *impl = xercesc::DOMImplementationRegistry::getDOMImplementation(xercesc::XMLString::transcode("Core"));

    xercesc::DOMDocument *document = impl->createDocument(0, xercesc::XMLString::transcode("IFT_ROPE"), 0);

    xercesc::DOMElement *rootElement = document->getDocumentElement();
    rootElement->appendChild(document->createTextNode(xercesc::XMLString::transcode("\n")));

    for (i = 0; i < this->numRopes; i++)
    {
        rootElement->appendChild(this->Ropes[i]->Save(*document));
    }
    rootElement->appendChild(document->createTextNode(xercesc::XMLString::transcode("\n")));

#if XERCES_VERSION_MAJOR < 3
    document->setVersion(xercesc::XMLString::transcode("1.0"));
    document->setStandalone(true);
    document->setEncoding(xercesc::XMLString::transcode("utf8"));
    xercesc::DOMWriter *writer = impl->createDOMWriter();
    // "discard-default-content" "validation" "format-pretty-print"
    xercesc::XMLFormatTarget *xmlTarget = new xercesc::LocalFileFormatTarget(fn);
    bool written = writer->writeNode(xmlTarget, *rootElement);
    if (!written)
        fprintf(stderr, "RopePlugin::Save(): error during write\n");

    delete writer;
    delete xmlTarget;
    delete document;
#else
    xercesc::DOMLSSerializer *writer = ((xercesc::DOMImplementationLS *)impl)->createLSSerializer();
    //xercesc::DOMConfiguration* dc = writer->getDomConfig();
    //dc->setParameter(xercesc::XMLUni::fgDOMErrorHandler,errorHandler);
    //dc->setParameter(xercesc::XMLUni::fgDOMWRTDiscardDefaultContent,true);
    xercesc::DOMLSOutput *theOutput = ((xercesc::DOMImplementationLS *)impl)->createLSOutput();
    theOutput->setEncoding(xercesc::XMLString::transcode("utf8"));
    bool written = writer->writeToURI(rootElement, xercesc::XMLString::transcode(fn));
    if (!written)
        fprintf(stderr, "RopePlugin::Save info: Could not open file for writing %s!\n", fn);
    delete writer;
    delete document;
#endif

    return written;
}

bool RopePlugin::Load(const char *fn)
{
    int i;
    int len;

    fprintf(stderr, "**************************************\n");
    fprintf(stderr, "RopePlugin::Load(%s)\n", fn);
    xercesc::XercesDOMParser *parser = new xercesc::XercesDOMParser();
    parser->setValidationScheme(xercesc::XercesDOMParser::Val_Never);

    try
    {
        parser->parse(fn);
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
        len = nodeList->getLength();
        fprintf(stderr, "RopePlugin::Load : NumberOfNodes=%d\n", len);
        for (i = 0; i < nodeList->getLength(); ++i)
        {
            int rID;

            fprintf(stderr, "RopePlugin::Load : node=%d\n", i);
            xercesc::DOMElement *node = dynamic_cast<xercesc::DOMElement *>(nodeList->item(i));
            if (node)
            {
                rID = this->addRope(this->Frame1, this->SelComboBox);
                this->Ropes[rID]->Load(node);
                this->Ropes[rID]->createGeom();
            }
            else
                fprintf(stderr, "RopePlugin::Load() error: i=%d\n", i);
        }
    }
    else
        fprintf(stderr, "RopePlugin::Load : NO rootElement\n");
    return true;
}

void RopePlugin::recurseSetLenFactor(float val)
{
    int i;

    for (i = 0; i < this->numRopes; i++)
        this->Ropes[i]->recurseSetLenFactor(val);
}
