/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
**                                                            (C)2001 HLRS  **
**                                                                          **
** Description: Rope Plugin (does nothing)                              **
**                                                                          **
**                                                                          **
** Author: U.Woessner		                                                **
**                                                                          **
** History:  								                                **
** Nov-01  v1	    				       		                            **
**                                                                          **
**                                                                       **
\****************************************************************************/

#define USE_MATH_DEFINES
#include <math.h>
#include <config/coConfig.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRTui.h>
#include <cover/coVRShader.h>
#include <OpenVRUI/coTrackerButtonInteraction.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <osg/PolygonMode>
#include "SimpleRopePlugin.h"
#include <osg/LineWidth>
#include <stdio.h>
#include <QDir>
using namespace osg;

//Constructor of Wire
Wire::Wire(float r, float len, float slength, float rlength, int ns, int nls, int WNum, float a, osg::Group *group)
{
    //Declaration of short terms of parameters
    numSegments = ns;
    numLengthSegments = nls;
    length = len;
    strandlength = slength;
    ropelength = rlength;
    radius = r;
    angle = a;
    radiusb = r / cos(a);

    //start of design functions
    geom = new osg::Geometry();
    geode = new Geode();
    geom->setUseDisplayList(false);
    geom->setUseVertexBufferObjects(false);
    geode->addDrawable(geom.get());
    vert = new Vec3Array;
    primitives = new DrawArrayLengths(PrimitiveSet::POLYGON);
    normals = new Vec3Array;
    //cindices = new UShortArray()   //colors = new Vec4Array();
    StateSet *geoState = geom->getOrCreateStateSet();

    createGeom();

    geom->setVertexArray(vert.get());
    geom->setNormalArray(normals.get());
    geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
    //geom->setColorIndices(cindices.get());
    //geom->setColorArray(colors.get());
    geom->addPrimitiveSet(primitives.get());

    geoState = geode->getOrCreateStateSet();
    if (globalmtl.get() == NULL)
    {
        globalmtl = new Material;
        globalmtl->ref();
        globalmtl->setColorMode(Material::AMBIENT_AND_DIFFUSE);
        globalmtl->setAmbient(Material::FRONT_AND_BACK, Vec4(0.2f, 0.2f, 0.2f, 1.0));
        globalmtl->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.0));
        globalmtl->setSpecular(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.0));
        globalmtl->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.0));
        globalmtl->setShininess(Material::FRONT_AND_BACK, 10.0f);
    }

    geoState->setRenderingHint(StateSet::OPAQUE_BIN);
    geoState->setMode(GL_BLEND, StateAttribute::OFF);
    geoState->setAttributeAndModes(globalmtl.get(), StateAttribute::ON);

    coVRShaderList::instance()->get("SolidClipping")->apply(geode, geom);
    char name[1000];
    sprintf(name, "Wire %d", WNum);
    geode->setName(name);
    group->addChild(geode.get());
    //end of design functions
}
void Wire::setColor(Vec4 color)
{
    globalmtl->setDiffuse(Material::FRONT_AND_BACK, color);
}
osg::Vec4 Wire::getColor()
{
    return globalmtl->getDiffuse(Material::FRONT_AND_BACK);
}

void Wire::createGeom() //Erstellen eines Drahtes
{
    primitives->clear();
    normals->clear();
    vert->clear();
    indices.clear();
    coord.clear();
    norm.clear();

    int nL = (int)(numLengthSegments * length * strandlength);
    float segHeight = ropelength / numLengthSegments;
    for (int h = 0; h < (nL + 1); h++)
    {
        for (int s = 0; s < numSegments; s++)
        {
            float angle = 2 * M_PI / numSegments * s;
            Vec3 v;
            v.set(sin(angle) * radius, cos(angle) * radius, h * segHeight);
            coord.push_back(v);
            v[2] = 0;
            v.normalize();
            norm.push_back(v);
        }
    }
    for (int h = 0; h < nL; h++)
    {
        for (int s = 0; s < numSegments; s++)
        {
            if (s < (numSegments - 1))
            {
                indices.push_back((h + 1) * numSegments + s);
                indices.push_back((h + 1) * numSegments + s + 1);
                indices.push_back((h)*numSegments + s + 1);
                indices.push_back((h)*numSegments + s);
            }
            else
            {
                indices.push_back((h + 1) * numSegments + s);
                indices.push_back((h + 1) * numSegments);
                indices.push_back((h)*numSegments);
                indices.push_back((h)*numSegments + s);
            }
        }
    }
    for (size_t i = 0; i < indices.size(); ++i)
    {
        vert->push_back(coord[indices[i]]);
        normals->push_back(norm[indices[i]]);
        if (i % 4 == 0)
            primitives->push_back(4);
    }

    /* das machen wir
      spaeter */
    // calcColors();
    geom->dirtyBound();
}
//Wire Destructor delets all functions and selections, which have been made
Wire::~Wire()
{
    while (geode->getNumParents())
        geode->getParent(0)->removeChild(geode.get());
}

void Wire::setRadius(float r)
{
    radius = r;
    createGeom();
}

float Wire::getRadius()
{
    return radius;
}
void Wire::setAngle(float a)
{
    angle = a;
    radiusb = radius / cos(a);
    createGeom();
}

float Wire::getAngle()
{
    return angle;
}

float Strand::getCoreRadius()
{
    return core->getRadius();
}
void Wire::rotate(float startAngle, float lot, float r) //Funktion "rotate" für Drähte wird ausgeführt
{
    lengthOfTwist = lot;
    osg::Matrix m, incRot, trans, startRot;
    trans.setTrans(r, 0, 0);
    startRot.setRotate(osg::Quat(startAngle, osg::Vec3(0, 0, 1)));
    m = trans * startRot;
    double alpha = (2.0 * M_PI / numLengthSegments) * (ropelength / lengthOfTwist);
    incRot.setRotate(osg::Quat(alpha, osg::Vec3(0, 0, 1)));
    osg::Vec3Array *v = vert.get();
    osg::Vec3Array *n = normals.get();
    int vNum = 0;

    int nL = (int)(numLengthSegments * length * strandlength);

    float twistangle = -atan((2 * r * sin(alpha / 2.0)) / (ropelength / numLengthSegments));
    osg::Matrix twist;
    twist.makeRotate(twistangle, osg::Vec3(1, 0, 0));
    for (int h = 0; h < (nL + 1); h++)
    {
        osg::Matrix trans, itrans;
        trans.makeTranslate(0, 0, h * (ropelength / numLengthSegments));
        itrans.makeTranslate(0, 0, -h * (ropelength / numLengthSegments));
        osg::Matrix summ = itrans * twist * trans * m;
        for (int s = 0; s < numSegments; s++)
        {
            coord[vNum] = coord[vNum] * summ;
            norm[vNum] = osg::Matrix::transform3x3(norm[vNum], summ);
            vNum++;
        }
        m = m * incRot;
    }
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int k = indices[i];
        (*v)[i] = coord[k];
        (*n)[i] = norm[k];
    }
}
float Wire::getLengthOfTwist() // Herausgabe der Variablen für Schlaglänge
{
    return lengthOfTwist;
}

//Constructor of Strand
Strand::Strand(int nw, float coreRadius, float hullRadius, float len, float rlength, int numSegents, int numLengthSegments, int SNum, osg::Group *group, float lot, float tr)
{
    numWires = nw;
    strandGroup = new osg::Group();
    group->addChild(strandGroup.get());

    length = len;
    ropelength = rlength;
    char name[1000];
    sprintf(name, "Strand %d", SNum);
    strandGroup->setName(name);

    lengthOfTwist = lot;
    twistRadius = tr;

    core = new Wire(coreRadius, 1.0, length, ropelength, numSegents, numLengthSegments, 0, 0, strandGroup.get());
    for (int i = 0; i < numWires - 1; i++)
    {
        float angle = atan(lengthOfTwist / (2.0 * M_PI * twistRadius));
        wires[i] = new Wire(hullRadius, 1.0, length, ropelength, numSegents, numLengthSegments, i + 1, angle, strandGroup.get());
        wires[i]->rotate((2.0 * M_PI / (numWires - 1)) * i, lengthOfTwist, twistRadius);
    }
}
//Strand Destructor delets all functions and selections, which have been made
Strand::~Strand()
{
    delete core;
    for (int i = 0; i < (numWires - 1); i++)
    {
        delete wires[i];
    }
    while (strandGroup->getNumParents())
        strandGroup->getParent(0)->removeChild(strandGroup.get());
}
void Strand::setColor(osg::Vec4 color)
{
    core->setColor(color);
    for (int i = 0; i < (numWires - 1); i++)
    {
        wires[i]->setColor(color);
    }
}
void Strand::setLengthOfTwist(float lot) // Zuweisen der Variablen für Schlaglänge
{
    lengthOfTwist = lot;
}
float Strand::getLengthOfTwist() // Herausgabe der Variablen für Schlaglänge
{
    return lengthOfTwist;
}
float Strand::getLengthOfTwistWire() // Herausgabe der Variablen für Schlaglänge
{
    return core->getLengthOfTwist();
}
void Strand::setlength(float len) //Die Variable len wird weitergegeben an length(Wire)
{
    length = len;
}

void Wire::setstrandlength(float length) //Die Variable strandlength wird gesetzt
{
    strandlength = length;
}

void Strand::setropelength(float length) //Die Variable ropelength wird weitergegeben an length(Strand)
{
    ropelength = length;
}
float Strand::getropelength() //Die Variable ropelength wird ausgegeben
{
    return ropelength;
}
void Strand::setwirelength(int ind, float length) //Länge eines bestimmten Wires
{
    if (ind == numWires - 1) //Länge des CoreWires
    {
        core->setlength(length);
    }
    if (ind == numWires) //Länge der gesammten Litze
    {
        core->setlength(length);
        for (int i = 0; i < numWires - 1; i++)
        {
            wires[i]->setlength(length);
        }
    }
    if (ind < numWires - 1) //Länge der jeweils anderen Wires
    {
        wires[ind]->setlength(length);
    }
}
float Strand::getWireLength(int index) //
{
    if (index >= numWires - 1)
    {
        return core->getLength();
    }
    else
    {
        return wires[index]->getLength();
    }
}

void Strand::setcorelengthfact(float length) //Länge eines bestimmten Wires
{
    core->setlength(length);
}

void Strand::setwirecolor(int ind, osg::Vec4 color) //Die Farbe der einzelnen Drähte wird festgelegt
{
    if (ind == numWires - 1) //Farbe für den CoreWire
    {
        core->setColor(color);
    }

    if (ind == numWires) //Farbe für die gesammte Litze
    {
        core->setColor(color);
        for (int i = 0; i < numWires - 1; i++)
        {
            wires[i]->setColor(color);
        }
    }

    if (ind < numWires - 1) //Farbe für die jeweils anderen Drähte
    {
        wires[ind]->setColor(color);
    }
}

osg::Vec4 Strand::getWireColor(int index) //Die Farbe der einzelnen Drähte wird festgelegt
{
    if (index >= numWires - 1)
    {
        return core->getColor();
    }
    else
    {
        return wires[index]->getColor();
    }
}

void Wire::setropelength(float length)
{
    ropelength = length;
    // angepasste "Anzeigelaenge"
    numLengthSegments = (int)(ropelength / 10.);
}

void Wire::setlength(float len)
{
    length = len;
}

float Wire::getLength()
{
    return length;
}

void Strand::createGeom()
{
    core->setstrandlength(length); //Kerndraht soll gleiche Länge wie Litze haben
    core->setropelength(ropelength); //Kerndraht soll auch gleiche Länge haben wie Seil
    core->createGeom(); //Kerndraht wird erstellt
    for (int i = 0; i < numWires - 1; i++) //Nun werden weitere Drähte erstellt und vom Mittelpunkt verschoben und rotiert. Eine Litze entsteht.
    {
        wires[i]->setstrandlength(length);
        wires[i]->setropelength(ropelength);
        wires[i]->createGeom();
        wires[i]->rotate((2.0 * M_PI / (numWires - 1)) * i, lengthOfTwist, twistRadius);
    }
}

void Strand::rotate(float startAngle, float lengthOfTwist, float radius)
{
    core->rotate(startAngle, lengthOfTwist, radius);
    for (int i = 0; i < (numWires - 1); i++)
    {
        wires[i]->rotate(startAngle, lengthOfTwist, radius);
    }
}

SimpleRopePlugin::SimpleRopePlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
}

//Constructor of Rope
bool SimpleRopePlugin::init()
{
    fprintf(stderr, "SimpleRopePlugin::SimpleRopePlugin\n");

    readConfig();

    paramTab = new coTUITab("Rope", coVRTui::instance()->mainFolder->getID());
    paramTab->setPos(0, 0);

    //Erstellen einer neuen Gruppe
    ropeGroup = new osg::Group();
    ropeGroup->setName("Rope 0");
    cover->getObjectsRoot()->addChild(ropeGroup.get());

    /*core of rope is a strand. The first core strand of the rope is straigth that means there is no roation of the strand itself.
A strand is built of seven wires with a radius of ten and a hull radius of also 10. 
The length of the wires is 1000 and its made with eigth segments and 100 length segments. 
The number of this segment (strand) is set to ziro. 
Twistlength ist 200 and the twistradius is 20. radius of wire plus hullradius.
*/
    length = 1000.0;
    numStrands = 6;
    numWires = 7;
    core = new Strand(7, 10, 10, 1.0, length, 8, 100, 0, ropeGroup.get(), 200, 20);
    core->setropelength(length);
    //for five strands, built them, move them to a distance of six times the radius
    //place the center point of the strands on every 2.0*M_PI/6 and rotate them with a twist length of 600.
    for (int i = 0; i < numStrands; i++)
    {
        //strands[i] = NULL;

        strands[i] = new Strand(7, 10, 10, 1.0, length, 8, 100, i + 1, ropeGroup.get(), 200, 20);
        strands[i]->setropelength(length);
        strands[i]->rotate((2.0 * M_PI / numStrands) * i, 600, 60);
    }

    TabFolder = new coTUITabFolder("numHTab", paramTab->getID());
    TabFolder->setPos(0, 0);

    Tab1 = new coTUITab("Basic Settings", TabFolder->getID());
    Tab1->setPos(0, 0);
    //Tab2 = new coTUITab("Color",TabFolder->getID());
    //Tab2->setPos(3,0);
    Tab3 = new coTUITab("Show Room", TabFolder->getID());
    Tab3->setPos(4, 0);
    Frame0 = new coTUIFrame("Frame0 of Tab3", Tab3->getID());
    Frame0->setPos(0, 0);
    Frame1 = new coTUIFrame("Frame1 of Tab3", Tab3->getID());
    Frame1->setPos(1, 0);
    Frame2 = new coTUIFrame("Frame2 of Tab3", Tab3->getID());
    Frame2->setPos(2, 0);

    Frame3 = new coTUIFrame("Frame3 of Tab1", Tab1->getID());
    Frame3->setPos(0, 0);

    //Tab1 (Basic Settings)
    strandLengthOfTwistSlider = new coTUIFloatSlider("numHSlider", Frame3->getID());
    strandLengthOfTwistSlider->setEventListener(this);
    strandLengthOfTwistSlider->setPos(1, 1);
    strandLengthOfTwistSlider->setMin(2);
    strandLengthOfTwistSlider->setMax(1000);
    strandLengthOfTwistSlider->setValue(200);
    strandLengthOfTwistLabel = new coTUILabel("Length of Twist of Strands:", Frame3->getID());
    strandLengthOfTwistLabel->setPos(0, 1);

    lengthOfTwistSlider = new coTUIFloatSlider("numHSlider", Frame3->getID());
    lengthOfTwistSlider->setEventListener(this);
    lengthOfTwistSlider->setPos(1, 2);
    lengthOfTwistSlider->setMin(2);
    lengthOfTwistSlider->setMax(1000);
    lengthOfTwistSlider->setValue(600);
    lengthOfTwistLabel = new coTUILabel("Length of Twist of Rope:", Frame3->getID());
    lengthOfTwistLabel->setPos(0, 2);

    lengthSlider = new coTUIFloatSlider("numHSlider", Frame3->getID());
    lengthSlider->setEventListener(this);
    lengthSlider->setPos(1, 3);
    lengthSlider->setMin(100);
    lengthSlider->setMax(5000);
    lengthSlider->setValue(1000);
    lengthLabel = new coTUILabel("Length Rope:", Frame3->getID());
    lengthLabel->setPos(0, 3);

    Label2OfTab1 = new coTUILabel("", Frame3->getID()); //Platzhalter
    Label2OfTab1->setPos(0, 4);

    Label1OfTab1 = new coTUILabel("Select:", Frame3->getID());
    Label1OfTab1->setPos(0, 5);

    StrandComboBox = new coTUIComboBox("numHSlider", Frame3->getID());
    StrandComboBox->setEventListener(this);
    StrandComboBox->setPos(0, 6);
    StrandComboBox->addEntry("Strand 1");
    StrandComboBox->addEntry("Strand 2");
    StrandComboBox->addEntry("Strand 3");
    StrandComboBox->addEntry("Strand 4");
    StrandComboBox->addEntry("Strand 5");
    StrandComboBox->addEntry("Strand 6");
    StrandComboBox->addEntry("Core");

    WireComboBox = new coTUIComboBox("numHSlider", Frame3->getID());
    WireComboBox->setEventListener(this);
    WireComboBox->setPos(1, 6);
    WireComboBox->addEntry("Wire 1");
    WireComboBox->addEntry("Wire 2");
    WireComboBox->addEntry("Wire 3");
    WireComboBox->addEntry("Wire 4");
    WireComboBox->addEntry("Wire 5");
    WireComboBox->addEntry("Wire 6");
    WireComboBox->addEntry("Core");
    WireComboBox->addEntry("All");

    lengthfactSlider = new coTUIFloatSlider("numHSlider", Frame3->getID());
    lengthfactSlider->setEventListener(this);
    lengthfactSlider->setPos(1, 7);
    lengthfactSlider->setMin(0.2);
    lengthfactSlider->setMax(1.3);
    lengthfactSlider->setValue(1.0);
    lengthfactLabel = new coTUILabel("Length factor", Frame3->getID());
    lengthfactLabel->setPos(0, 7);

    WireColorButton = new coTUIButton("Color", Frame3->getID());
    WireColorButton->setEventListener(this);
    WireColorButton->setPos(0, 8);

    //Tab2 (Color)

    ColorTriangle = new coTUIColorTriangle("numHColor", Tab1->getID());
    ColorTriangle->setEventListener(this);
    ColorTriangle->setPos(0, 1);

    Colorbutton = new coTUIButton("Rope Color", Frame3->getID());
    Colorbutton->setEventListener(this);
    Colorbutton->setPos(0, 9);

    //Tab3 (Show Room)
    //Frame0
    Label0OfTab3 = new coTUILabel("Twist Type:", Frame0->getID());
    Label0OfTab3->setPos(0, 0);

    Label14OfTab3 = new coTUILabel("Parallel:", Frame0->getID());
    Label14OfTab3->setPos(0, 1);

    parallelleftbutton = new coTUIButton("left", Frame0->getID());
    parallelleftbutton->setEventListener(this);
    parallelleftbutton->setPos(0, 2);

    parallelrightbutton = new coTUIButton("right", Frame0->getID());
    parallelrightbutton->setEventListener(this);
    parallelrightbutton->setPos(0, 3);

    Label15OfTab3 = new coTUILabel("Crossed:", Frame0->getID());
    Label15OfTab3->setPos(0, 4);

    crossedleftbutton = new coTUIButton("left", Frame0->getID());
    crossedleftbutton->setEventListener(this);
    crossedleftbutton->setPos(0, 5);

    crossedrightbutton = new coTUIButton("right", Frame0->getID());
    crossedrightbutton->setEventListener(this);
    crossedrightbutton->setPos(0, 6);

    //Frame1
    Label1OfTab3 = new coTUILabel("Fast Buttons:", Frame1->getID());
    Label1OfTab3->setPos(0, 0);

    FiftyPercentButton = new coTUIButton("50% Strands", Frame1->getID());
    FiftyPercentButton->setEventListener(this);
    FiftyPercentButton->setPos(0, 1);

    StrandAndCoreButton = new coTUIButton("Core and Strand", Frame1->getID());
    StrandAndCoreButton->setEventListener(this);
    StrandAndCoreButton->setPos(0, 2);

    OnlyOneStrandButton = new coTUIButton("Strand", Frame1->getID());
    OnlyOneStrandButton->setEventListener(this);
    OnlyOneStrandButton->setPos(0, 3);

    Label2OfTab3 = new coTUILabel("", Frame1->getID()); //Platzhalter
    Label2OfTab3->setPos(0, 4);

    OnlyOneWireButton = new coTUIButton("Wire", Frame1->getID());
    OnlyOneWireButton->setEventListener(this);
    OnlyOneWireButton->setPos(0, 5);

    OneWireWithCoreButton = new coTUIButton("Core and Wire", Frame1->getID());
    OneWireWithCoreButton->setEventListener(this);
    OneWireWithCoreButton->setPos(0, 6);

    Label2OfTab3 = new coTUILabel("", Frame1->getID()); //Platzhalter
    Label2OfTab3->setPos(0, 8);

    OriginButton = new coTUIButton("Origin", Frame1->getID());
    OriginButton->setEventListener(this);
    OriginButton->setPos(0, 9);

    //Frame2
    Label3OfTab3 = new coTUILabel("Data:", Frame2->getID());
    Label3OfTab3->setPos(0, 0);

    FileButton = new coTUIButton("Save", Frame2->getID());
    FileButton->setEventListener(this);
    FileButton->setPos(0, 1);

    LoadButton = new coTUIButton("Load", Frame2->getID());
    LoadButton->setEventListener(this);
    LoadButton->setPos(0, 2);

    return true;
}

// this is called if the plugin is removed at runtime
// which currently never happens

// Destructor delets all functions and selections, which have been made, while closing plugin
SimpleRopePlugin::~SimpleRopePlugin()
{
    fprintf(stderr, "SimpleRopePlugin::~SimpleRopePlugin\n");

    delete core;
    for (int i = 0; i < 6; i++)
    {
        delete strands[i];
    }

    while (ropeGroup->getNumParents())
        ropeGroup->getParent(0)->removeChild(ropeGroup.get());
    delete strandLengthOfTwistSlider;
    delete strandLengthOfTwistLabel;
    delete lengthOfTwistSlider;
    delete lengthOfTwistLabel;
    delete lengthSlider;
    delete lengthLabel;
    delete lengthfactSlider;
    delete lengthfactLabel;
    delete Label1OfTab1;
    delete Label2OfTab1;
    delete Label0OfTab3;
    delete Label1OfTab3;
    delete Label2OfTab3;
    delete Label3OfTab3;
    delete Label14OfTab3;
    delete Label15OfTab3;
    delete Tab1;
    //delete Tab2;
    delete Tab3;
    delete Frame0;
    delete Frame1;
    delete Frame2;
    delete Frame3;
    delete StrandComboBox;
    delete WireComboBox;
    delete ColorTriangle;
    delete parallelrightbutton;
    delete parallelleftbutton;
    delete crossedleftbutton;
    delete crossedrightbutton;
    delete Colorbutton;
    delete FiftyPercentButton;
    delete StrandAndCoreButton;
    delete OnlyOneStrandButton;
    delete WireColorButton;
    delete OnlyOneWireButton;
    delete FileButton;
    delete LoadButton;
    delete OriginButton;
    delete OneWireWithCoreButton;
    delete TabFolder;
    delete paramTab;
}

// if a tablet event happened, than the program will look which event it was
void SimpleRopePlugin::tabletEvent(coTUIElement *tUIItem)
{

    // Schlaglänge der Drähte in der Litze
    if (tUIItem == strandLengthOfTwistSlider)
    {
        core->setLengthOfTwist(strandLengthOfTwistSlider->getValue()); //Schlaglänge der Drähte in der Kernlitze
        core->createGeom();
        for (int i = 0; i < numStrands; i++)
        {
            strands[i]->setLengthOfTwist(strandLengthOfTwistSlider->getValue()); //Schlaglänge der Drähte in den Außenlitzen
            strands[i]->createGeom(); //erstellen der Litzen
            strands[i]->rotate((2.0 * M_PI / numStrands) * i, lengthOfTwistSlider->getValue(), 60); //rotieren der Litzen
        }
    }

    //Schlaglänge der Litzen im Seil (dabei muss die Kernlitze nicht rotiert werden)
    if (tUIItem == lengthOfTwistSlider)
    {
        for (int i = 0; i < numStrands; i++)
        {
            strands[i]->setLengthOfTwist(strandLengthOfTwistSlider->getValue()); //Schlaglänge der Drähte in den Außenlitzen
            strands[i]->createGeom(); //erstellen der Litzen
            strands[i]->rotate((2.0 * M_PI / numStrands) * i, lengthOfTwistSlider->getValue(), 60); //rotieren der Litzen
        }
    }

    //Gesamtlänge des Seils
    if (tUIItem == lengthSlider)
    {
        core->setropelength(lengthSlider->getValue()); //Länge (Litze) von Slider nehmen
        core->createGeom(); //Kernlitze wird erstellt

        for (int i = 0; i < numStrands; i++)
        {
            strands[i]->setropelength(lengthSlider->getValue());
            strands[i]->setLengthOfTwist(strandLengthOfTwistSlider->getValue());
            strands[i]->createGeom();
            strands[i]->rotate((2.0 * M_PI / numStrands) * i, lengthOfTwistSlider->getValue(), 60);
        }
    }

    //Kombinationsmöglichkeiten mit den Comboboxen
    if (tUIItem == lengthfactSlider)
    {
        int wireI, strandI; //Die Variablen wireI und strandI entsprechen den integers aus den Comboboxen
        wireI = WireComboBox->getSelectedEntry();
        strandI = StrandComboBox->getSelectedEntry();

        core->setropelength(lengthSlider->getValue());
        core->createGeom();

        if (strandI == numStrands) //Auswahl der Kernlitze, wenn strandI=numStrands also =6 ist
        {
            core->setwirelength(wireI, lengthfactSlider->getValue());
            core->setropelength(lengthSlider->getValue());
            core->createGeom();
        }

        else //Auswahl der anderen Litzen, wenn strandI andere Werte als 6 annimmt
        {
            for (int i = 0; i < numStrands; i++)
            {
                if (i == strandI)
                    strands[i]->setwirelength(wireI, lengthfactSlider->getValue());

                strands[i]->setropelength(lengthSlider->getValue());
                strands[i]->setLengthOfTwist(strandLengthOfTwistSlider->getValue());
                strands[i]->createGeom();
                strands[i]->rotate((2.0 * M_PI / numStrands) * i, lengthOfTwistSlider->getValue(), 60);
            }
        }
    }
}

void SimpleRopePlugin::tabletPressEvent(coTUIElement *tUIItem)
{
    if (tUIItem == parallelleftbutton)
    {
        core->setLengthOfTwist(-200);
        core->createGeom();
        for (int i = 0; i < numStrands; i++)
        {

            strands[i]->setLengthOfTwist(-200);
            strands[i]->createGeom();
            strands[i]->rotate((2.0 * M_PI / numStrands) * i, -600, 60);
        }
    }

    if (tUIItem == parallelrightbutton)
    {
        core->setLengthOfTwist(200);
        core->createGeom();
        for (int i = 0; i < numStrands; i++)
        {

            strands[i]->setLengthOfTwist(200);
            strands[i]->createGeom();
            strands[i]->rotate((2.0 * M_PI / numStrands) * i, 600, 60);
        }
    }

    if (tUIItem == crossedleftbutton)
    {
        core->setLengthOfTwist(200);
        core->createGeom();
        for (int i = 0; i < numStrands; i++)
        {

            strands[i]->setLengthOfTwist(200);
            strands[i]->createGeom();
            strands[i]->rotate((2.0 * M_PI / numStrands) * i, -600, 60);
        }
    }

    if (tUIItem == crossedrightbutton)
    {
        core->setLengthOfTwist(-200);
        core->createGeom();
        for (int i = 0; i < numStrands; i++)
        {

            strands[i]->setLengthOfTwist(-200);
            strands[i]->createGeom();
            strands[i]->rotate((2.0 * M_PI / numStrands) * i, 600, 60);
        }
    }

    if (tUIItem == Colorbutton) // Farbe für das gesammte Seil
    {
        core->setColor(Vec4(ColorTriangle->getRed(), ColorTriangle->getGreen(), ColorTriangle->getBlue(), 1.0));
        for (int i = 0; i < numStrands; i++)
        {
            strands[i]->setColor(Vec4(ColorTriangle->getRed(), ColorTriangle->getGreen(), ColorTriangle->getBlue(), 1.0));
        }
    }

    if (tUIItem == WireColorButton)
    {
        int wireI, strandI;
        wireI = WireComboBox->getSelectedEntry();
        strandI = StrandComboBox->getSelectedEntry();

        if (strandI == numStrands)
        {
            core->setwirecolor(wireI, Vec4(ColorTriangle->getRed(), ColorTriangle->getGreen(), ColorTriangle->getBlue(), 1.0));
        }

        if (strandI < numStrands)
        {
            for (int i = 0; i < numStrands; i++)
            {
                if (i == strandI)
                {
                    strands[i]->setwirecolor(wireI, Vec4(ColorTriangle->getRed(), ColorTriangle->getGreen(), ColorTriangle->getBlue(), 1.0));
                }
            }
        }
    }

    if (tUIItem == FiftyPercentButton)
    {
        core->setropelength(1000);
        core->setlength(1.0);
        core->setwirelength(numWires, 1);
        core->createGeom();
        for (int i = 0; i < numStrands; i++)
        {
            strands[i]->setwirelength(numWires, 0.5);
            strands[i]->setropelength(1000);
            strands[i]->setlength(1);
            strands[i]->setLengthOfTwist(strandLengthOfTwistSlider->getValue());
            strands[i]->createGeom();
            strands[i]->rotate((2.0 * M_PI / numStrands) * i, lengthOfTwistSlider->getValue(), 60);
        }
    }

    if (tUIItem == OriginButton)
    {
        core->setLengthOfTwist(200);
        core->setropelength(1000);
        core->setlength(1.0);
        core->setwirelength(numWires, 1);
        core->setColor(Vec4(0.9, 0.9, 0.9, 1.0));
        core->createGeom();

        for (int i = 0; i < numStrands; i++)
        {
            strands[i]->setwirelength(numWires, 1);
            strands[i]->setropelength(1000);
            strands[i]->setlength(1.0);
            strands[i]->setLengthOfTwist(200);
            strands[i]->setColor(Vec4(0.9, 0.9, 0.9, 1.0));
            strands[i]->createGeom();
            strands[i]->rotate((2.0 * M_PI / numStrands) * i, 600, 60);
        }
    }

    if (tUIItem == StrandAndCoreButton)
    {
        core->setropelength(1000);
        core->setlength(1);
        core->setwirelength(numWires, 1);
        core->createGeom();

        for (int i = 0; i < numStrands; i++)
        {
            strands[i]->setropelength(1000);

            if (i == 0)
            {
                strands[i]->setlength(1);
                strands[i]->setwirelength(numWires, 1);
            }
            else
                strands[i]->setlength(0);
            strands[i]->setLengthOfTwist(strandLengthOfTwistSlider->getValue());
            strands[i]->createGeom();
            strands[i]->rotate((2.0 * M_PI / numStrands) * i, lengthOfTwistSlider->getValue(), 60);
        }
    }

    if (tUIItem == OnlyOneStrandButton)
    {
        core->setropelength(0);
        core->createGeom();

        for (int i = 0; i < numStrands; i++)
        {

            strands[i]->setropelength(1000);
            if (i == 0)
            {
                strands[i]->setlength(1);
                strands[i]->setwirelength(numWires, 1);
            }

            else
                strands[i]->setlength(0);
            strands[i]->setLengthOfTwist(strandLengthOfTwistSlider->getValue());
            strands[i]->createGeom();
            strands[i]->rotate((2.0 * M_PI / numStrands) * i, lengthOfTwistSlider->getValue(), 60);
        }
    }

    if (tUIItem == OnlyOneWireButton)
    {
        core->setropelength(1000);
        core->setwirelength(numWires, 0);
        core->createGeom();

        for (int i = 0; i < numStrands; i++)
        {
            strands[i]->setropelength(1000);

            if (i == 0)
            {
                strands[i]->setwirelength(numWires, 0);
                strands[i]->setwirelength(0, 1);
            }

            else
                strands[i]->setlength(0);
            strands[i]->setLengthOfTwist(strandLengthOfTwistSlider->getValue());
            strands[i]->createGeom();
            strands[i]->rotate((2.0 * M_PI / numStrands) * i, lengthOfTwistSlider->getValue(), 60);
        }
    }

    if (tUIItem == OneWireWithCoreButton)
    {
        core->setropelength(1000);
        core->setwirelength(numWires, 0);
        core->createGeom();

        for (int i = 0; i < numStrands; i++)
        {
            strands[i]->setropelength(1000);

            if (i == 0)
            {
                strands[i]->setwirelength(numWires, 0);
                strands[i]->setwirelength(0, 1);
                strands[i]->setwirelength(numWires - 1, 1);
            }

            else
                strands[i]->setlength(0);
            strands[i]->setLengthOfTwist(strandLengthOfTwistSlider->getValue());
            strands[i]->createGeom();
            strands[i]->rotate((2.0 * M_PI / numStrands) * i, lengthOfTwistSlider->getValue(), 60);
        }
    }

    // Datei speichern
    if (tUIItem == FileButton)
    {
        osg::Vec4 color;

        char dateiname[] = "C:\\TEMP\\Testname.txt";

        if ((fileParam = fopen(dateiname, "w")) != NULL)
        {
            fprintf(fileParam, "SimpleRopePlugin Configuration File\n\n");

            fprintf(fileParam, "[Global Parameters]\n");
            fprintf(fileParam, "Length of Twist of Strand = %f\n", strands[0]->getLengthOfTwist());
            fprintf(fileParam, "Length of Twist of Rope = %f\n", strands[0]->getLengthOfTwistWire());
            fprintf(fileParam, "Length Rope = %f\n\n", core->getropelength());

            fprintf(fileParam, "\n[Length factor of Strands]\n"); //Längenfaktoren für die Litzen
            fprintf(fileParam, "Core Strand = %i, %f\n\n", numWires - 1, core->getLength());

            for (int i = 0; i < numStrands; i++)
            {
                fprintf(fileParam, "Strand = %i, %f\n\n", i, strands[i]->getLength());
            }

            fprintf(fileParam, "\n[Wires]\n");
            color = core->getWireColor(numWires - 1); //Kerndraht von Kernlitze
            fprintf(fileParam, "Core Wire [S#LRRGBT] = %i, %i, %f, %f, %f, %f, %f, %f\n\n", numWires - 1, numWires - 1, core->getWireLength(numWires - 1), core->getCoreRadius(), color[0], color[1], color[2], color[3]);

            for (int j = 0; j < numWires - 1; j++) //Restliche Drähte der Kernlitze
            {
                color = core->getWireColor(j);
                fprintf(fileParam, "Wire [S#LRRGBT] = %i, %i, %f, %f, %f, %f, %f, %f\n\n", numWires - 1, j, core->getWireLength(j), core->getCoreRadius(), color[0], color[1], color[2], color[3]);
            }

            //Parameter für die jeweils anderen Litzen
            for (int i = 0; i < numStrands; i++)
            {
                color = strands[i]->getWireColor(numWires - 1);
                fprintf(fileParam, "Core Wire [S#LRRGBT] = %i, %i, %f, %f, %f, %f, %f, %f\n\n", i, numWires - 1, strands[i]->getWireLength(numWires - 1), core->getCoreRadius(), color[0], color[1], color[2], color[3]);

                for (int j = 0; j < numWires - 1; j++)
                {
                    color = strands[i]->getWireColor(j);
                    fprintf(fileParam, "Wire [S#LRRGBT] = %i, %i, %f, %f, %f, %f, %f, %f\n\n", i, j, strands[i]->getWireLength(j), core->getCoreRadius(), color[0], color[1], color[2], color[3]);
                }
            }

            fclose(fileParam);
        }
    }

    if (tUIItem == LoadButton)
    {
        const int maxchar = 1000; // Anzahl maximal gelesener Zeichen
        //char buf[maxchar]; // Puffer für gelesene Zeichen
        char *buf = new char[maxchar];

        // Flags für Interpretation des gelesenen Inhalts
        int counter = 0;
        //int count = 0;
        int lese_global = 0, lese_wires = 0, lese_lengthfactorstrands = 0;
        //float gefunden = 0.0f;
        float lotos = 0.0f; //Length of Twist of Strand
        float lotor = 0.0f; //Length of Twist of Rope
        float lr = 0.0f; //Length Rope
        float nStrand = 0.0f; //Nummer der jeweiligen Litze
        float lfcs = 0.0f; //Length factor of Core Strand
        float nWires = 0.0f; //Nummer des jeweiligen Drahtes
        float lfow = 0.0f; //Length factor of Wire
        float r = 0.0f;
        (void)r; //Radius der Drähte
        float colorred = 0.0f; //Farbanteil von Rot
        float colorgreen = 0.0f; //Farbanteil von Gruen
        float colorblue = 0.0f; //Farbanteil von Blau
        float colorT = 0.0f; //Hell-, Dunkelanteil

        // Datei öffnen (Lesezugriff)
        if ((fileParam = fopen("C:\\TEMP\\Testname.txt", "r")) != NULL)
        {
            // Schleife über gelesene Zeilen
            while ((buf = fgets(buf, maxchar, fileParam)))
            {
                // Auswerten von Kennwörtern für Sektionen und setzen der entsprechenden Flags
                if (strcmp("[Global Parameters]\n", buf) == 0) // falls Zeichenketten identisch
                {
                    lese_global = 1;
                    lese_wires = 0;
                    lese_lengthfactorstrands = 0;
                    counter = 0;
                    continue;
                }
                else if (strcmp("[Length factor of Strands]\n", buf) == 0) // falls Zeichenketten identisch
                {
                    lese_global = 0;
                    lese_wires = 0;
                    lese_lengthfactorstrands = 1;
                    counter = 0;
                    continue;
                }
                else if (strcmp("[Wires]\n", buf) == 0) // falls Zeichenketten identisch
                {
                    lese_global = 0;
                    lese_wires = 1;
                    lese_lengthfactorstrands = 0;
                    counter = 0;
                    continue;
                }

                // Auslesen der Informationen und speichern in Variablen
                if (lese_global == 1)
                {
                    if (counter == 0) //  Length of Twist of Strand
                    {
                        char *buf2;
                        buf2 = buf;
                        while (buf2[0] != '\n')
                        {
                            if (buf2[0] == '=')
                            {
                                buf2 += 2; // gleichbedeutend mit: buf2 = buf2 + 2;
                                lotos = atof(buf2);
                                counter++;
                                break;
                            }
                            buf2++;
                        }
                        continue;
                    }
                    if (counter == 1) //Length of Twist of Rope
                    {
                        char *buf2;
                        buf2 = buf;
                        while (buf2[0] != '\n')
                        {
                            if (buf2[0] == '=')
                            {
                                buf2 += 2; // gleichbedeutend mit: buf2 = buf2 + 2;
                                lotor = atof(buf2);
                                counter++;
                                break;
                            }
                            buf2++;
                        }
                        continue;
                    }
                    if (counter == 2) //Length Rope
                    {
                        char *buf2;
                        buf2 = buf;
                        while (buf2[0] != '\n')
                        {
                            if (buf2[0] == '=')
                            {
                                buf2 += 2; // gleichbedeutend mit: buf2 = buf2 + 2;
                                lr = atof(buf2);
                                break;
                            }
                            buf2++;
                        }
                    }
                }

                if (lese_lengthfactorstrands == 1) //Längenfaktoren der einzelnen Litzen
                {
                    char *buf2;
                    buf2 = buf;
                    while (buf2[0] != '\n')
                    {
                        if (buf2[0] == '=')
                        {
                            buf2 += 2; // gleichbedeutend mit: buf2 = buf2 + 2;
                            nStrand = atof(buf2);
                        }

                        if (buf2[0] == ',')
                        {
                            buf2 += 2; // gleichbedeutend mit: buf2 = buf2 + 2;
                            lfcs = atof(buf2);
                        }
                        buf2++;
                    }
                    if (nStrand == 6)
                    {
                        core->setlength(lfcs);
                    }
                    else
                    {
                        for (int i = 0; i < numStrands; i++)
                        {
                            if (i == nStrand)
                                strands[i]->setlength(lfcs);
                        }
                    }
                    continue;
                }

                if (lese_wires == 1) //Daten der einzelnen Litzen
                {
                    counter = 0;
                    char *buf2;
                    buf2 = buf;
                    while (buf2[0] != '\n')
                    {
                        if (buf2[0] == '=') //Litzennummer
                        {
                            buf2 += 2; // gleichbedeutend mit: buf2 = buf2 + 2;
                            nStrand = atof(buf2);
                            counter++;
                        }

                        else if (buf2[0] == ',' && counter == 1) //Drahtnummer
                        {
                            buf2 += 2; // gleichbedeutend mit: buf2 = buf2 + 2;
                            nWires = atof(buf2);
                            counter++;
                        }

                        else if (buf2[0] == ',' && counter == 2) //Längenfaktor der Drähte
                        {
                            buf2 += 2; // gleichbedeutend mit: buf2 = buf2 + 2;
                            lfow = atof(buf2);
                            counter++;
                        }

                        else if (buf2[0] == ',' && counter == 3) //Drahtradius
                        {
                            buf2 += 2; // gleichbedeutend mit: buf2 = buf2 + 2;
                            r = atof(buf2);
                            counter++;
                        }

                        else if (buf2[0] == ',' && counter == 4) //Rotanteil
                        {
                            buf2 += 2; // gleichbedeutend mit: buf2 = buf2 + 2;
                            colorred = atof(buf2);
                            counter++;
                        }

                        else if (buf2[0] == ',' && counter == 5) //Gelbanteil
                        {
                            buf2 += 2; // gleichbedeutend mit: buf2 = buf2 + 2;
                            colorgreen = atof(buf2);
                            counter++;
                        }
                        else if (buf2[0] == ',' && counter == 6) //Blauanteil
                        {
                            buf2 += 2; // gleichbedeutend mit: buf2 = buf2 + 2;
                            colorblue = atof(buf2);
                            counter++;
                        }
                        else if (buf2[0] == ',' && counter == 7) //Hell-, Dunkelanteil
                        {
                            buf2 += 2; // gleichbedeutend mit: buf2 = buf2 + 2;
                            colorT = atof(buf2);
                            counter++;
                        }

                        buf2++;
                    }
                    if (nStrand == numStrands) //Auswahl der Kernlitze, wenn strandI=numStrands also =6 ist
                    {
                        core->setwirelength((int)nWires, lfow);
                        core->setwirecolor((int)nWires, Vec4(colorred, colorgreen, colorblue, colorT));
                        core->setropelength(lr);
                        core->setLengthOfTwist(lotos);
                        core->createGeom();
                    }

                    else //Auswahl der anderen Litzen, wenn strandI andere Werte als 6 annimmt
                    {
                        for (int i = 0; i < numStrands; i++)
                        {
                            if (i == nStrand)
                            {
                                strands[i]->setwirelength((int)nWires, lfow);
                                strands[i]->setropelength(lr);
                                strands[i]->setLengthOfTwist(lotos);
                                strands[i]->createGeom();
                                strands[i]->rotate((2.0 * M_PI / numStrands) * i, lotor, 60);
                                strands[i]->setwirecolor((int)nWires, Vec4(colorred, colorgreen, colorblue, colorT));
                            }
                        }
                    }

                    continue;
                }
            }

            // Datei schliessen
            fclose(fileParam);
        }
        else
        {
            printf("Datei Testname kann nicht eroeffnet werden\n");
        }

        delete[] buf;
    }
}

void SimpleRopePlugin::deleteColorMap(const std::string &name)
{
    float *mval = mapValues[name];
    mapSize.erase(name);
    mapValues.erase(name);
    delete[] mval;
}

//------------------------------------------------------------------------------
//
// read colormaps from xml config file
// read local colormaps
//------------------------------------------------------------------------------
void SimpleRopePlugin::readConfig()
{
    covise::coConfig *config = covise::coConfig::getInstance();

    // read the name of all colormaps in file
    auto list = config->getVariableList("Colormaps").entries();
    for (const auto &e : list)
        mapNames.insert(e.entry);

    // read the values for each colormap
    for (const auto &map : mapNames)
    {
        // get all definition points for the colormap
        auto cmapname = "Colormaps." + map;
        auto variable = config->getVariableList(cmapname).entries();

        mapSize.insert({map, variable.size()});
        float *cval = new float[variable.size() * 5];
        mapValues.insert({map, cval});

        // read the rgbax values
        int it = 0;
        for (int l = 0; l < variable.size() * 5; l = l + 5)
        {
            std::string tmp = cmapname + ".Point:" + std::to_string(it);
            cval[l] = config->getFloat("x", tmp, -1.0);
            if (cval[l] == -1)
            {
                cval[l] = (1.0 / (variable.size() - 1)) * (l / 5);
            }
            cval[l + 1] = config->getFloat("r", tmp, 1.0);
            cval[l + 2] = config->getFloat("g", tmp, 1.0);
            cval[l + 3] = config->getFloat("b", tmp, 1.0);
            cval[l + 4] = config->getFloat("a", tmp, 1.0);
            it++;
        }
    }

    // read values of local colormap files in .covise
    auto place = covise::coConfigDefaultPaths::getDefaultLocalConfigFilePath() + "colormaps";
    QDir directory(place.c_str());
    if (directory.exists())
    {
        QStringList filters;
        filters << "colormap_*.xml";
        directory.setNameFilters(filters);
        directory.setFilter(QDir::Files);
        QStringList files = directory.entryList();

        // loop over all found colormap xml files
        for (int j = 0; j < files.size(); j++)
        {
            covise::coConfigGroup *colorConfig = new covise::coConfigGroup("ColorMap");
            colorConfig->addConfig(place + "/" + files[j].toStdString(), "local", true);

            // read the name of the colormaps
            auto list = colorConfig->getVariableList("Colormaps").entries();

            // loop over all colormaps in one file
            for (const auto &entry : list)
            {

                // remove global colormap with same name
                auto index = mapNames.find(entry.entry);
                if (index != mapNames.end())
                {
                    deleteColorMap(entry.entry);
                }

                // get all definition points for the colormap
                std::string cmapname = "Colormaps." + entry.entry;
                auto variable = colorConfig->getVariableList(cmapname).entries();

                mapSize.insert({entry.entry, variable.size()});
                float *cval = new float[variable.size() * 5];
                mapValues.insert({entry.entry, cval});

                // read the rgbax values
                int it = 0;
                for (int l = 0; l < variable.size() * 5; l = l + 5)
                {
                    std::string tmp = cmapname + ".Point:" + std::to_string(it);
                    cval[l] = std::stof(colorConfig->getValue("x", tmp, " -1.0").entry);
                    if (cval[l] == -1)
                    {
                        cval[l] = (1.0 / (variable.size() - 1)) * (l / 5);
                    }
                    cval[l + 1] = std::stof(colorConfig->getValue("r", tmp, "1.0").entry);
                    cval[l + 2] = std::stof(colorConfig->getValue("g", tmp, "1.0").entry);
                    cval[l + 3] = std::stof(colorConfig->getValue("b", tmp, "1.0").entry);
                    cval[l + 4] = std::stof(colorConfig->getValue("a", tmp, "1.0").entry);
                    it++;
                }
            }
            config->removeConfig(place + "/" + files[j].toStdString());
        }
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Interpolate a cmap to a given number of steps
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Vec4 SimpleRopePlugin::getColor(float pos)
{

    Vec4 actCol;
    int idx = 0;
    // cerr << "name: " << (const char *)mapNames[currentMap].toAscii() << endl;
    // map and mapS were set according to current map, but currentMap was never set
    float *map = mapValues[*mapNames.begin()];
    int mapS = mapSize[*mapNames.begin()];
    if (map == NULL)
    {
        return actCol;
    }
    while (map[(idx + 1) * 5] <= pos)
    {
        idx++;
        if (idx > mapS - 2)
        {
            idx = mapS - 2;
            break;
        }
    }
    double d = (pos - map[idx * 5]) / (map[(idx + 1) * 5] - map[idx * 5]);
    actCol[0] = (float)((1 - d) * map[idx * 5 + 1] + d * map[(idx + 1) * 5 + 1]);
    actCol[1] = (float)((1 - d) * map[idx * 5 + 2] + d * map[(idx + 1) * 5 + 2]);
    actCol[2] = (float)((1 - d) * map[idx * 5 + 3] + d * map[(idx + 1) * 5 + 3]);
    actCol[3] = (float)((1 - d) * map[idx * 5 + 4] + d * map[(idx + 1) * 5 + 4]);

    return actCol;
}
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Interpolate a cmap to a given number of steps
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

SimpleRopePlugin::FlColor *SimpleRopePlugin::interpolateColormap(FlColor *map, int numSteps)
{

    FlColor *actMap = new FlColor[numSteps];
    double delta = 1.0 / (numSteps - 1) * (numColors - 1);
    double x;
    int i;

    delta = 1.0 / (numSteps - 1);
    int idx = 0;
    for (i = 0; i < numSteps - 1; i++)
    {
        x = i * delta;
        while (map[idx + 1][4] <= x)
        {
            idx++;
            if (idx > numColors - 2)
            {
                idx = numColors - 2;
                break;
            }
        }
        double d = (x - map[idx][4]) / (map[idx + 1][0] - map[idx][0]);
        actMap[i][0] = (float)((1 - d) * map[idx][1] + d * map[idx + 1][1]);
        actMap[i][1] = (float)((1 - d) * map[idx][2] + d * map[idx + 1][2]);
        actMap[i][2] = (float)((1 - d) * map[idx][3] + d * map[idx + 1][3]);
        actMap[i][3] = (float)((1 - d) * map[idx][4] + d * map[idx + 1][4]);
        actMap[i][4] = -1;
    }
    actMap[numSteps - 1][0] = map[numColors - 1][0];
    actMap[numSteps - 1][1] = map[numColors - 1][1];
    actMap[numSteps - 1][2] = map[numColors - 1][2];
    actMap[numSteps - 1][3] = map[numColors - 1][3];
    actMap[numSteps - 1][4] = -1;

    return actMap;
}

void SimpleRopePlugin::calcColors()
{
    float values[10000];
    colors->clear();
    float minVal = 10000000, maxVal = -100000000;
    for (int h = 0; h < nheight; h++)
    {
        float area = 0;
        float len = 0;
        for (int s = 0; s < numSegments; s++)
        {

            Vec3 v1, v2, n;
            osg::Vec3Array *v = vert.get();
            if (s == numSegments - 1)
                v1 = (*v)[h * numSegments + s] - (*v)[h * numSegments];
            else
                v1 = (*v)[h * numSegments + s] - (*v)[h * numSegments + s + 1];
            v2 = (*v)[h * numSegments + s];
            v1[2] = 0;
            v2[2] = 0;
            n = v1 ^ v2;
            float a = n.length() / 2;
            float u = 2 * M_PI * (v1 - v2).length();
            len += u;
            area += a;
        }
        if (area > 0)
        {
            values[h] = len / area;
        }
        else
        {
            values[h] = 0;
        }
        if (values[h] < minVal)
            minVal = values[h];
        if (values[h] > maxVal)
            maxVal = values[h];
    }
    if (minVal == maxVal)
        maxVal = minVal + 0.000000000001;
    for (int h = 0; h < nheight; h++)
    {
        //Vec4 c1(1.0,0,0,1.0), c2(0,1.0,0,1.0), c;
        //c = c1*((values[h]-minVal) / (maxVal - minVal) ) + c2*(1.0 -((values[h]-minVal) / (maxVal - minVal)));
        colors->push_back(getColor((values[h] - minVal) / (maxVal - minVal)));
    }
    for (int h = 0; h < nheight; h++)
    {
        for (int s = 0; s < numSegments; s++)
        {
            if (h > 0)
            {
#if 0
            cindices->push_back(h-1);
            cindices->push_back(h-1);
            cindices->push_back(h);
            cindices->push_back(h);
#endif
            }
        }
    }
    geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
}

void
SimpleRopePlugin::preFrame()
{
}

COVERPLUGIN(SimpleRopePlugin)
