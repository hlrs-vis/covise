/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coTrackerButtonInteraction.h>
#include <OpenVRUI/sginterface/vruiIntersection.h>

#include <osg/Switch>
#include <osg/Matrix>
#include <OpenVRUI/osg/mathUtils.h>

#include <cover/coVRLabel.h>
#include <cover/coVRCollaboration.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <cover/VRSceneGraph.h>

#include <PluginUtil/PluginMessageTypes.h>
#include <config/CoviseConfig.h>
#include <net/tokenbuffer.h>

#include "Measure.h"

using covise::coCoviseConfig;
using covise::TokenBuffer;

/**
	@author Tim Kang
	@author Jurgen Schulze
	@date April 2011

	Measure is a COVISE plugin used to create a virtual measuring tape.
	You are able to set the initial coneSize and textSize by adding those values
	into the config XML file.
*/

Measure *Measure::plugin = NULL;
osg::ref_ptr<osg::Material> Dimension::globalWhitemtl;
osg::ref_ptr<osg::Material> Dimension::globalRedmtl;
coCheckboxMenuItem *linearItem; // Checkbox used to select whether tape is active
double coneSize = 150; // Variable for initial multiple factor of coneSize
double fontFactor = 3; // Variable for initial multiple factor of textSize
double lineWidth = 28; // Variable for initial multiple factor of lineWidth
bool checkboxTog = true; //horrible hack for checkbox

std::vector<coCheckboxMenuItem *> checkboxArray; // used for units
std::vector<std::string> unitArray; // used for printf format strings
std::vector<float> scaleArray; // used for scale factors

LinearDimension::LinearDimension(int idParam, Measure *m)
    : Dimension(idParam, m)
{

    if (globalWhitemtl.get() == NULL)
    {
        osg::Material *globalWhitemtl = new osg::Material;
        globalWhitemtl->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
        globalWhitemtl->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.2, 0.2, 0.2, 1.0));
        globalWhitemtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1, 1, 1, 1.0));
        globalWhitemtl->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
        globalWhitemtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0, 0, 0, 1.0));
        globalWhitemtl->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);
    }
    if (globalRedmtl.get() == NULL)
    {
        osg::Material *globalRedmtl = new osg::Material;
        globalRedmtl->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
        globalRedmtl->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.2, 0, 0, 1.0));
        globalRedmtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1, 0, 0, 1.0));
        globalRedmtl->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
        globalRedmtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0, 0, 0, 1.0));
        globalRedmtl->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);
    }

    osg::Geode *geodeCyl = new osg::Geode;
    osg::ShapeDrawable *sd;
    sd = new osg::ShapeDrawable(new osg::Cylinder(osg::Vec3(0, 0, 0.5), 1, 1));
    osg::StateSet *ss = sd->getOrCreateStateSet();
    geodeCyl->addDrawable(sd);
    ss->setAttributeAndModes(globalWhitemtl.get(), osg::StateAttribute::ON);

    geos->addChild(geodeCyl);

    geodeCyl = new osg::Geode;
    sd = new osg::ShapeDrawable(new osg::Cylinder(osg::Vec3(0, 0, 0.5), 1, 1));
    ss = sd->getOrCreateStateSet();
    geodeCyl->addDrawable(sd);
    ss->setAttributeAndModes(globalRedmtl.get(), osg::StateAttribute::ON);

    geos->addChild(geodeCyl);

    line = new osg::MatrixTransform;

    line->addChild(geos);
    cover->getObjectsRoot()->addChild(line.get());
}

LinearDimension::~LinearDimension()
{

    while (line->getNumParents())
        line->getParent(0)->removeChild(line.get());
}

void LinearDimension::update()
{
    Dimension::update();

    osg::Matrix m1;
    osg::Matrix m;
    m.makeIdentity();
    if (marks[0])
        marks[0]->getMat(m1);

    // Used to set correct scaling factor
    osg::Vec3 wpoint1 = osg::Vec3(0, 0, 0);
    osg::Vec3 wpoint2 = osg::Vec3(0, 0, 300);
    osg::Vec3 opoint1 = wpoint1 * cover->getInvBaseMat();
    osg::Vec3 opoint2 = wpoint2 * cover->getInvBaseMat();

    //distance formula
    osg::Vec3 wDiff = wpoint2 - wpoint1;
    osg::Vec3 oDiff = opoint2 - opoint1;
    double distWld = wDiff.length();
    double distObj = oDiff.length();

    double scaleFactor = distObj / ((31 - lineWidth) * distWld);

    double TscaleFactor = distObj / distWld; //scale factor for texts
    osg::Vec3 pos(0, 0, 100 * scaleFactor);
    osg::Vec3 pos2(0, 0, 100 * scaleFactor);
    pos = m1.preMult(pos);
    m.preMult(osg::Matrix::scale(10 * scaleFactor, 10 * scaleFactor, 10));

    osg::Matrix Scale;
    Scale.makeScale(10 * scaleFactor, 10 * scaleFactor, 10);
    osg::Matrix Rot;
    Rot.makeIdentity();
    osg::Matrix Trans;
    osg::Matrix mat;
    osg::Vec3 zAxis(0, 0, 1);
    Trans.makeTranslate(pos[0], pos[1], pos[2]);
    osg::Vec3 dist;
    osg::Vec3 trans2;
    osg::Vec3 transm;

    float len = -1.f;
    if (marks[0] && marks[1])
    {
        osg::Matrix m0, m1;
        marks[0]->getMat(m0);
        marks[1]->getMat(m1);
        osg::Vec3 t0 = m0.getTrans();
        osg::Vec3 t1 = m1.getTrans();
        osg::Vec3 dist = t1 - t0;
        len = dist.length();
        if (cover->debugLevel(0) && len != oldDist)
        {
            fprintf(stderr,"T0: x = %f, y = %f, z=%f \n",t0.x(), t0.y(), t0.z());
            fprintf(stderr,"T1: x = %f, y = %f, z=%f \n",t1.x(), t1.y(), t1.z());
        }
    }

    if (marks[1])
    {
        marks[1]->getMat(m);
        trans2 = m.getTrans();
        m = m1;
        m.setTrans(trans2);
        marks[1]->setPos(m);
        pos2 = m.preMult(pos2);
        dist = pos2 - pos;

        transm = (pos2 + pos) / 2.0;
        m.setTrans(transm);

        m.preMult(osg::Matrix::scale(TscaleFactor * fontFactor, TscaleFactor * fontFactor, TscaleFactor * fontFactor));
        myDCS->setMatrix(m);
        if (len != oldDist)
        {
            for (size_t i = 0; i < checkboxArray.size(); ++i)
            {
                if (checkboxArray[i]->getState())
                {
                    oldDist = len;
                    sprintf(labelString, "%6.3f %s", len / scaleArray[i], unitArray[i].c_str());
                    MakeText();
                    break;
                }
            }
        }
        if (len > 0)
        {
            dist /= len;
        }
        else
        {
            dist[0] = 1.0;
            dist[1] = 0.0;
            dist[2] = 0.0;
        }
        Scale.makeScale(10 * scaleFactor, 10 * scaleFactor, len);
        Rot.makeRotate(zAxis, dist);
        mat = Scale * Rot * Trans;
    }

    line->setMatrix(mat);
}

bool Dimension::isplaced()
{
    if (placedMarks == 0)
        return true;
    int i;
    for (i = 0; i < placedMarks; i++)
    {
        if (marks[i]->placing)
            return true;
    }
    return false;
}

Dimension::Dimension(int idParam, Measure *m)
{
    plugin = m;
    placedMarks = 0;
    id = idParam;
    oldDist = 0.0;
    placing = false;
    int i;
    for (i = 0; i < 2; i++)
    {
        //marks[i] = new Mark(i, this);
		marks[i] = nullptr;
    }
    myDCS = new osg::MatrixTransform();
    geos = new osg::Switch();
    strcpy(labelString, "0.0");
    labelText = NULL;
    cover->getObjectsRoot()->addChild(myDCS);
}

Dimension::~Dimension()
{
    int i;
    for (i = 0; i < placedMarks; i++)
    {
        delete marks[i];
    }
    if (

        myDCS->getNumParents())
        myDCS->getParent(0)->removeChild(myDCS);
}

/// Private method to generate OSG text string and attach it to a node.
void Dimension::MakeText()
{
    if (!labelText)
    {
        labelText = new osgText::Text();

        labelText->setDataVariance(osg::Object::DYNAMIC);
        labelText->setFont(coVRFileManager::instance()->getFontFile(NULL));
        labelText->setDrawMode(osgText::Text::TEXT);

        labelText->setColor(osg::Vec4(1, 1, 1, 1));

        labelText->setAlignment(osgText::Text::CENTER_BASE_LINE);

        labelText->setCharacterSize(40.0);
        labelText->setLayout(osgText::Text::LEFT_TO_RIGHT);
        labelText->setAxisAlignment(osgText::Text::XY_PLANE);

        osg::ref_ptr<osg::Geode> textNode = new osg::Geode();
        textNode->addDrawable(labelText);

        coBillboard *billBoard = new coBillboard();
        billBoard->setMode(coBillboard::AXIAL_ROT);
        billBoard->setMode(coBillboard::POINT_ROT_WORLD);

        osg::Vec3 zaxis(0, 1, 0);
        //  osg::Vec3 zaxis = -cover->getViewerMat().getTrans();
        billBoard->setAxis(zaxis);

        osg::Vec3 normal(0, 0, 1);

        billBoard->setNormal(normal);
        billBoard->addChild(textNode.get());
        myDCS->addChild(billBoard);
    }

    labelText->setText(labelString, osgText::String::ENCODING_UTF8);
}

void Dimension::update()
{
    int i;
    if ((placedMarks < 2) && (marks[placedMarks] == NULL))
    {
        marks[placedMarks] = new Mark(placedMarks, this);
        placedMarks++;
    }
    for (i = 0; i < placedMarks; i++)
    {
        marks[i]->update();
    }
}

Measure::Measure()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    plugin = this;
}

bool Measure::init()
{

    //get the values from the config xml files
    coneSize = coCoviseConfig::getFloat("COVER.Plugin.Measure.ConeSize", 150.0);
    fontFactor = coCoviseConfig::getFloat("COVER.Plugin.Measure.TextSize", 3.0);
    lineWidth = coCoviseConfig::getFloat("COVER.Plugin.Measure.LineWidth", 28.0);

    menuSelected = false;
    // snapToEdges = 99999999;
    maxDimID = 0;
    // get the root node for COVISE objects
    objectsRoot = cover->getObjectsRoot();

    // create a menu entry for this plugin
    createMenuEntry();
    currentMeasure = NULL;
    moving = false;
    interactionA = new coTrackerButtonInteraction(coInteraction::ButtonA, "MeasurePlacement", coInteraction::Medium);

    return true;
}

// this is called if the plugin is removed at runtime
Measure::~Measure()
{
    for (dims.reset(); dims.current(); dims.remove())
        ;
    // remove the menu entry for this plugin
    removeMenuEntry();
    delete interactionA;
}

void Mark::update()
{

    resize();
    TokenBuffer tb;
    if ((placing) || (moveMarker))
    {
        if (!interactionA->isRegistered())
        {
            coInteractionManager::the()->registerInteraction(interactionA);
        }
    }
    else
    {
        if (interactionA->isRegistered())
        {
            coInteractionManager::the()->unregisterInteraction(interactionA);
        }
    }
    if (placing)
    {
        if (interactionA->isRegistered())
        {
            tb << MOVE_MARK;
            tb << dim->getID();
            tb << id;
            osg::Matrix trans;
            osg::Matrix mat;
            trans.makeTranslate(0, 500, 0);
            mat = trans * cover->getPointerMat() * cover->getInvBaseMat();

            int i, j;
            for (i = 0; i < 4; i++)
            {
                for (j = 0; j < 4; j++)
                {
                    tb << (float)mat(i, j);
                }
            }
            cover->sendMessage(Measure::plugin,
                               coVRPluginSupport::TO_SAME,
                               PluginMessageTypes::Measure0,
                               tb.getData().length(),
                               tb.getData().data());
        }
        if (interactionA->wasStarted()) // button pressed
        {
            checkboxTog = !checkboxTog;
            if (checkboxTog)
            {
                linearItem->setState(false);
            }
            placing = false;
            if (interactionA->isRegistered())
            {
                coInteractionManager::the()->unregisterInteraction(interactionA);
            }
        }
    }
    else if (moveMarker)
    {
        if (interactionA->isRegistered())
        {
            if (interactionA->wasStarted()) // button pressed
            {
                invStartHand.invert(cover->getPointerMat());
                startPos = pos->getMatrix() * cover->getBaseMat();
                moveStarted = true;
            }
            if (interactionA->isRunning())
            {
                if (!moveStarted)
                {
                    invStartHand.invert(cover->getPointerMat());
                    startPos = pos->getMatrix() * cover->getBaseMat();
                }
                moveStarted = true;
                tb << MOVE_MARK;
                tb << dim->getID();
                tb << id;
                osg::Matrix dMat = invStartHand * cover->getPointerMat();
                osg::Matrix current;
                osg::Matrix tmp;
                tmp = startPos * dMat;
                current = tmp * cover->getInvBaseMat();

                int i, j;
                for (i = 0; i < 4; i++)
                {
                    for (j = 0; j < 4; j++)
                    {
                        tb << (float)current(i, j);
                    }
                }
                cover->sendMessage(Measure::plugin,
                    coVRPluginSupport::TO_SAME,
                    PluginMessageTypes::Measure0,
                    tb.getData().length(),
                    tb.getData().data());
            }
            if (interactionA->wasStopped())
            {
                if (moveStarted)
                {
                    moveMarker = false;
                    moveStarted = false;
                    if (interactionA->isRegistered())
                    {
                        coInteractionManager::the()->unregisterInteraction(interactionA);
                    }

                    setIcon(0);
                }
            }
        }
    }
}

Mark::Mark(int i, Dimension *d)
{
    id = i;
    dim = d;

    placing = true;
    moveMarker = false;
    moveStarted = false;
    pos = new osg::MatrixTransform;
    sc = new osg::MatrixTransform;
    pos->addChild(sc);
    icons = new osg::Switch();
    sc->addChild(icons);
    geo = coVRFileManager::instance()->loadIcon("marker");
    icons->addChild(geo);
    geo = coVRFileManager::instance()->loadIcon("marker2");
    icons->addChild(geo);
    icons->setSingleChildOn(0);
    cover->getObjectsRoot()->addChild(pos);
    vNode = new OSGVruiNode(pos);
    vruiIntersection::getIntersectorForAction("coAction")->add(vNode, this);
    interactionA = new coTrackerButtonInteraction(coInteraction::ButtonA, "MarkPlacement", coInteraction::Medium);
}

Mark::~Mark()
{
    vruiIntersection::getIntersectorForAction("coAction")->remove(vNode);
    pos->getParent(0)->removeChild(pos);
    delete vNode;
    delete interactionA;
}

/**
@param hitPoint,hit  Performer intersection information
@return ACTION_CALL_ON_MISS if you want miss to be called,
otherwise ACTION_DONE is returned
*/
int Mark::hit(vruiHit *)
{
    if ((coVRCollaboration::instance()->getCouplingMode() == coVRCollaboration::MasterSlaveCoupling
         && !coVRCollaboration::instance()->isMaster())
        || placing)
        return ACTION_CALL_ON_MISS;

    moveMarker = true;
    setIcon(1);

    return ACTION_CALL_ON_MISS;
}

/// Miss is called once after a hit, if the button is not intersected anymore.
void Mark::miss()
{

    if (!interactionA->isRunning())
    {
        moveMarker = false;
        moveStarted = false;
        setIcon(0);
    }
}

void Mark::setIcon(int i)
{
    icons->setSingleChildOn(i);
}

void Mark::resize()
{
    osg::Vec3 wpoint1 = osg::Vec3(0, 0, 0);
    osg::Vec3 wpoint2 = osg::Vec3(0, 0, 300);
    osg::Vec3 opoint1 = wpoint1 * cover->getInvBaseMat();
    osg::Vec3 opoint2 = wpoint2 * cover->getInvBaseMat();

    //distance formula
    osg::Vec3 wDiff = wpoint2 - wpoint1;
    osg::Vec3 oDiff = opoint2 - opoint1;
    double distWld = wDiff.length();
    double distObj = oDiff.length();

    //controls the cone size
    double scaleFactor = (coneSize / 150) * distObj / distWld;

    //sc->setMatrix controls the size of the markers
    sc->setMatrix(osg::Matrix::scale(scaleFactor, scaleFactor, scaleFactor));
}

void Mark::setPos(osg::Matrix &mat)
{
    coCoord c;
    c = mat;
    c.makeMat(mat);
    pos->setMatrix(mat);
    //mat.print(1,1,"coorded mat:",stderr);
    resize();
}

/** Opencover calls this function before each frame is rendered */
void Measure::preFrame()
{
    for (dims.reset(); dims.current(); dims.next())
    {

        dims.current()->update();
    }
}

void Mark::getMat(osg::Matrix &m)
{
    m = pos->getMatrix();
}

void
Measure::setCurrentMeasure(Mark *m)
{
    currentMeasure = m;
}

void
Measure::createMenuEntry()
{

    measureMenuItem = new coSubMenuItem("Measure");
    measureMenu = new coRowMenu("Measure");
    measureMenuItem->setMenu(measureMenu);

    linearItem = new coCheckboxMenuItem("Tape Measure", false);
    linearItem->setMenuListener(this);
    measureMenu->add(linearItem);

    /*snapItem = new coCheckboxMenuItem("Snap To Edges",false);
   snapItem->setMenuListener(this);
   measureMenu->add(snapItem);
   */
    //To set the units of measurement
    unitsMenuItem = new coSubMenuItem("Units");
    unitsMenu = new coRowMenu("Units");
    unitsMenuItem->setMenu(unitsMenu);
    measureMenu->add(unitsMenuItem);

    checkboxArray.push_back(new coCheckboxMenuItem("Mikrometer", false));
    scaleArray.push_back(1.e-6f);
    unitArray.push_back("um");

    checkboxArray.push_back(new coCheckboxMenuItem("Millimeter", false));
    scaleArray.push_back(1.e-3f);
    unitArray.push_back("mm");

    checkboxArray.push_back(new coCheckboxMenuItem("Centimeter", false));
    scaleArray.push_back(1.e-2f);
    unitArray.push_back("cm");

    checkboxArray.push_back(new coCheckboxMenuItem("Meter", true));
    scaleArray.push_back(1.f);
    unitArray.push_back("m");

    checkboxArray.push_back(new coCheckboxMenuItem("Inch", false));
    scaleArray.push_back(0.0254f);
    unitArray.push_back("in");

    checkboxArray.push_back(new coCheckboxMenuItem("Foot", false));
    scaleArray.push_back(0.0254f * 12.f);
    unitArray.push_back("ft");

    checkboxArray.push_back(new coCheckboxMenuItem("Yard", false));
    scaleArray.push_back(0.0254f * 12.f * 3.f);
    unitArray.push_back("yd");

    for (size_t i = 0; i < checkboxArray.size(); ++i)
    {
        checkboxArray[i]->setMenuListener(this);
        unitsMenu->add(checkboxArray[i]);
    }

    markerScalePoti = new coPotiMenuItem("Cone Size", 0, 600, 150);
    markerScalePoti->setMenuListener(this);
    measureMenu->add(markerScalePoti);

    fontScalePoti = new coPotiMenuItem("Font Size", 0, 10, 3);
    fontScalePoti->setMenuListener(this);
    measureMenu->add(fontScalePoti);

    lineWidthPoti = new coPotiMenuItem("Line Width", 1, 30, 28);
    lineWidthPoti->setMenuListener(this);
    measureMenu->add(lineWidthPoti);

    clearItem = new coButtonMenuItem("Clear All");
    clearItem->setMenuListener(this);
    measureMenu->add(clearItem);

    cover->getMenu()->add(measureMenuItem);
}

void
Measure::removeMenuEntry()
{
    delete measureMenu;
    delete measureMenuItem;
    delete fontScalePoti;
    delete lineWidthPoti;
    delete markerScalePoti;
    delete clearItem;
    //delete snapItem;
    delete linearItem;
    delete unitsMenu;
    delete unitsMenuItem;

    for (size_t i = 0; i < checkboxArray.size(); ++i)
        delete checkboxArray[i];
}

void Measure::menuEvent(coMenuItem *item)
{

    if (coVRCollaboration::instance()->getCouplingMode() == coVRCollaboration::MasterSlaveCoupling
        && !coVRCollaboration::instance()->isMaster())
        return;
    TokenBuffer tb;
    if (item == linearItem)
    {
        if (linearItem->getState())
        {
            tb << NEW_DIMENSION;
            tb << maxDimID;
            cover->sendMessage(this,
                coVRPluginSupport::TO_SAME,
                PluginMessageTypes::Measure0,
                tb.getData().length(),
                tb.getData().data());
        }
        else
        {
            dims.removeLast();
            checkboxTog = true;
        }
    }

    if (item == clearItem)
    {
        for (dims.reset(); dims.current();)
        {
            dims.remove();
        }
    }

    if (item == markerScalePoti)
    {
        coneSize = markerScalePoti->getValue();
    }

    if (item == fontScalePoti)
    {
        fontFactor = fontScalePoti->getValue();
    }

    if (item == lineWidthPoti)
    {
        double maxLineWidth = 23.0 + log(coneSize) / log(2.0);
        double oldLineWidth = lineWidthPoti->getValue();

        if (lineWidthPoti->getValue() > maxLineWidth)
        {
            lineWidthPoti->setValue(oldLineWidth);
        }
        else
        {
            lineWidth = lineWidthPoti->getValue();
        }
    }

    /*if(item == snapItem)
   {
   
   }*/

    //For the units
    for (size_t i = 0; i < checkboxArray.size(); ++i)
    {

        if (item == checkboxArray[i])
        {
            for (size_t j = 0; j < checkboxArray.size(); ++j)
            {
                checkboxArray[j]->setState(i == j);
            }
        }
    }
}

void Measure::message(int toWhom, int type, int len, const void* buf)
{

    if (type != PluginMessageTypes::Measure0 && type != PluginMessageTypes::Measure1)
        return;

    TokenBuffer tb{covise::DataHandle{(char*)buf, len, false}};
    char msgType;
    tb >> msgType;
    switch (msgType)
    {
    case NEW_DIMENSION:
    {
        int id;
        tb >> id;
        maxDimID++;
        dims.append(new LinearDimension(id, this));
    }
    break;
    case MOVE_MARK:
    {
        int Did;
        tb >> Did;
        int Mid;
        tb >> Mid;
        int i, j;
        float f;
        osg::Matrix mat;
        for (i = 0; i < 4; i++)
        {
            for (j = 0; j < 4; j++)
            {
                tb >> f;
                mat(i, j) = f;
            }
        }
        Dimension *dim = NULL;
        for (dims.reset(); dims.current(); dims.next())
        {
            if (dims.current()->getID() == Did)
            {
                dim = dims.current();
                break;
            }
        }
		if (dim)
		{
			if (!dim->marks[Mid])
			{
				dim->marks[Mid] = new Mark(Mid, dim);
			}
			dim->marks[Mid]->setPos(mat);
		}
            
    }
    break;
    case REMOVE: // Remove
    {
        marker.reset();
        while (marker.current())
        {
            int id;
            tb >> id;
            coInteractionManager::the()->registerInteraction(interactionA);
            marker.next();
        }
    }
    break;
    }
}

COVERPLUGIN(Measure)
