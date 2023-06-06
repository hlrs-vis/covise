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
#include <osg/io_utils>
#include <OpenVRUI/osg/mathUtils.h>

#include <osgDB/Registry>
#include <osgDB/ReadFile>

#include <cover/coVRLabel.h>
#include <cover/coVRCollaboration.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <cover/VRSceneGraph.h>

#include <PluginUtil/PluginMessageTypes.h>
#include <config/CoviseConfig.h>
#include <net/tokenbuffer.h>

#include <math.h>

#include "BulletProbe.h"

using covise::coCoviseConfig;
using covise::TokenBuffer;

BulletProbe *BulletProbe::plugin = NULL;

osg::ref_ptr<osg::Material> Dimension::globalWhitemtl;
//osg::ref_ptr<osg::Material> Dimension::globalRedmtl;

coCheckboxMenuItem *linearItem;
double coneSize = 0.01; // Variable for initial multiple factor of coneSize
//double fontFactor = 3; // Variable for initial multiple factor of textSize
double lineWidth = 15; // Variable for initial multiple factor of lineWidth
bool checkboxTog = true; //horrible hack for checkbox

std::vector<coCheckboxMenuItem *> checkboxArray; // used for units
std::vector<std::string> unitArray; // used for printf format strings
std::vector<float> scaleArray; // used for scale factors

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
Dimension::Dimension(int idParam, BulletProbe *m)
    : plugin(m),
      placedMarks(0),
      id(idParam),
      oldDist(0.0),
      placing(false),
      fExtraLength(1.0)
{
    marks[0] = nullptr;
    marks[1] = nullptr;

    myDCS = new osg::MatrixTransform();
    geos = new osg::Switch();
//    strcpy(labelString, "0.0");
//    labelText = NULL;
    cover->getObjectsRoot()->addChild(myDCS);
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
Dimension::~Dimension()
{
    int i;
    for (i = 0; i < placedMarks; i++)
    {
        delete marks[i];
    }
    if (myDCS->getNumParents())
    {
        myDCS->getParent(0)->removeChild(myDCS);
    }
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void Dimension::create()
{
    marks[0]->placing = false;
    marks[1]->placing = false;
    placedMarks = 2;
}

// ----------------------------------------------------------------------------
//! true if a marker is currently being placed
// ----------------------------------------------------------------------------
bool Dimension::isplaced()
{
    if (placedMarks == 0)
    {
        return true;
    }
    
    int i;
    for (i = 0; i < placedMarks; i++)
    {
        if (marks[i]->placing)
        {
            return true;
        }
    }
    return false;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void Dimension::update()
{
    int i;
    if ((placedMarks < 2) && (marks[placedMarks] == nullptr))
    {
        marks[placedMarks] = new Mark(placedMarks, this);
        placedMarks++;
    }
    for (i = 0; i < placedMarks; i++)
    {
        marks[i]->update();
    }
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
int Dimension::getID()
{
    return id;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void Dimension::setVisible(bool visibleOnOff)
{
    if (visibleOnOff)
    {
        geos->setNodeMask(~0);
    }
    else
    {
        geos->setNodeMask(0);
    }

    if (marks[0])
    {
        marks[0]->setVisible(visibleOnOff);
    }
    if (marks[1])
    {
        marks[1]->setVisible(visibleOnOff);
    }
}

// ----------------------------------------------------------------------------
//! generate OSG text string and attach it to a node.
// ----------------------------------------------------------------------------
//void Dimension::makeText()
//{
    // if (!labelText)
    // {
    //     labelText = new osgText::Text();

    //     labelText->setDataVariance(osg::Object::DYNAMIC);
    //     labelText->setFont(coVRFileManager::instance()->getFontFile(NULL));
    //     labelText->setDrawMode(osgText::Text::TEXT);

    //     labelText->setColor(osg::Vec4(1, 1, 1, 1));

    //     labelText->setAlignment(osgText::Text::CENTER_BASE_LINE);

    //     labelText->setCharacterSize(40.0);
    //     labelText->setLayout(osgText::Text::LEFT_TO_RIGHT);
    //     labelText->setAxisAlignment(osgText::Text::XY_PLANE);

    //     osg::ref_ptr<osg::Geode> textNode = new osg::Geode();

    //     textNode->addDrawable(labelText);qqqqq

    //     coBillboard *billBoard = new coBillboard();
    //     billBoard->setMode(coBillboard::AXIAL_ROT);
    //     billBoard->setMode(coBillboard::POINT_ROT_WORLD);

    //     osg::Vec3 zaxis(0, 1, 0);
    //     //  osg::Vec3 zaxis = -cover->getViewerMat().getTrans();
    //     billBoard->setAxis(zaxis);

    //     osg::Vec3 normal(0, 0, 1);

    //     billBoard->setNormal(normal);
    //     billBoard->addChild(textNode.get());
    //     myDCS->addChild(billBoard);
    // }

//    labelText->setText(labelString, osgText::String::ENCODING_UTF8);
//}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
LinearDimension::LinearDimension(int idParam, BulletProbe *m)
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
    // if (globalRedmtl.get() == NULL)
    // {
    //     osg::Material *globalRedmtl = new osg::Material;
    //     globalRedmtl->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    //     globalRedmtl->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0, 0, 0, 1.0));
    //     globalRedmtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1, 0, 0, 1.0));
    //     globalRedmtl->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
    //     globalRedmtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0, 0, 0, 1.0));
    //     globalRedmtl->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);
    // }

    osg::ShapeDrawable *sd;
//    sd = new osg::ShapeDrawable(new osg::Cylinder(osg::Vec3(0, 0, 0.5), 10, 10));

    osg::Geode *geodeCyl;// = new osg::Geode;    
//    geodeCyl->addDrawable(sd);

    osg::StateSet *ss;// = sd->getOrCreateStateSet();
//    ss->setAttributeAndModes(globalWhitemtl.get(), osg::StateAttribute::ON);

//    geos->addChild(geodeCyl);

    geodeCyl = new osg::Geode;
    osg::StateSet* stateset = new osg::StateSet();
    
    osg::Texture2D* texture = coVRFileManager::instance()->loadTexture("redwhite_multi");

    // texture->setWrap( osg::Texture::WRAP_S, osg::Texture::REPEAT );
    // texture->setWrap( osg::Texture::WRAP_T, osg::Texture::REPEAT );
    stateset->setTextureAttributeAndModes(0,texture,osg::StateAttribute::ON);
    geodeCyl->setStateSet( stateset );

    sd = new osg::ShapeDrawable(new osg::Cylinder(osg::Vec3(0.0, 0.0, 0.5), 1, 1)); // TODO

    osg::ref_ptr<osg::Vec2Array> texcoords = new osg::Vec2Array;

    // texcoords->push_back( osg::Vec2(0.5f * cos(i) + 0.5f, 0.5f * sin(i) + 0.5f));
    // sd->setTexCoordArray(0, texcoords.get());
    
    ss = sd->getOrCreateStateSet();
    geodeCyl->addDrawable(sd);
    ss->setAttributeAndModes(globalWhitemtl.get(), osg::StateAttribute::ON);

    geos->addChild(geodeCyl);

    line = new osg::MatrixTransform;

    line->addChild(geos);
    cover->getObjectsRoot()->addChild(line.get());
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
LinearDimension::~LinearDimension()
{
    while (line->getNumParents())
    {
        line->getParent(0)->removeChild(line.get());
    }
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
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
    osg::Vec3 pos(0, 0, 0);//100 * scaleFactor);
    osg::Vec3 pos2(0, 0, 0);//100 * scaleFactor);
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
        // if (cover->debugLevel(0) && len != oldDist)
        // {
        //     fprintf(stderr,"BulletProbe PlugIn T0: x = %f, y = %f, z=%f \n",t0.x(), t0.y(), t0.z());
        //     fprintf(stderr,"                   T1: x = %f, y = %f, z=%f \n",t1.x(), t1.y(), t1.z());
        //     fprintf(stderr,"                   length = %f \n",len);
        // }
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
        
        // double fontFactor = 1.0;
        // m.preMult(osg::Matrix::scale(TscaleFactor * fontFactor,
        //                              TscaleFactor * fontFactor,
        //                              TscaleFactor * fontFactor));
        
        myDCS->setMatrix(m);
         if (len != oldDist)
         {
             for (size_t i = 0; i < checkboxArray.size(); ++i)
             {
                 if (checkboxArray[i]->getState())
                 {
                     oldDist = len;
                     // sprintf(labelString, "%6.3f %s", len / scaleArray[i],
                     //         unitArray[i].c_str());
                     //makeText();
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
         Scale.makeScale(10 * scaleFactor, 10 * scaleFactor,
                         len * fExtraLength); // set new length
//         Scale.makeScale(1, 1, len); // set new length
         Rot.makeRotate(zAxis, dist);
         mat = Scale * Rot * Trans;
    }

    line->setMatrix(mat);
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void LinearDimension::setExtraLength(float _fExtraLength)
{
    fExtraLength = _fExtraLength;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
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

    geo = coVRFileManager::instance()->loadIcon("sphere2");
    icons->addChild(geo);
    geo = coVRFileManager::instance()->loadIcon("sphere");
    icons->addChild(geo);
    icons->setSingleChildOn(0);
    cover->getObjectsRoot()->addChild(pos);
    vNode = new OSGVruiNode(pos);
    vruiIntersection::getIntersectorForAction("coAction")->add(vNode, this);
    interactionA = new coTrackerButtonInteraction(coInteraction::ButtonA,
                                                  "MarkPlacement",
                                                  coInteraction::Medium);
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
Mark::~Mark()
{
    vruiIntersection::getIntersectorForAction("coAction")->remove(vNode);
    pos->getParent(0)->removeChild(pos);
    
    delete vNode;
    delete interactionA;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
int Mark::hit(vruiHit *)
{
    if ((coVRCollaboration::instance()->getCouplingMode()
         == coVRCollaboration::MasterSlaveCoupling
         && !coVRCollaboration::instance()->isMaster())
        || placing)
    {
        return ACTION_CALL_ON_MISS;
    }

    moveMarker = true;
    setIcon(1);

    return ACTION_CALL_ON_MISS;
}

// ----------------------------------------------------------------------------
//! Miss is called once after a hit, if the button is not intersected anymore.
// ----------------------------------------------------------------------------
void Mark::miss()
{
    if (!interactionA->isRunning())
    {
        moveMarker = false;
        moveStarted = false;
        setIcon(0);
    }
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
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
            cover->sendMessage(BulletProbe::plugin,
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

                cover->sendMessage(BulletProbe::plugin,
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

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
int Mark::getID()
{
    return id;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void Mark::setPos(osg::Matrix &mat)
{
    coCoord c;
    c = mat;
    c.makeMat(mat);

    pos->setMatrix(mat);
    //mat.print(1,1,"coorded mat:",stderr);
    resize();
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
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

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void Mark::getMat(osg::Matrix &m)
{
    m = pos->getMatrix();
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void Mark::setIcon(int i)
{
    icons->setSingleChildOn(i);
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void Mark::create()
{
    if (!interactionA->isRegistered())
    {
        coInteractionManager::the()->registerInteraction(interactionA);
    }
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void Mark::setVisible(bool visibleOnOff)
{
    if (visibleOnOff)
    {
        icons->setNodeMask(~0);
    }
    else
    {
        icons->setNodeMask(0);
    }
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
BulletProbe::BulletProbe()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    plugin = this;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
BulletProbe::~BulletProbe()
{
    for (dims.reset(); dims.current(); dims.remove());

    removeMenuEntry();

    delete interactionA;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
bool BulletProbe::init()
{
    //get the values from the config xml files
    coneSize = coCoviseConfig::getFloat("COVER.Plugin.BulletProbe.ConeSize", 150.0);
    // fontFactor = coCoviseConfig::getFloat("COVER.Plugin.BulletProbe.TextSize", 3.0);
    lineWidth = coCoviseConfig::getFloat("COVER.Plugin.BulletProbe.LineWidth", 27.0);

    maxDimID = 0;
    
    // get the root node for COVISE objects
    objectsRoot = cover->getObjectsRoot();

    createMenuEntry();

    currentProbe = NULL;
    moving = false;
    interactionA = new coTrackerButtonInteraction(coInteraction::ButtonA,
                                                  "MeasurePlacement",
                                                  coInteraction::Medium);
    return true;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void BulletProbe::preFrame()
{
    for (dims.reset(); dims.current(); dims.next())
    {
        dims.current()->update();
    }
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void BulletProbe::message(int toWhom, int type, int len, const void* buf)
{
    if (type != PluginMessageTypes::Measure0
        && type != PluginMessageTypes::Measure1)
    {
        return;
    }

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

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void BulletProbe::setCurrentMeasure(Mark *m)
{
    currentProbe = m;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void BulletProbe::menuEvent(coMenuItem *item)
{
    if (coVRCollaboration::instance()->getCouplingMode()
        == coVRCollaboration::MasterSlaveCoupling
        && !coVRCollaboration::instance()->isMaster())
    {
        return;
    }
    
    if (item == linearItem) // button new probe
    {
        if (linearItem->getState())
        {
            TokenBuffer tb;
            
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
    else if (item == cmiHideAll)
    {
        for (dims.reset(); dims.current(); dims.next())
        {
            dims.current()->setVisible(!cmiHideAll->getState());
        }
    }
    else if (item == bmiLoadFromFile)
    {
        ifstream infile; 
        osg::Matrix m0, m1;

        int count = 0;
        
        float a00,a01,a02,a03;
        float a10,a11,a12,a13;
        float a20,a21,a22,a23;
        float a30,a31,a32,a33;

        float b00,b01,b02,b03;
        float b10,b11,b12,b13;
        float b20,b21,b22,b23;
        float b30,b31,b32,b33;
            
        char data[100];

        std::string filename =  "marks.dat";  // TODO: read form config
        
        infile.open(filename); 
 
        if (infile.is_open())
        {
            cout << "Reading from file " << filename << endl;
            
            while (!infile.eof())
            {
                infile >> data;

                if (data[0] == '{')
                {
                    if (count % 2 == 0)
                    {
                        count += 1;
                        
                        infile >> a00 >> a01 >> a02 >> a03;
                        infile >> a10 >> a11 >> a12 >> a13;
                        infile >> a20 >> a21 >> a22 >> a23;
                        infile >> a30 >> a31 >> a32 >> a33;

                        m0 = osg::Matrix(a00,a01,a02,a03,
                                         a10,a11,a12,a13,
                                         a20,a21,a22,a23,
                                         a30,a31,a32,a33);
                    }
                    else
                    {
                        count += 1;

                        infile >> b00 >> b01 >> b02 >> b03;
                        infile >> b10 >> b11 >> b12 >> b13;
                        infile >> b20 >> b21 >> b22 >> b23;
                        infile >> b30 >> b31 >> b32 >> b33;

                        m1 = osg::Matrix(b00,b01,b02,b03,
                                         b10,b11,b12,b13,
                                         b20,b21,b22,b23,
                                         b30,b31,b32,b33);
                        
                        TokenBuffer tb0,tb1,tb2;
                        
                        tb0 << NEW_DIMENSION;
                        tb0 << count / 2 - 1;
                        
                        cover->sendMessage(this,
                               coVRPluginSupport::TO_SAME,
                               PluginMessageTypes::Measure0,
                               tb0.getData().length(),
                               tb0.getData().data());

                        tb1 << MOVE_MARK;
                        tb1 << count / 2 - 1;
                        tb1 << 0;

                        tb1 << a00 << a01 << a02 << a03;
                        tb1 << a10 << a11 << a12 << a13;
                        tb1 << a20 << a21 << a22 << a23;
                        tb1 << a30 << a31 << a32 << a33;

                        cover->sendMessage(BulletProbe::plugin,
                                           coVRPluginSupport::TO_SAME,
                                           PluginMessageTypes::Measure0,
                                           tb1.getData().length(),
                                           tb1.getData().data());

                        tb2 << MOVE_MARK;
                        tb2 << count / 2 - 1;
                        tb2 << 1;

                        tb2 << b00 << b01 << b02 << b03;
                        tb2 << b10 << b11 << b12 << b13;
                        tb2 << b20 << b21 << b22 << b23;
                        tb2 << b30 << b31 << b32 << b33;
                        
                        cover->sendMessage(BulletProbe::plugin,
                                           coVRPluginSupport::TO_SAME,
                                           PluginMessageTypes::Measure1,
                                           tb2.getData().length(),
                                           tb2.getData().data());

                        // do what placing is doing during manual creation
                        // for the loaded dim & marks
                        for (dims.reset(); dims.current(); dims.next())
                        {
                            dims.current()->marks[0]->placing = false;
                            dims.current()->marks[0]->create();
                            dims.current()->marks[1]->placing = false;
                            dims.current()->marks[0]->create();
                            dims.current()->create();
                        }
                    }
                }
            }
        }
        else
        {
            cout << "BulletProbe: Reading from file "
                 << filename << "failed" << endl;
        }
    }
    else if (item == bmiSaveToFile)
    {
        ofstream outfile;
        outfile.open("marks.dat", ios::out | ios::trunc );
        
        osg::Matrix m0, m1;
        for (dims.reset(); dims.current(); dims.next())
        {
            dims.current()->marks[0]->getMat(m0);
            dims.current()->marks[1]->getMat(m1);
            
            outfile << m0 << endl;
            outfile << m1 << endl;
        }
    }
    else if (item == clearItem) // button clear all
    {
        for (dims.reset(); dims.current(); dims.remove());
    }
    else if (item == markerScalePoti)
    {
        coneSize = markerScalePoti->getValue();
    }
    else if (item == lineWidthPoti)
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
    else if (item == pmiExtraLength)
    {
        for (dims.reset(); dims.current(); dims.next())
        {
            dims.current()->setExtraLength(pmiExtraLength->getValue());
        }
    }

    // for the units
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

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void BulletProbe::createMenuEntry()
{
    measureMenuItem = new coSubMenuItem("BulletProbe");
    measureMenu = new coRowMenu("BulletProbe");
    measureMenuItem->setMenu(measureMenu);

    linearItem = new coCheckboxMenuItem("New Probe", false);
    linearItem->setMenuListener(this);
    measureMenu->add(linearItem);

    cmiHideAll = new coCheckboxMenuItem("Hide All", false);
    cmiHideAll->setMenuListener(this);
    measureMenu->add(cmiHideAll);
    
    bmiLoadFromFile = new coButtonMenuItem("Load From File");
    bmiLoadFromFile->setMenuListener(this);
    measureMenu->add(bmiLoadFromFile);
                                                
    bmiSaveToFile = new coButtonMenuItem("Save To File");
    bmiSaveToFile->setMenuListener(this);
    measureMenu->add(bmiSaveToFile);

    bmiItem1 = new coButtonMenuItem("   ");
    bmiItem1->setMenuListener(this);
    bmiItem1->setActive(false);
    measureMenu->add(bmiItem1);
    
    clearItem = new coButtonMenuItem("Clear All");
    clearItem->setMenuListener(this);
    measureMenu->add(clearItem);

    bmiItem2 = new coButtonMenuItem("   ");
    bmiItem2->setMenuListener(this);
    bmiItem2->setActive(false);
    measureMenu->add(bmiItem2);
    
    //To set the units of measurement
    // unitsMenuItem = new coSubMenuItem("Units");
    // unitsMenu = new coRowMenu("Units");
    // unitsMenuItem->setMenu(unitsMenu);
    //measureMenu->add(unitsMenuItem);

    //checkboxArray.push_back(new coCheckboxMenuItem("Mikrometer", false));
    scaleArray.push_back(1.e-6f);
    unitArray.push_back("um");

    // checkboxArray.push_back(new coCheckboxMenuItem("Millimeter", false));
    scaleArray.push_back(1.e-3f);
    unitArray.push_back("mm");

    // checkboxArray.push_back(new coCheckboxMenuItem("Centimeter", false));
    scaleArray.push_back(1.e-2f);
    unitArray.push_back("cm");

    // checkboxArray.push_back(new coCheckboxMenuItem("Meter", true));
    scaleArray.push_back(1.f);
    unitArray.push_back("m");

    // checkboxArray.push_back(new coCheckboxMenuItem("Inch", false));
    scaleArray.push_back(0.0254f);
    unitArray.push_back("in");

    // checkboxArray.push_back(new coCheckboxMenuItem("Foot", false));
    scaleArray.push_back(0.0254f * 12.f);
    unitArray.push_back("ft");

    // checkboxArray.push_back(new coCheckboxMenuItem("Yard", false));
    scaleArray.push_back(0.0254f * 12.f * 3.f);
    unitArray.push_back("yd");
    // for (size_t i = 0; i < checkboxArray.size(); ++i)
    // {
    //     checkboxArray[i]->setMenuListener(this);
    //     unitsMenu->add(checkboxArray[i]);
    // }

    markerScalePoti = new coPotiMenuItem("Marker Size", 0, 600, 150);
    markerScalePoti->setMenuListener(this);
    measureMenu->add(markerScalePoti);

    // fontScalePoti = new coPotiMenuItem("Font Size", 0, 10, 3);
    // fontScalePoti->setMenuListener(this);
    // measureMenu->add(fontScalePoti);

    lineWidthPoti = new coPotiMenuItem("Line Width", 1, 30, 30);
    lineWidthPoti->setMenuListener(this);
    measureMenu->add(lineWidthPoti);

    pmiExtraLength = new coPotiMenuItem("Add Extra Length", 0.99, 5, 1);
    pmiExtraLength->setMenuListener(this);
    measureMenu->add(pmiExtraLength);

    cover->getMenu()->add(measureMenuItem);
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
void BulletProbe::removeMenuEntry()
{
    delete measureMenu;
    delete measureMenuItem;
    //delete fontScalePoti;
    delete lineWidthPoti;
    delete markerScalePoti;
    delete clearItem;

    delete cmiHideAll;
    delete bmiLoadFromFile;
    delete bmiSaveToFile;

    delete bmiItem1;
    delete bmiItem2;
    
    //delete snapItem;
    delete linearItem;
    // delete unitsMenu;
    // delete unitsMenuItem;

    for (size_t i = 0; i < checkboxArray.size(); ++i)
    {
        delete checkboxArray[i];
    }
}

// ----------------------------------------------------------------------------

COVERPLUGIN(BulletProbe)
