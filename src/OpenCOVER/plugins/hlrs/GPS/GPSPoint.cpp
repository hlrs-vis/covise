/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: GPS OpenCOVER Plugin (is polite)                          **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner, T.Gudat	                                             **
 **                                                                          **
 ** History:  								     **
 ** June 2008  v1	    				       		     **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "GPSPoint.h"
#include "GPS.h"

#include <cover/coBillboard.h>
#include <cover/coVRLabel.h>
#include <osg/Material>
#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/ShapeDrawable>
#include <osg/Geode>
#include <osg/MatrixTransform>
#include <osg/Billboard>
#include <osg/Switch>
#include <osgText/Font>
#include <osgText/Text>

#include <osgDB/ReadFile>
#include <osg/CullFace>
#include <osg/AlphaFunc>
#include <cover/coVRFileManager.h>
#include <cover/coVRConfig.h>

#include <proj_api.h>
#include <chrono>

#include <xercesc/dom/DOM.hpp>
#if XERCES_VERSION_MAJOR < 3
#include <xercesc/dom/DOMWriter.hpp>
#else
#include <xercesc/dom/DOMLSSerializer.hpp>
#endif
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/util/XMLUni.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>

namespace opencover
{
class coVRLabel;
}
// GPSPoint
GPSPoint::GPSPoint()
{
    geoTrans = new osg::MatrixTransform();
    geoScale = new osg::MatrixTransform();
    Point = new osg::Group();

    osg::Matrix m;
    m.makeScale(osg::Vec3(1,1,1));
    geoScale->setMatrix(m);

    switchSphere = new osg::Switch();
    switchSphere->setName("Switch Sphere");
    switchSphere->setNewChildDefaultValue(false);
    switchDetail = new osg::Switch();
    switchDetail->setName("Switch Detail");
    switchDetail->setNewChildDefaultValue(true);

    geoTrans->addChild(geoScale);
    geoScale->addChild(Point);
    Point->addChild(switchSphere);
    Point->addChild(switchDetail);
}
GPSPoint::~GPSPoint()
{
    fprintf(stderr, "GPSPoint deleted\n");
}
void GPSPoint::setIndex(int i)
{
    Point->setName(std::to_string(i) + ".");
    geoTrans->setName("TM " + std::to_string(i) + ".");
    geoScale->setName("Scale " + std::to_string(i) + ".");
}

void GPSPoint::readFile (xercesc::DOMElement *node)
{
    double x;
    double y;
    double z;
    double t;
    float v;
    std::string typetext;
    XMLCh *t1 = NULL;

    char *lon = xercesc::XMLString::transcode(node->getAttribute(t1 = xercesc::XMLString::transcode("lon"))); xercesc::XMLString::release(&t1);
    char *lat = xercesc::XMLString::transcode(node->getAttribute(t1 = xercesc::XMLString::transcode("lat"))); xercesc::XMLString::release(&t1);

    sscanf(lon, "%lf", &x);
    sscanf(lat, "%lf", &y);

    //fprintf(stderr, "read from file:   lon: %s\n",lon);
    //fprintf(stderr, "read from file:   lat: %s\n",lat);


    xercesc::DOMNodeList *nodeContentList = node->getChildNodes();
    int nodeContentListLength = nodeContentList->getLength();
    for (int i = 0; i < nodeContentListLength; ++i)
    {
        xercesc::DOMElement *nodeContent = dynamic_cast<xercesc::DOMElement *>(nodeContentList->item(i));
        if (!nodeContent)
            continue;
        char *tmp = xercesc::XMLString::transcode(nodeContent->getNodeName());
        std::string nodeContentName = tmp;
        xercesc::XMLString::release(&tmp);
        if(nodeContentName == "ele")
        {
            char *alt = xercesc::XMLString::transcode(nodeContent->getTextContent());
            //fprintf(stderr, "read from file:   ele: %s\n",alt);
            sscanf(alt, "%lf", &z);
            xercesc::XMLString::release(&alt);
        }
        else if(nodeContentName == "time")
        {
            char *time = xercesc::XMLString::transcode(nodeContent->getTextContent());
            //fprintf(stderr, "read from file:   time: %s\n",time);
            sscanf(time, "%lf", &t);
            xercesc::XMLString::release(&time);
        }
        else if(nodeContentName == "name")
        {
            char *name = xercesc::XMLString::transcode(nodeContent->getTextContent());
            //fprintf(stderr, "read from file:   name: %s\n",name);
            typetext = name;
            if(typetext.empty())
            {
                typetext = "NO MESSAGE";
            }
            xercesc::XMLString::release(&name);
        }
        else if(nodeContentName == "link")
        {
            xercesc::DOMNodeList *extensionsList = nodeContent->getChildNodes();
            int extensionsListLength = extensionsList->getLength();
            for (int k = 0; k < extensionsListLength; ++k)
            {
                xercesc::DOMElement *extensionNode = dynamic_cast<xercesc::DOMElement *>(extensionsList->item(k));
                if (!extensionNode)
                    continue;
                char *tmp = xercesc::XMLString::transcode(extensionNode->getNodeName());
                std::string extensionNodeName = tmp;
                xercesc::XMLString::release(&tmp);
                if(extensionNodeName == "text")
                {
                    char *linktext = xercesc::XMLString::transcode(extensionNode->getTextContent());
                    fprintf(stderr, "name of related file: %s\n",linktext);
                    filename = linktext;
                    if(typetext.empty())
                    {
                        fprintf(stderr, "Error: no related file\n");
                    }
                    xercesc::XMLString::release(&linktext);
                }
                else {
                    fprintf(stderr, "unknown extension node named: %s\n",nodeContentName.c_str() );
                }
            }
        }
        else {
            fprintf(stderr, "unknown content node named: %s\n",nodeContentName.c_str() );
        }

    }
    this->setPointData(x, y, z, t, v, typetext);

    xercesc::XMLString::release(&lat);
    xercesc::XMLString::release(&lon);
}

void GPSPoint::setPointData (double x, double y, double z, double t, float v, std::string &name)
{
    altitude = GPSPlugin::instance()->getAlt(x,y)+GPSPlugin::instance()->zOffset;
    x *= DEG_TO_RAD;
    y *= DEG_TO_RAD;
    longitude = x;
    latitude = y;
    time = t;
    speed = v;

    int error = pj_transform(GPSPlugin::instance()->pj_from, GPSPlugin::instance()->pj_to, 1, 1, &longitude, &latitude, NULL);

    osg::Matrix m;
    m.makeTranslate(osg::Vec3(longitude,latitude,altitude-4));
    geoTrans->setMatrix(m);


    if(error !=0 )
    {
        fprintf(stderr, "------ \nError transforming coordinates, code %d \n", error);
        fprintf (stderr, "%s \n ------ \n", pj_strerrno (error));
    }

    if(name.empty())
    {
        fprintf(stderr, "Error: unidentified GPSPoint\n");
    }
    else if(name == "Good")
    {
        PT = Good;
    }
    else if(name == "Medium")
    {
        PT = Medium;
    }
    else if(name == "Bad")
    {
        PT = Bad;
    }
    else if(name == "Angst")
    {
        PT = Angst;
    }
    else if(name == "Foto")
    {
        PT = Foto;
    }
    else if(name == "Sprachaufnahme")
    {
        PT = Sprachaufnahme;
    }
    else if(name == "Barriere")
    {
        PT = Barriere;
    }
    else if(name == "Fussgaenger" || name == "Fahrrad" || name == "Öpnv" || name == "Miv")
    {
        PT = OtherChoice;
    }
    else {
        PT = Text;
        text = name;
    }
    Point->setName(Point->getName() + " " + name);
    geoTrans->setName(geoTrans->getName() + " " + name);
}
void GPSPoint::draw()
{
    osg::Vec4 *color = new osg::Vec4();
    std::string name;

    switch (PT){
    case Good:
        *color = osg::Vec4(0.0f, 1.0f, 0.0f, 1.0f);
        //createPictureBox(text = "good" , color);
        createSign(GPSPlugin::instance()->iconGood);
        break;
    case Medium:
        *color = osg::Vec4(1.0f, 1.0f, 0.0f, 1.0f);
        //createPictureBox(text = "medium" , color);
        createSign(GPSPlugin::instance()->iconMedium);
        break;
    case Bad:
        *color = osg::Vec4(1.0f, 0.0f, 0.0f, 1.0f);
        //createPictureBox(text = "bad" , color);
        createSign(GPSPlugin::instance()->iconBad);
        break;
    case Angst:
        *color = osg::Vec4(0.5f, 1.0f, 1.0f, 1.0f);
        //createPictureBox(text = "Angst" , color);
        createSign(GPSPlugin::instance()->iconAngst);
        break;
    case Text:
        *color = osg::Vec4(0.0f, 0.0f, 1.0f, 1.0f);
        //createTextBox();
        createSign(GPSPlugin::instance()->iconText);
        break;
    case Foto:
        *color = osg::Vec4(0.5f, 0.0f, 1.0f, 1.0f);
        //createPictureBox(text = "FOTO" , color);
        createSign(GPSPlugin::instance()->iconFoto);
        break;
    case Sprachaufnahme:
        *color = osg::Vec4(0.0f, 0.5f, 1.0f, 1.0f);
        //createPictureBox(text = "Audio" , color);
        createSign(GPSPlugin::instance()->iconSprachaufnahme);
        break;
    case Barriere:
        *color = osg::Vec4(0.5f, 0.5f, 1.0f, 1.0f);
        //createPictureBox(text = "Something different" , color);
        createSign(GPSPlugin::instance()->iconBarriere);
        break;
    case OtherChoice:
        *color = osg::Vec4(0.0f, 0.0f, 0.0f, 0.0f);
        //createPictureBox(text = "Something different" , color);
        //createSign(GPSPlugin::instance()->iconBarriere);
        break;
    }
    createSphere(color);
}

void GPSPoint::createSphere(osg::Vec4 *colVec)
{
    float Radius = 5.0f;

    sphere = new osg::Sphere(osg::Vec3(0, 0 , 5), Radius);

    osg::ref_ptr<osg::Material> material_sphere = new osg::Material();
    material_sphere->setDiffuse(osg::Material::FRONT_AND_BACK, *colVec);
    material_sphere->setAmbient(osg::Material::FRONT_AND_BACK, *colVec);

    sphereD = new osg::ShapeDrawable(sphere);
    osg::ref_ptr<osg::StateSet> stateSet = sphereD->getOrCreateStateSet();
    stateSet->setAttribute /*AndModes*/ (material_sphere.get(), osg::StateAttribute::PROTECTED);
    stateSet->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    sphereD->setStateSet(stateSet);

    PointSphere = new osg::Geode();
    PointSphere->addDrawable(sphereD);
    switchSphere->addChild(PointSphere);

}



void GPSPoint::createBillboard()
{
    BBoard = new coBillboard();
    BBoard->setNormal(osg::Vec3(0,-1,0));
    BBoard->setAxis(osg::Vec3(0,0,1));
    BBoard->setMode(coBillboard::AXIAL_ROT);
    //BBgeode = new osg::Geode();
    //BBoard->addChild(BBgeode);

    switchDetail->addChild(BBoard);
}
void GPSPoint::createSign(osg::Image *img)
{

    createBillboard();

    auto start = std::chrono::steady_clock::now();
    osg::Geode *signGeode = new osg::Geode();
    signGeode->setName(text.c_str());

    osg::StateSet *signStateSet = signGeode->getOrCreateStateSet();

    signStateSet->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    signStateSet->setMode(GL_BLEND, osg::StateAttribute::ON);
    osg::CullFace *cullFace = new osg::CullFace();
    cullFace->setMode(osg::CullFace::BACK);
    signStateSet->setAttributeAndModes(cullFace, osg::StateAttribute::ON);
    osg::AlphaFunc *alphaFunc = new osg::AlphaFunc(osg::AlphaFunc::GREATER, 0.0);
    signStateSet->setAttributeAndModes(alphaFunc, osg::StateAttribute::ON);

    float hsize = 1;
    float vsize = 1;

    bool withPost = true;

    osg::Texture2D *signTex = new osg::Texture2D;
    signTex->setWrap(osg::Texture2D::WRAP_S, osg::Texture::CLAMP_TO_EDGE);
    signTex->setWrap(osg::Texture2D::WRAP_T, osg::Texture::CLAMP_TO_EDGE);

    signTex->setImage(img);

    signStateSet->setTextureAttributeAndModes(3, signTex, osg::StateAttribute::ON);

    // TODO implement shader for signs and street marks
    if (streetmarkMaterial.get() == NULL)
    {
        streetmarkMaterial = new osg::Material;
        streetmarkMaterial->ref();
        streetmarkMaterial->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
        streetmarkMaterial->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.2f, 0.2f, 0.2f, 1.0));
        streetmarkMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
        streetmarkMaterial->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
        streetmarkMaterial->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 1.0f, 1.0));
        streetmarkMaterial->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);
    }

    signStateSet->setAttributeAndModes(streetmarkMaterial.get(), osg::StateAttribute::ON);

    signTex->setFilter(osg::Texture::MIN_FILTER, osg::Texture::NEAREST);
    signTex->setFilter(osg::Texture::MAG_FILTER, osg::Texture::NEAREST);

    float pr = 0.03;
    float height = 2.5;
    osg::Vec3 v[12];
    v[0].set(-hsize / 2.0, 0, 0 +height);
    v[1].set(hsize / 2.0, 0, 0 +height);
    v[2].set(hsize / 2.0, 0, vsize +height);
    v[3].set(-hsize / 2.0, 0, vsize +height);

    v[4].set(-pr, 0.01,+height);
    v[5].set(-pr, 0.01 + 2 * pr,  +height);
    v[6].set(pr, 0.01 + 2 * pr,  +height);
    v[7].set(pr, 0.01, +height);
    v[8].set(-pr, 0.01, 0);
    v[9].set(-pr, 0.01 + 2 * pr, 0);
    v[10].set(pr, 0.01 + 2 * pr, 0);
    v[11].set(pr, 0.01, 0);

    osg::Vec3 np[4];
    np[0].set(-0.7, -0.7, 0);
    np[1].set(-0.7, 0.7, 0);
    np[2].set(0.7, 0.7, 0);
    np[3].set(0.7, -0.7, 0);

    osg::Vec3 n;
    n.set(0, -1, 0);

    osg::Geometry *signGeometry;
    signGeometry = new osg::Geometry();
    signGeometry->setUseDisplayList(opencover::coVRConfig::instance()->useDisplayLists());
    signGeometry->setUseVertexBufferObjects(opencover::coVRConfig::instance()->useVBOs());
    signGeode->addDrawable(signGeometry);

    osg::Vec3Array *signVertices;
    signVertices = new osg::Vec3Array;
    signGeometry->setVertexArray(signVertices);

    osg::Vec3Array *signNormals;
    signNormals = new osg::Vec3Array;
    signGeometry->setNormalArray(signNormals);
    signGeometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

    osg::Vec2Array *signTexCoords;
    signTexCoords = new osg::Vec2Array;
    signGeometry->setTexCoordArray(3, signTexCoords);

    signVertices->push_back(v[0]);
    signTexCoords->push_back(osg::Vec2(0, 0));
    signNormals->push_back(n);

    signVertices->push_back(v[1]);
    signTexCoords->push_back(osg::Vec2(1, 0));
    signNormals->push_back(n);

    signVertices->push_back(v[2]);
    signTexCoords->push_back(osg::Vec2(1, 1));
    signNormals->push_back(n);

    signVertices->push_back(v[3]);
    signTexCoords->push_back(osg::Vec2(0, 1));
    signNormals->push_back(n);


    if (withPost)
    {
        osg::Vec2 tc;
        tc.set(0.5, 0.02);
        signVertices->push_back(v[4]);
        signTexCoords->push_back(tc);
        signNormals->push_back(np[0]);
        signVertices->push_back(v[8]);
        signTexCoords->push_back(tc);
        signNormals->push_back(np[0]);
        signVertices->push_back(v[11]);
        signTexCoords->push_back(tc);
        signNormals->push_back(np[3]);
        signVertices->push_back(v[7]);
        signTexCoords->push_back(tc);
        signNormals->push_back(np[3]);

        signVertices->push_back(v[7]);
        signTexCoords->push_back(tc);
        signNormals->push_back(np[3]);
        signVertices->push_back(v[11]);
        signTexCoords->push_back(tc);
        signNormals->push_back(np[3]);
        signVertices->push_back(v[10]);
        signTexCoords->push_back(tc);
        signNormals->push_back(np[2]);
        signVertices->push_back(v[6]);
        signTexCoords->push_back(tc);
        signNormals->push_back(np[2]);

        signVertices->push_back(v[6]);
        signTexCoords->push_back(tc);
        signNormals->push_back(np[2]);
        signVertices->push_back(v[10]);
        signTexCoords->push_back(tc);
        signNormals->push_back(np[2]);
        signVertices->push_back(v[9]);
        signTexCoords->push_back(tc);
        signNormals->push_back(np[1]);
        signVertices->push_back(v[5]);
        signTexCoords->push_back(tc);
        signNormals->push_back(np[1]);

        signVertices->push_back(v[5]);
        signTexCoords->push_back(tc);
        signNormals->push_back(np[1]);
        signVertices->push_back(v[9]);
        signTexCoords->push_back(tc);
        signNormals->push_back(np[1]);
        signVertices->push_back(v[8]);
        signTexCoords->push_back(tc);
        signNormals->push_back(np[0]);
        signVertices->push_back(v[4]);
        signTexCoords->push_back(tc);
        signNormals->push_back(np[0]);
    }

    osg::DrawArrays *sign = new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, signVertices->size());
    signGeometry->addPrimitiveSet(sign);
    BBoard->addChild(signGeode);

    auto end = std::chrono::steady_clock::now();
    fprintf(stderr, "Signcreation %d microseconds\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

}
void GPSPoint::createTextBox()
{
    createBillboard();

    osgText::Text *textBox = new osgText::Text();
    textBox->setAlignment(osgText::Text::CENTER_BASE_LINE);
    textBox->setAxisAlignment(osgText::Text::XZ_PLANE );
    textBox->setColor(osg::Vec4(1, 0, 1, 1));
    //osg::ref_ptr<osgText::Font> font = coVRFileManager::instance()->loadFont(NULL);
    //text->setFont(font);
    textBox->setCharacterSize(5);
    textBox->setText(text.c_str());
    textBox->setPosition(osg::Vec3(0, 5, 0));

    //BBgeode->addDrawable(textBox);
}


void GPSPoint::createPictureBox(std::string &text, osg::Vec4 *colVec)
{
    createBillboard();

    //createSign();
}
