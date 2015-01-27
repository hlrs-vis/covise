/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "RoadSignal.h"
#include <osgDB/ReadFile>
#include <osg/CullFace>
#include <osg/AlphaFunc>
#include <cover/coVRFileManager.h>

TrafficLightSignalTurnCallback::TrafficLightSignalTurnCallback(osgSim::MultiSwitch *ms)
    : multiSwitch(ms)
{
}

void TrafficLightSignalTurnCallback::on()
{
    if (multiSwitch)
        multiSwitch->setActiveSwitchSet(0);
}
void TrafficLightSignalTurnCallback::off()
{
    if (multiSwitch)
        multiSwitch->setActiveSwitchSet(1);
}

unordered_map<std::string, SignalPrototype *> RoadSignal::signalsMap;

RoadSignal::RoadSignal(const std::string &setId, const std::string &setName, const double &setS, const double &setT, const bool &setDynamic,
                       const OrientationType &setOrient, const double &setZOffset, const std::string &setCountry,
                       const int &setType, const int &setSubtype, const std::string &subClass, const double &Size, const double &setValue)
    : Element(setId)
    , name(setName)
    , s(setS)
    , t(setT)
    , dynamic(setDynamic)
    , orientation(setOrient)
    , zOffset(setZOffset)
    , country(setCountry)
    , type(setType)
    , subtype(setSubtype)
    , subclass(subClass)
    , size(Size)
    , value(setValue)
{
    SignalPrototype *sp = NULL;
    unordered_map<std::string, SignalPrototype *>::iterator it = signalsMap.find(name);
    if (it == signalsMap.end())
    {
        sp = new SignalPrototype(name, country, type, subtype, subclass, zOffset == 0);
        signalsMap[name] = sp;
    }
    else
    {
        sp = it->second;
    }
    if (sp->signalNode.valid())
    {

        osg::Quat signalDir(-1.0 * M_PI, osg::Vec3d(0.0, 0.0, 1.0));
        if (orientation == NEGATIVE_TRACK_DIRECTION)
        {
            signalDir = osg::Quat(0.0 * M_PI, osg::Vec3d(0.0, 0.0, 1.0));
        }

        roadSignalNode = new osg::PositionAttitudeTransform();
        roadSignalNode->setPosition(osg::Vec3d(signalTransform.v().x(), signalTransform.v().y(), signalTransform.v().z()));
        roadSignalNode->setAttitude(osg::Quat(signalTransform.q().x(), signalTransform.q().y(), signalTransform.q().z(), signalTransform.q().w()) * signalDir);
        roadSignalNode->setName(setId);
        roadSignalNode->addChild(sp->signalNode.get());
    }
    else
    {
        roadSignalNode = NULL;
    }
}

SignalPrototype::SignalPrototype(std::string n, std::string c, int t, int st, std::string sc, bool isFlat)
{
    name = n;
    country = c;
    std::string fn = "share/covise/signals/" + country + "/" + n + ".osg";
    type = t;
    subtype = st;
    subclass = sc;
    flat = isFlat;
    const char *filename = opencover::coVRFileManager::instance()->getName(fn.c_str());
    if (filename)
    {
        signalNode = osgDB::readNodeFile(filename);
    }
    else
    {
        createGeometry();
        //fprintf(stderr,"SignalPrototype:: file %s not found.",fn);
    }
}
void SignalPrototype::createGeometry()
{
    bool withPost = false;
    std::string name_noPost = name;
    size_t len = name_noPost.length();
    if (len > 2)
    {
        if ((name_noPost[len - 2] == '_') && (name_noPost[len - 1] == 'p'))
        {
            withPost = true;
            name_noPost = name_noPost.substr(0, len - 2);
        }
    }
    osg::Geode *signGeode = new osg::Geode();
    signGeode->setName(name.c_str());

    osg::StateSet *signStateSet = signGeode->getOrCreateStateSet();

    signStateSet->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    signStateSet->setMode(GL_BLEND, osg::StateAttribute::ON);
    osg::CullFace *cullFace = new osg::CullFace();
    cullFace->setMode(osg::CullFace::BACK);
    signStateSet->setAttributeAndModes(cullFace, osg::StateAttribute::ON);
    osg::AlphaFunc *alphaFunc = new osg::AlphaFunc(osg::AlphaFunc::GREATER, 0.0);
    signStateSet->setAttributeAndModes(alphaFunc, osg::StateAttribute::ON);

    std::string fn = "share/covise/signals/" + country + "/" + name_noPost + ".png";

    float hsize = 0.6;
    float vsize = 0.6;

    const char *fileName = opencover::coVRFileManager::instance()->getName(fn.c_str());
    if (!fileName)
    {
        std::string fn = "share/covise/signals/" + country + "/" + name_noPost + ".tif";
        fileName = opencover::coVRFileManager::instance()->getName(fn.c_str());
    }

    if (fileName)
    {
        osg::Image *signTexImage = osgDB::readImageFile(fileName);
        osg::Texture2D *signTex = new osg::Texture2D;
        signTex->setWrap(osg::Texture2D::WRAP_S, osg::Texture::CLAMP_TO_EDGE);
        signTex->setWrap(osg::Texture2D::WRAP_T, osg::Texture::CLAMP_TO_EDGE);
        hsize = signTexImage->s() / 2000.0;
        vsize = signTexImage->t() / 1000.0;
        if (signTexImage)
            signTex->setImage(signTexImage);
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
        if (flat)
        {

            signTex->setFilter(osg::Texture::MIN_FILTER, osg::Texture::NEAREST);
            signTex->setFilter(osg::Texture::MAG_FILTER, osg::Texture::NEAREST);
        }
        else
        {
            signStateSet->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
        }
    }
    else
    {
        char tmpstr[200];

        if (subtype == -1)
            sprintf(tmpstr, "%d", type);
        else if (subclass == "")
            sprintf(tmpstr, "%d-%d", type, subtype);
        else
            sprintf(tmpstr, "%d-%d_%s", type, subtype, subclass.c_str());
        std::string fn = "share/covise/signals/" + country + "/" + tmpstr + ".png";

        float hsize = 0.6;
        float vsize = 0.6;

        const char *fileName = opencover::coVRFileManager::instance()->getName(fn.c_str());
        if (!fileName)
        {
            std::string fn = "share/covise/signals/" + country + "/" + tmpstr + ".tif";
            fileName = opencover::coVRFileManager::instance()->getName(fn.c_str());
        }
        if (fileName)
        {
            osg::Image *signTexImage = osgDB::readImageFile(fileName);
            osg::Texture2D *signTex = new osg::Texture2D;
            signTex->setWrap(osg::Texture2D::WRAP_S, osg::Texture::CLAMP_TO_EDGE);
            signTex->setWrap(osg::Texture2D::WRAP_T, osg::Texture::CLAMP_TO_EDGE);
            hsize = signTexImage->s() / 2000.0;
            vsize = signTexImage->t() / 1000.0;
            if (signTexImage)
                signTex->setImage(signTexImage);
            signStateSet->setTextureAttributeAndModes(3, signTex, osg::StateAttribute::ON);
        }
        else
        {
            std::cerr << "ERROR: no texture found named: " << fn;
        }
    }

    float pr = 0.03;
    float height = 2.5;
    osg::Vec3 v[12];
    v[0].set(-hsize / 2.0, 0, 0);
    v[1].set(hsize / 2.0, 0, 0);
    v[2].set(hsize / 2.0, 0, vsize);
    v[3].set(-hsize / 2.0, 0, vsize);
    v[4].set(-pr, 0.01, vsize * 0.8);
    v[5].set(-pr, 0.01 + 2 * pr, vsize * 0.8);
    v[6].set(pr, 0.01 + 2 * pr, vsize * 0.8);
    v[7].set(pr, 0.01, vsize * 0.8);
    v[8].set(-pr, 0.01, -height);
    v[9].set(-pr, 0.01 + 2 * pr, -height);
    v[10].set(pr, 0.01 + 2 * pr, -height);
    v[11].set(pr, 0.01, -height);

    osg::Vec3 np[4];
    np[0].set(-0.7, -0.7, 0);
    np[1].set(-0.7, 0.7, 0);
    np[2].set(0.7, 0.7, 0);
    np[3].set(0.7, -0.7, 0);

    osg::Vec3 n;
    n.set(0, -1, 0);

    osg::Geometry *signGeometry;
    signGeometry = new osg::Geometry();
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
    signTexCoords->push_back(osg::Vec2(0.5, 0));
    signNormals->push_back(n);

    signVertices->push_back(v[2]);
    signTexCoords->push_back(osg::Vec2(0.5, 1));
    signNormals->push_back(n);

    signVertices->push_back(v[3]);
    signTexCoords->push_back(osg::Vec2(0, 1));
    signNormals->push_back(n);

    n.set(0, 1, 0);

    signVertices->push_back(v[3]);
    signTexCoords->push_back(osg::Vec2(0.5, 1));
    signNormals->push_back(n);
    signVertices->push_back(v[2]);
    signTexCoords->push_back(osg::Vec2(1, 1));
    signNormals->push_back(n);
    signVertices->push_back(v[1]);
    signTexCoords->push_back(osg::Vec2(1, 0));
    signNormals->push_back(n);
    signVertices->push_back(v[0]);
    signTexCoords->push_back(osg::Vec2(0.5, 0));
    signNormals->push_back(n);

    if (withPost)
    {
        osg::Vec2 tc;
        tc.set(0.75, 0.5);
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

    signalNode = signGeode;
}

SignalPrototype::~SignalPrototype()
{
}

void RoadSignal::setTransform(const Transform &_transform)
{
    signalTransform = _transform;
    if (roadSignalNode)
    {
        osg::Quat signalDir(0.5 * M_PI, osg::Vec3d(0.0, 0.0, 1.0));
        if (orientation == NEGATIVE_TRACK_DIRECTION)
        {
            signalDir = osg::Quat(-0.5 * M_PI, osg::Vec3d(0.0, 0.0, 1.0));
        }
        if (zOffset == 0)
        {
            // street marking, no upright signal

            roadSignalNode->setPosition(osg::Vec3d(signalTransform.v().x(), signalTransform.v().y(), signalTransform.v().z() + 0.04));
            osg::Quat toFlat(-M_PI_2, osg::Vec3d(1.0, 0.0, 0.0));
            roadSignalNode->setAttitude(toFlat * osg::Quat(signalTransform.q().x(), signalTransform.q().y(), signalTransform.q().z(), signalTransform.q().w()) * signalDir);
        }
        else
        {
            roadSignalNode->setPosition(osg::Vec3d(signalTransform.v().x(), signalTransform.v().y(), signalTransform.v().z() + zOffset));
            roadSignalNode->setAttitude(osg::Quat(signalTransform.q().x(), signalTransform.q().y(), signalTransform.q().z(), signalTransform.q().w()) * signalDir);
        }
        if (country == "China")
        {
            if (size == 1)
            {
                roadSignalNode->setScale(osg::Vec3(5, 1, 5));
            }
            else if (size == 2)
            {
                roadSignalNode->setScale(osg::Vec3(6, 1, 6));
            }
            else if (size == 3)
            {
                roadSignalNode->setScale(osg::Vec3(10, 1, 10));
            }
            else if (size == 4)
            {
                roadSignalNode->setScale(osg::Vec3(12, 1, 12));
            }
            else
            {
                roadSignalNode->setScale(osg::Vec3(size, size, size));
            }
        }
        else
        {
            if (size != 2)
            {
                if (size == 1)
                {
                    roadSignalNode->setScale(osg::Vec3(0.75, 0.75, 0.75));
                }
                else if (size == 3)
                {
                    roadSignalNode->setScale(osg::Vec3(1.25, 1.25, 1.25));
                }
                else
                {
                    roadSignalNode->setScale(osg::Vec3(size, size, size));
                }
            }
        }
    }
}

osg::PositionAttitudeTransform *RoadSignal::getRoadSignalNode()
{

    return roadSignalNode;
}

osg::Node *TrafficLightSignal::trafficSignalNodeTemplate = NULL;

TrafficLightSignal::TrafficLightSignal(const std::string &setId, const std::string &setName, const double &setS, const double &setT, const bool &setDynamic,
                                       const OrientationType &setOrient, const double &setZOffset, const std::string &setCountry,
                                       const int &setType, const int &setSubtype, const std::string &setSubclass, const double &setSize, const double &setValue)
    : RoadSignal(setId, setName, setS, setT, setDynamic, setOrient, setZOffset, setCountry, setType, setSubtype, setSubclass, setSize, setValue)
    , signalGreenCallback(NULL)
    , signalYellowCallback(NULL)
    , signalRedCallback(NULL)
    , trafficSignalNode(NULL)
//,
//signalTurn((value>0.0) ? GO : STOP),
//signalTurnFinished(false),
//yellowPhaseTime(3.0),
//timer(yellowPhaseTime)
{
    const char *fileName = opencover::coVRFileManager::instance()->getName("materials/signal.osg");
    if (fileName && !trafficSignalNodeTemplate)
    {
        trafficSignalNodeTemplate = osgDB::readNodeFile(fileName);
    }
}

void TrafficLightSignal::setSignalGreenCallback(SignalTurnCallback *stc)
{
    signalGreenCallback = stc;
}

void TrafficLightSignal::setSignalYellowCallback(SignalTurnCallback *stc)
{
    signalYellowCallback = stc;
}

void TrafficLightSignal::setSignalRedCallback(SignalTurnCallback *stc)
{
    signalRedCallback = stc;
}

void TrafficLightSignal::switchGreenSignal(SignalSwitchType switchType)
{
    if (signalGreenCallback)
    {
        switch (switchType)
        {
        case ON:
            value = 1.0;
            signalGreenCallback->on();
            break;
        case OFF:
            signalGreenCallback->off();
            break;
        }
    }
}

void TrafficLightSignal::switchYellowSignal(SignalSwitchType switchType)
{
    if (signalYellowCallback)
    {
        switch (switchType)
        {
        case ON:
            value = 0.0;
            signalYellowCallback->on();
            break;
        case OFF:
            signalYellowCallback->off();
            break;
        }
    }
}

void TrafficLightSignal::switchRedSignal(SignalSwitchType switchType)
{
    if (signalRedCallback)
    {
        switch (switchType)
        {
        case ON:
            value = -1.0;
            signalRedCallback->on();
            break;
        case OFF:
            signalRedCallback->off();
            break;
        }
    }
}

osg::PositionAttitudeTransform *TrafficLightSignal::getRoadSignalNode()
{
    if (!trafficSignalNode)
    {
        if (!trafficSignalNodeTemplate)
        {
            return NULL;
        }
        else
        {
            osg::Quat signalDir(-0.5 * M_PI, osg::Vec3d(0.0, 0.0, 1.0));
            if (orientation == NEGATIVE_TRACK_DIRECTION)
            {
                signalDir = osg::Quat(0.5 * M_PI, osg::Vec3d(0.0, 0.0, 1.0));
            }

            trafficSignalNode = new osg::PositionAttitudeTransform();
            trafficSignalNode->addChild(dynamic_cast<osg::Node *>(trafficSignalNodeTemplate->clone(osg::CopyOp::DEEP_COPY_NODES)));
            trafficSignalNode->setPosition(osg::Vec3d(signalTransform.v().x(), signalTransform.v().y(), signalTransform.v().z()));
            trafficSignalNode->setAttitude(osg::Quat(signalTransform.q().x(), signalTransform.q().y(), signalTransform.q().z(), signalTransform.q().w()) * signalDir);
            trafficSignalNode->setName(name);

            return trafficSignalNode;
        }
    }

    return trafficSignalNode;
}

/*void TrafficLightSignal::signalGo()
{
   value = 0.0;
   signalTurn = GO;
   timer = 0.0;
   signalTurnFinished = false;
   if(signalYellowCallback) (*signalYellowCallback)();
}

void TrafficLightSignal::signalStop()
{
   value = 0.0;
   signalTurn = STOP;
   timer = 0.0;
   signalTurnFinished = false;
   if(signalYellowCallback) (*signalYellowCallback)();
}

void TrafficLightSignal::update(const double& dt)
{
   timer += dt;
   if(timer>=yellowPhaseTime && !signalTurnFinished) {
      switch(signalTurn) {
       case GO:
         value = 1.0;
         if(signalGreenCallback) (*signalGreenCallback)();
         signalTurnFinished = true;
         break;   
       case STOP:
         value = -1.0;
         if(signalRedCallback) (*signalRedCallback)();
         signalTurnFinished = true;
         break;
      }
   }
}*/
