/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//

#include "DLab.h"

#include <util/unixcompat.h>
#include <net/covise_host.h>
#include <net/covise_socket.h>
#include <OpenVRUI/osg/mathUtils.h>

DLabPlugin *DLabPlugin::plugin = NULL;

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeDLab(scene);
}

// Define the built in VrmlNodeType:: "DLab" fields

VrmlNodeType *VrmlNodeDLab::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("DLab", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class
    t->addExposedField("enabled", VrmlField::SFBOOL);
    t->addEventOut("ints_changed", VrmlField::MFINT32);
    t->addEventOut("floats_changed", VrmlField::MFFLOAT);

    return t;
}

VrmlNodeType *VrmlNodeDLab::nodeType() const
{
    return defineType(0);
}

VrmlNodeDLab::VrmlNodeDLab(VrmlScene *scene)
    : VrmlNodeChild(scene)
    , d_enabled(true)
{
    setModified();
}

VrmlNodeDLab::VrmlNodeDLab(const VrmlNodeDLab &n)
    : VrmlNodeChild(n.d_scene)
    , d_enabled(n.d_enabled)
{
    setModified();
}

VrmlNodeDLab::~VrmlNodeDLab()
{
}

VrmlNode *VrmlNodeDLab::cloneMe() const
{
    return new VrmlNodeDLab(*this);
}

VrmlNodeDLab *VrmlNodeDLab::toDLab() const
{
    return (VrmlNodeDLab *)this;
}

ostream &VrmlNodeDLab::printFields(ostream &os, int indent)
{
    if (!d_enabled.get())
        PRINT_FIELD(enabled);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeDLab::setField(const char *fieldName,
                               const VrmlField &fieldValue)
{
    if
        TRY_FIELD(enabled, SFBool)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeDLab::getField(const char *fieldName)
{
    if (strcmp(fieldName, "enabled") == 0)
        return &d_enabled;
    else if (strcmp(fieldName, "floats_changed") == 0)
        return &d_floats;
    else if (strcmp(fieldName, "ints_changed") == 0)
        return &d_ints;
    else
        cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
}

void VrmlNodeDLab::eventIn(double timeStamp,
                              const char *eventName,
                              const VrmlField *fieldValue)
{

    // Check exposedFields
    //else
    {
        VrmlNode::eventIn(timeStamp, eventName, fieldValue);
    }

    setModified();
}

void VrmlNodeDLab::render(Viewer *)
{
    if (!d_enabled.get())
        return;

    double timeStamp = System::the->time();
    if (DLabPlugin::plugin->numFloats)
    {
        d_floats.set(DLabPlugin::plugin->numFloats, DLabPlugin::plugin->floatValues);
        eventOut(timeStamp, "floats_changed", d_floats);
    }
    if (DLabPlugin::plugin->numInts)
    {
        d_ints.set(DLabPlugin::plugin->numInts, DLabPlugin::plugin->intValues);
        eventOut(timeStamp, "ints_changed", d_ints);
    }
}

DLabPlugin::DLabPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "DLabPlugin::DLabPlugin\n");

    plugin = this;
    conn = NULL;

    serverHost = NULL;
    localHost = new Host("localhost");

    port = coCoviseConfig::getInt("COVER.Plugin.DLab.TCPPort", 12345);
    std::string line = coCoviseConfig::getEntry("value","COVER.Plugin.DLab.Server","192.168.1.170");
    if (!line.empty())
    {
        if (strcasecmp(line.c_str(), "NONE") == 0)
            serverHost = NULL;
        else
            serverHost = new Host(line.c_str());
        cerr << serverHost->getName() << endl;
    }

    numFloats = 62;
    numInts = 61;

    floatValues = new float[numFloats];
    intValues = new int[numInts];
}

void DLabPlugin::createGeometry()
{

        osg::Vec3Array *coord = new osg::Vec3Array(4);
        (*coord)[0].set(-1, 0, -1);
        (*coord)[1].set( 1, 0, -1);
        (*coord)[2].set( 1, 0,  1);
        (*coord)[3].set(-1, 0,  1);

        osg::Vec2Array *texcoord = new osg::Vec2Array(4);

        (*texcoord)[0].set(0.0, 0.0);
        (*texcoord)[1].set(1.0, 0.0);
        (*texcoord)[2].set(1.0, 1.0);
        (*texcoord)[3].set(0.0, 1.0);

        geode = new osg::Geode();
        osg::Geometry *geometry = new osg::Geometry();

        ushort *vertices = new ushort[4];
        vertices[0] = 0;
        vertices[1] = 1;
        vertices[2] = 2;
        vertices[3] = 3;
        osg::DrawElementsUShort *plane = new osg::DrawElementsUShort(osg::PrimitiveSet::QUADS, 4, vertices);

        geometry->setVertexArray(coord);
        geometry->addPrimitiveSet(plane);
        geometry->setTexCoordArray(0, texcoord);

        osg::ref_ptr<osg::StateSet> state = geode->getOrCreateStateSet();
        state->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
        state->setMode(GL_BLEND, osg::StateAttribute::ON);
        state->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
	const char * fn = coVRFileManager::instance()->getName("share/covise/icons/fadenkreuz.png");
	osg::Image *blendTexImage=NULL;
	if(fn)
	{
	    blendTexImage = osgDB::readImageFile(fn);
	}
	osg::Texture2D *blendTex = new osg::Texture2D;
	blendTex->ref();
	blendTex->setWrap(osg::Texture2D::WRAP_S, osg::Texture::CLAMP);
	blendTex->setWrap(osg::Texture2D::WRAP_T, osg::Texture::CLAMP);
	if (blendTexImage)
	{
	    blendTex->setImage(blendTexImage);
	}
	state->setTextureAttributeAndModes(0, blendTex, osg::StateAttribute::ON);
	
        osg::TexEnv *texenv = new osg::TexEnv;
        texenv->setMode(osg::TexEnv::REPLACE);

        state->setTextureAttribute(0, texenv);
        
        geode->addDrawable(geometry);
        geode->setStateSet(state.get());
	targetTransform = new osg::MatrixTransform();
	targetTransform->addChild(geode);
	
}

// this is called if the plugin is removed at runtime
DLabPlugin::~DLabPlugin()
{
    fprintf(stderr, "DLabPlugin::~DLabPlugin\n");

    delete[] floatValues;
    delete[] intValues;
    delete conn;
    delete serverHost;
}

bool DLabPlugin::init()
{
    VrmlNamespace::addBuiltIn(VrmlNodeDLab::defineType());
    createGeometry();
    opencover::cover->getScene()->addChild(targetTransform);

    return true;
}

bool DLabPlugin::readVal(void *buf, unsigned int numBytes)
{
    unsigned int toRead = numBytes;
    unsigned int numRead = 0;
    int readBytes = 0;
    while (numRead < numBytes)
    {
        readBytes = conn->getSocket()->Read(((unsigned char *)buf) + readBytes, toRead);
        if (readBytes < 0)
        {
            cerr << "error reading data from socket" << endl;
            return false;
        }
        numRead += readBytes;
        toRead = numBytes - numRead;
    }
    return true;
}

void
DLabPlugin::preFrame()
{  
    if (conn)
    {
    const char * dataFromRemote=NULL;
        while ((dataFromRemote=conn->readLine()))
        {
	if(strcmp(dataFromRemote,"ConnectionClosed")==0)
	{
                delete conn;
                conn = NULL;
                //newNumFloats = 0;
                //newNumInts = 0;
                numFloats = 0;
                numInts = 0;
                cerr << "reset " << numInts << endl;
    }
    else
    {
    //fprintf(stderr,dataFromRemote);
        numFloats = sscanf(dataFromRemote,"%f %f",&floatValues[0],&floatValues[1]);
   // newNumFloats=2;
   // newNumInts=0;
    }
    /*
            int newNumFloats = 2; // should read these numbers from the server!!
            int newNumInts = 0; // should read these numbers from the server!!
            if (!readVal(&newNumFloats, sizeof(int)))
            {
                delete conn;
                conn = NULL;
                newNumFloats = 0;
                newNumInts = 0;
                numFloats = 0;
                numInts = 0;
                cerr << "reset " << newNumInts << endl;
            }
            if (!readVal(&newNumInts, sizeof(int)))
            {
                delete conn;
                conn = NULL;
                newNumFloats = 0;
                newNumInts = 0;
                numFloats = 0;
                numInts = 0;
                cerr << "reseti " << newNumInts << endl;
            }
            if (newNumFloats > 0 && newNumFloats != numFloats)
            {
                numFloats = (int)newNumFloats;
                delete[] floatValues;
                floatValues = new float[numFloats];
            }
            if (newNumInts > 0 && newNumInts != numInts)
            {
                numInts = (int)newNumInts;
                delete[] intValues;
                intValues = new int[numInts];
            }
            if (!readVal(floatValues, numFloats * sizeof(float)))
            {
                delete conn;
                conn = NULL;
                newNumFloats = 0;
                newNumInts = 0;
                numFloats = 0;
                numInts = 0;
                cerr << "reseti2 " << newNumInts << endl;
            }
            if (!readVal(intValues, numInts * sizeof(int)))
            {
                delete conn;
                conn = NULL;
                newNumFloats = 0;
                newNumInts = 0;
                numFloats = 0;
                numInts = 0;
                cerr << "reseti2 " << newNumInts << endl;
            }*/
        }
    }
    else if ((coVRMSController::instance()->isMaster()) && (serverHost != NULL))
    {
        // try to connect to server every 2 secnods
        if ((cover->frameTime() - oldTime) > 2)
        {
        cerr << "trying"  << endl;
            conn = new SimpleClientConnection(serverHost, port, 0);

            if (!conn->is_connected()) // could not open server port
            {
#ifndef _WIN32
                if (errno != ECONNREFUSED)
                {
                    fprintf(stderr, "Could not connect to DLab on %s; port %d\n", serverHost->getName(), port);
                    delete serverHost;
                    serverHost = NULL;
                }
#endif
                delete conn;
                conn = NULL;
                conn = new SimpleClientConnection(localHost, port, 0);
                if (!conn->is_connected()) // could not open server port
                {
#ifndef _WIN32
                    if (errno != ECONNREFUSED)
                    {
                        fprintf(stderr, "Could not connect to DLab on %s; port %d\n", localHost->getName(), port);
                    }
#endif
                    delete conn;
                    conn = NULL;
                }
                else
                {
                    fprintf(stderr, "Connected to DLab on %s; port %d\n", localHost->getName(), port);
                }
            }
            else
            {
                fprintf(stderr, "Connected to DLab on %s; port %d\n", serverHost->getName(), port);
		conn->getSocket()->setNonBlocking(true);
		struct linger linger;
    linger.l_onoff = 0;
    linger.l_linger = 0;
                 int bufSize=100;

		setsockopt(conn->get_id(NULL), SOL_SOCKET, SO_LINGER, (char *)&linger, sizeof(linger));
		//setsockopt(conn->get_id(NULL), SOL_SOCKET, SO_RCVBUF, (char *)&bufSize, sizeof(bufSize));
            }
            if (conn && conn->is_connected())
            {
                int id = 2;
                conn->getSocket()->write(&id, sizeof(id));
            }
            oldTime = cover->frameTime();
        }
    }
    if (coVRMSController::instance()->isMaster())
    {
        //cerr << numFloats << endl;
        //cerr << numInts << endl;
        coVRMSController::instance()->sendSlaves((char *)&numFloats, sizeof(int));
        coVRMSController::instance()->sendSlaves((char *)&numInts, sizeof(int));
        if (numFloats)
            coVRMSController::instance()->sendSlaves((char *)floatValues, numFloats * sizeof(float));
        if (numInts)
            coVRMSController::instance()->sendSlaves((char *)intValues, numInts * sizeof(int));
    }
    else
    {
        int newNumFloats = 0;
        int newNumInts = 0;
        coVRMSController::instance()->readMaster((char *)&newNumFloats, sizeof(int));
        coVRMSController::instance()->readMaster((char *)&newNumInts, sizeof(int));
        //cerr << newNumFloats << endl;
        //cerr << newNumInts << endl;
        if (newNumFloats > 0 && newNumFloats != numFloats)
        {
            cerr << "resize" << endl;
            numFloats = newNumFloats;
            delete[] floatValues;
            floatValues = new float[numFloats];
        }
        if (newNumInts > 0 && newNumInts != numInts)
        {
            cerr << "resize" << endl;
            numInts = newNumInts;
            delete[] intValues;
            intValues = new int[numInts];
        }
        if (newNumFloats > 0 && numFloats)
        {
            //cerr << "rf" << endl;
            coVRMSController::instance()->readMaster((char *)floatValues, numFloats * sizeof(float));
        }
        if (newNumFloats > 0 && numInts)
        {
            //cerr << "ri" << endl;
            coVRMSController::instance()->readMaster((char *)intValues, numInts * sizeof(int));
        }
    }
    
    // update Target
    
	osg::Matrix mat;
	mat.makeScale(100.0,100.0,100.0);
	mat.setTrans(0,1000,0);
	osg::Matrix rot;
	float maxx=1920.0;
	float maxy=1080.0;
	float x = -((floatValues[0]-(maxx/2.0))/maxx)*sin(50*M_PI/180.0);
	float y = -((floatValues[1]-(maxy/2.0))/maxy)*sin(28.1*M_PI/180.0);
    //fprintf(stderr,"\nfloat %f %f\n",x,y);
	coCoord ori;
	ori.hpr.set(asin(x)/M_PI*180.0,asin(y)/M_PI*180.0,0);
	ori.makeMat(rot);
	//osg::Matrix both = mat*opencover::cover->getViewerMat();
	//fprintf(stderr,"\n%f %f %f\n",both.getTrans().x(),both.getTrans().y(),both.getTrans().z());
        targetTransform->setMatrix(mat*rot*opencover::cover->getViewerMat());
	
}

COVERPLUGIN(DLabPlugin)
