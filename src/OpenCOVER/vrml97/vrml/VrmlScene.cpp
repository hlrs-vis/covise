/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
// VrmlScene.cpp
//

#include <vector>
#include <algorithm>
#include <cctype>

#include <errno.h>
#ifndef WIN32
#include <libgen.h>
#endif

#include "config.h"

#include "VrmlScene.h"

#include "Doc.h"
#include "Viewer.h"
#include "System.h"

#include "VrmlNamespace.h"
#include "VrmlNodeType.h"
#include "coEventQueue.h"

// Handle clicks on Anchor nodes
#include "VrmlNodeAnchor.h"

// Bindable children nodes
#include "VrmlNodeBackground.h"
#include "VrmlNodeFog.h"
#include "VrmlNodeNavigationInfo.h"
#include "VrmlNodeViewpoint.h"
#include "VrmlNodeCOVER.h"

// List of scene-scoped lights in the scene
#include "VrmlNodePointLight.h"
#include "VrmlNodeSpotLight.h"

// List of Movies in the scene
#include "VrmlNodeMovieTexture.h"

// List of AudioClips in the scene
#include "VrmlNodeAudioClip.h"

// List of Scripts in the scene
#include "VrmlNodeScript.h"

// List of TimeSensors in the scene
#include "VrmlNodeTimeSensor.h"

#include "MathUtils.h"
#include "util/coFileUtil.h"

#ifdef HAVE_CRYPTOPP
#include <cryptopp/default.h>
#endif
#include "util/Keyfile.h"

// Max time in seconds between updates. Make this user
// setable to balance performance with cpu usage.
#ifndef DEFAULT_DELTA
#define DEFAULT_DELTA 0.5
#endif

namespace vrml
{
bool headlightEnabled;
int enableLights = 0;
}
//
// Create a VrmlScene from a URL (optionally loading from a local copy,
// so I can run as a netscape helper but still retrieve embedded urls).
//

using namespace vrml;
using covise::coDirectory;

VrmlScene::VrmlScene(const char *sceneUrl, const char *localCopy)
    : d_url(0)
    , d_urlLocal(0)
    , d_namespace(0)
    , d_modified(false)
    , d_newView(false)
    , d_deltaTime(DEFAULT_DELTA)
    , d_pendingUrl(0)
    , d_pendingParameters(0)
    , d_pendingNodes(0)
    , d_pendingScope(0)
    , d_frameRate(0.0)
    , d_firstEvent(0)
    , d_lastEvent(0)
    , oldNi(0)
    , resetVPFlag(true)
    , d_WasEncrypted(false)
	, d_loadSuccess(false)
{
    d_nodes.addToScene(this, sceneUrl);
    d_backgrounds = new VrmlNodeList;
    d_backgroundStack = new VrmlNodeList;
    d_fogs = new VrmlNodeList;
    d_fogStack = new VrmlNodeList;
    d_navigationInfos = new VrmlNodeList;
    d_navigationInfoStack = new VrmlNodeList;
    d_viewpoints = new VrmlNodeList;
    d_viewpointStack = new VrmlNodeList;
    d_scopedLights = new VrmlNodeList;
    d_scripts = new VrmlNodeList;
    d_timers = new VrmlNodeList;
    d_movies = new VrmlNodeList;
    d_audioClips = new VrmlNodeList;
    d_arSensors = new VrmlNodeList;

    d_sensorEventQueue = new coEventQueue(this);
    d_incomingSensorEventQueue = new coEventQueue(this);
    System::the->createMenu();
    headlightEnabled = System::the->getHeadlight();
    cache = NULL;

    if (sceneUrl)
    {
        if (localCopy)
        {
            cache = new InlineCache(localCopy);
        }
		if (!load(sceneUrl, localCopy))
		{
			System::the->error("VRMLScene: Couldn't load '%s'.\n", sceneUrl);
		}
		else
		{
			d_loadSuccess = true;
		}
    }
}

void VrmlScene::setMenuVisible(bool vis)
{
    System::the->setMenuVisibility(vis);
}

VrmlScene::~VrmlScene()
{
	if (d_namespace != NULL)
	{
		VrmlNamespace::resetNamespaces(d_namespace->getNumber().first);
	}
	System::the->destroyMenu();
    d_nodes.addToScene(0, 0);
    d_nodes.removeChildren();

    bindableRemoveAll(d_backgroundStack);
    delete d_backgroundStack;
    bindableRemoveAll(d_fogStack);
    delete d_fogStack;
    bindableRemoveAll(d_navigationInfoStack);
    delete d_navigationInfoStack;
    bindableRemoveAll(d_viewpointStack);
    delete d_viewpointStack;

    delete d_backgrounds;
    delete d_fogs;
    delete d_navigationInfos;
    delete d_viewpoints;

    //delete viewpointMenu;
    //delete cbg;

    delete d_scopedLights;
    delete d_scripts;
    delete d_timers;
    delete d_movies;
    delete d_audioClips;

    delete d_url;
    delete d_urlLocal;

    delete d_pendingUrl;
    delete d_pendingParameters;
    delete d_pendingNodes;
    delete d_pendingScope;

    delete d_namespace;

    delete d_sensorEventQueue;
    delete d_incomingSensorEventQueue;
}

// Load a (possibly non-VRML) file...

bool VrmlScene::loadUrl(VrmlMFString *url, VrmlMFString *parameters, bool replace)
{
    if (!url)
        return false;

    int np = parameters ? parameters->size() : 0;
    char **params = parameters ? parameters->get() : 0;

    // try each url until we find one we can handle
    int i, n = url->size();
    char **urls = url->get();
    for (i = 0; i < n; ++i)
    {
        if (!urls[i])
            continue;

        // #Viewpoint
        if (*urls[i] == '#')
        {
            if (load(urls[i], NULL, replace))
                break;
        }

        // Load .wrl's, or pass off to system
        else
        { // Check mime type...
            char *tail = strrchr(urls[i], SLASH);
            if (!tail)
                tail = urls[i];
            char *mod = strchr(tail, '#');
            if (!mod)
                mod = urls[i] + strlen(urls[i]);
            if (isWrl(urls[i]))
            {
                if (load(urls[i], NULL, replace))
                    break;
            }
            else
            {
                if (System::the->loadUrl(urls[i], np, params))
                    break;
            }
        }
    }

    return i != n; // true if we found a url that loaded
}

// Called by viewer when a destroy request is received. The request
// is just passed on to the client via the worldChanged CB.

void VrmlScene::destroyWorld()
{
    doCallbacks(DESTROY_WORLD);
}

// Replace nodes

void VrmlScene::replaceWorld(VrmlMFNode &nodes, VrmlNamespace *ns,
                             Doc *url, Doc *urlLocal)
{
    System::the->debug("replaceWorld( url %s )\n", url->url().c_str());

    delete d_namespace;
    delete d_url;
    delete d_urlLocal;

    d_namespace = ns;
    d_url = url;
    d_urlLocal = urlLocal;

    // Clear bindable stacks.
    // they should all remove themselves from the stack
    bindableRemoveAll(d_backgroundStack);
    bindableRemoveAll(d_fogStack);
    bindableRemoveAll(d_navigationInfoStack);
    bindableRemoveAll(d_viewpointStack);

    // stop them

    /* end = d_audioClips->end();
   for (i = d_audioClips->begin(); i != end; ++i)
   {
      VrmlNodeAudioClip *c = (*i)->as<VrmlNodeAudioClip>();
      if (c) c->getAudio()->;
   }*/

    // Get rid of current world: pending events, nodes.
    flushEvents();
    d_nodes.removeChildren();

    d_nodes.flushRemoveList();

    // Do this to set the relative URL
    d_nodes.addToScene((VrmlScene *)this, urlDoc()->url().c_str());

    // Add the nodes to a Group and put the group in the scene.
    // This will load EXTERNPROTOs and Inlines.
    d_nodes.addChildren(nodes);

    // Send initial set_binds to bindable nodes
    double timeNow = System::the->time();
    VrmlSFBool flag(true);
    VrmlNode *bindable = 0; // compiler warning

    if (d_backgrounds->size() > 0 && (bindable = d_backgrounds->front()) != 0)
        bindable->eventIn(timeNow, "set_bind", &flag);

    if (d_fogs->size() > 0 && (bindable = d_fogs->front()) != 0)
        bindable->eventIn(timeNow, "set_bind", &flag);

    if (d_navigationInfos->size() > 0 && (bindable = d_navigationInfos->front()) != 0)
        bindable->eventIn(timeNow, "set_bind", &flag);

    if (d_viewpoints->size() > 0 && (bindable = d_viewpoints->front()) != 0)
        bindable->eventIn(timeNow, "set_bind", &flag);

    // Notify anyone interested that the world has changed
    doCallbacks(REPLACE_WORLD);

    setModified();
}

void VrmlScene::addWorld(VrmlMFNode &nodes, VrmlNamespace *ns,
                         Doc *url, Doc *urlLocal)
{
    System::the->debug("addWorld( url %s )\n", url->url().c_str());

    delete d_url;
    delete d_urlLocal;

    d_namespace = ns;
    d_url = url;
    d_urlLocal = urlLocal;

    // Do this to set the relative URL
    d_nodes.addToScene((VrmlScene *)this, urlDoc()->url().c_str());

    // Add the nodes to a Group and put the group in the scene.
    // This will load EXTERNPROTOs and Inlines.
    d_nodes.addChildren(nodes);

    // Send initial set_binds to bindable nodes
    double timeNow = System::the->time();
    VrmlSFBool flag(true);
    VrmlNode *bindable = 0; // compiler warning

    if (d_backgrounds->size() > 0 && (bindable = d_backgrounds->front()) != 0)
        bindable->eventIn(timeNow, "set_bind", &flag);

    if (d_fogs->size() > 0 && (bindable = d_fogs->front()) != 0)
        bindable->eventIn(timeNow, "set_bind", &flag);

    if (d_navigationInfos->size() > 0 && (bindable = d_navigationInfos->front()) != 0)
        bindable->eventIn(timeNow, "set_bind", &flag);

    if (d_viewpoints->size() > 0 && (bindable = d_viewpoints->front()) != 0)
        bindable->eventIn(timeNow, "set_bind", &flag);

    setModified();
}

void VrmlScene::doCallbacks(int reason)
{
    SceneCBList::iterator cb, cbend = d_sceneCallbacks.end();
    for (cb = d_sceneCallbacks.begin(); cb != cbend; ++cb)
        (*cb)(reason);
}

void VrmlScene::addWorldChangedCallback(SceneCB cb)
{
    d_sceneCallbacks.push_front(cb);
}

bool VrmlScene::wasEncrypted() const
{
    return d_WasEncrypted;
}

void VrmlScene::clearRelativeURL()
{
    delete d_url;
    d_url = NULL;
}
// Read a VRML97 file.
// This is only for [*.wrl][#viewpoint] url loading (no parameters).

bool VrmlScene::load(const char *url, const char *localCopy, bool replace)
{
    // Look for '#Viewpoint' syntax. There ought to be a current
    // scene if this format is used.
    VrmlSFBool flag(true);
    if (*url == '#')
    {
        VrmlNode *vp = d_namespace ? d_namespace->findNode(url + 1) : 0;

        // spec: ignore if named viewpoint not found
        if (vp)
        {
            vp->eventIn(System::the->time(), "set_bind", &flag);
            setModified();
        }

        return true;
    }

    //std::cerr << "VrmlScene::load(url=" << (url?url:"<null>") << ", localCopy=" << (localCopy?localCopy:"<null>") << ")" << std::endl;

    // Try to load a file. Prefer a local copy if available.
    Doc *tryUrl;
    if (localCopy)
        tryUrl = new Doc(localCopy, 0);
    else
    {
        tryUrl = new Doc(url, d_url);
    }
	;
    VrmlNamespace *newScope = new VrmlNamespace(System::the->getFileId(url));
    bool wasEncrypted = false;
    VrmlMFNode *newNodes = readWrl(tryUrl, newScope, &wasEncrypted);
    d_WasEncrypted = d_WasEncrypted | wasEncrypted;

    if (newNodes)
    {
        Doc *sourceUrl = tryUrl, *urlLocal = 0;
        if (localCopy)
        {
            sourceUrl = new Doc(url);
            urlLocal = tryUrl;
        }
        if (replace)
            replaceWorld(*newNodes, newScope, sourceUrl, urlLocal);
        else
            addWorld(*newNodes, newScope, sourceUrl, urlLocal);

        // repair IMPORTed routes (X3D)
        d_namespace->repairRoutes();

        delete newNodes;

        // Look for '#Viewpoint' syntax
        auto sourceUrlMod = sourceUrl->urlModifier();
        if (!sourceUrlMod.empty())
        {
            VrmlNode *vp = d_namespace->findNode(sourceUrlMod.c_str() + 1);
            double timeNow = System::the->time();
            if (vp)
                vp->eventIn(timeNow, "set_bind", &flag);
        }

        System::the->setCurrentFile(url);

        return true; // Success.
    }

    delete tryUrl;
    return false;
}

// Read a VRML file from one of the urls.

bool VrmlScene::isWrl(const std::string &filename)
{
    for (std::string ending: {".wrl", ".wrz", ".vrml", ".vrml.gz", ".wrl.gz", ".x3dv", ".x3dv.gz", ".x3dvz"})
    {
        auto len = ending.length();
        if (filename.length() < len)
            continue;
        std::string tail = filename.substr(filename.length()-ending.length());
        std::transform(tail.begin(), tail.end(), tail.begin(), ::tolower);
        if (tail == ending)
            return true;
    }
    return false;
}

VrmlMFNode *VrmlScene::readWrl(VrmlMFString *urls, Doc *relative,
                               VrmlNamespace *ns, bool *encrypted)
{
    Doc url;
    int i, n = urls->size();
    for (i = 0; i < n; ++i)
    {
        //System::the->debug("Trying to read url '%s'\n", urls->get(i));
        url.seturl(urls->get(i), relative);
        VrmlMFNode *kids = VrmlScene::readWrl(&url, ns, encrypted);
        if (kids)
            return kids;
        else if (i < n - 1 && strncmp(urls->get(i), "urn:", 4))
            System::the->warn("Couldn't read url '%s': %s\n",
                              urls->get(i), strerror(errno));
    }

    return nullptr;
}

// yacc globals

#define yyin lexerin
#define yyrestart lexerrestart
#define yyparse parserparse
extern void yystring(char *);
extern void yyfunction(int (*)(char *, int));
extern int yyparse();
extern void yyrestart(FILE *input_file);

extern FILE *yyin;
extern VrmlNamespace *yyNodeTypes;
extern VrmlMFNode *yyParsedNodes;
extern Doc *yyDocument;

#if HAVE_LIBPNG

#include <zlib.h>
#define YYIN yygz
extern gzFile yygz;

#else

#define YYIN yyin
#endif

namespace
{
std::vector<unsigned char> encryptedBuffer;
std::vector<char> vrmlBuffer;
std::vector<char>::iterator vrmlBufferActualPosition = vrmlBuffer.end();

void ResetEncryptedVrmlBuffer()
{
    std::vector<unsigned char> empty;
    encryptedBuffer.swap(empty);
}

void ResetVrmlBuffer()
{
    std::vector<char> empty;
    vrmlBuffer.swap(empty);
    vrmlBufferActualPosition = vrmlBuffer.end();
}

void InitVrmlBuffer(const vector<char> &buffer)
{
    vrmlBuffer = buffer;
    vrmlBufferActualPosition = vrmlBuffer.begin();
}

void InitVrmlBuffer(const char *buffer, int bufferSize)
{
    vrmlBuffer.assign(buffer, buffer + bufferSize);
    vrmlBufferActualPosition = vrmlBuffer.begin();
}

void DecryptVrmlBuffer()
{
    string decryptedString;
    vector<unsigned char> key = Keyfile::getKey();
#ifdef HAVE_CRYPTOPP
    CryptoPP::StringSource(&encryptedBuffer[0], encryptedBuffer.size(), true,
                           new CryptoPP::DefaultDecryptorWithMAC(&key[0], key.size(), new CryptoPP::StringSink(decryptedString)));
#endif
    InitVrmlBuffer(decryptedString.c_str(), (int)decryptedString.length());
}

void FillEncryptedVrmlBuffer(gzFile file)
{
    ResetEncryptedVrmlBuffer();
    vector<unsigned char> tmpBuf(4096);
    size_t readBytes = 0;
    while ((readBytes = gzread(file, &tmpBuf[0], tmpBuf.size())) > 0)
    {
        encryptedBuffer.insert(encryptedBuffer.end(), tmpBuf.begin(), tmpBuf.begin() + readBytes);
    }
}

void FillEncryptedVrmlBuffer(FILE *file)
{
    ResetEncryptedVrmlBuffer();
    vector<unsigned char> tmpBuf(4096);
    size_t readBytes = 0;
    while ((readBytes = fread(&tmpBuf[0], 1, tmpBuf.size(), file)) > 0)
    {
        encryptedBuffer.insert(encryptedBuffer.end(), tmpBuf.begin(), tmpBuf.begin() + readBytes);
    }
}

int ReadFromVrmlBuffer(char *buffer, int bufSize)
{

    if (vrmlBufferActualPosition == vrmlBuffer.end())
        return 0;

    int result = 0;
    for (result = 0; result < bufSize && (vrmlBufferActualPosition != vrmlBuffer.end());)
    {
        buffer[result++] = *vrmlBufferActualPosition;
        vrmlBufferActualPosition++;
    }
    return result;
}
}

// Read a VRML file and return the (valid) nodes.

VrmlMFNode *VrmlScene::readWrl(Doc *tryUrl, VrmlNamespace *ns, bool *encrypted)
{
    VrmlMFNode *result = 0;

    if (encrypted)
    {
        *encrypted = false;
    }

    System::the->debug("readWRL %s\n", tryUrl->url().c_str());

// Should verify MIME type...
#if HAVE_LIBPNG
    if ((YYIN = tryUrl->gzopen("rb")) != 0)
#else
    if ((YYIN = tryUrl->fopen("rb")) != 0)
#endif
    {
        unsigned int magic = 0;
        unsigned char *magicP = (unsigned char *)&magic;
        int readBytes = 0;
#if HAVE_LIBPNG
        readBytes = gzread(YYIN, &magic, sizeof(magic));
#else
        readBytes = fread(&magic, 1, sizeof(magic), YYIN);
#endif
        if (readBytes<0)
        {
            std::cerr << "VrmlScene: read failed" << std::endl;
        }

        if (magicP[0] == 0xde && magicP[1] == 0xad && magicP[2] == 0xc0 && magicP[3] == 0xde)
        {
#ifdef HAVE_CRYPTOPP
            FillEncryptedVrmlBuffer(YYIN);
            DecryptVrmlBuffer();
            result = readFunction(ReadFromVrmlBuffer, tryUrl, ns);
            ResetVrmlBuffer();
            ResetEncryptedVrmlBuffer();
#else
            std::cerr << "cannot read encrypted VRML - compiled without Crypto++" << std::endl;
#endif

#if HAVE_LIBPNG
            tryUrl->gzclose();
#else
            tryUrl->fclose();
#endif
            if (encrypted)
            {
                *encrypted = true;
            }
        }
        else
        {
#if HAVE_LIBPNG
            gzrewind(YYIN);
#else
            fseek(YYIN, 0, SEEK_SET);
#endif
            if (magicP[0] == 0xef && magicP[1] == 0xbb && magicP[2] == 0xbf)
            {
                //we have an utf8 with BOM, skip three characters
#if HAVE_LIBPNG
                readBytes = gzread(YYIN, &magic, 3);
#else
                readBytes = fread(&magic, 1, 3, YYIN);
#endif
            }
            // If the caller is not interested in PROTO defs, use a local namespace
            VrmlNamespace nodeDefs;
            if (ns)
                yyNodeTypes = ns;
			
            else
                yyNodeTypes = &nodeDefs;

            yyDocument = tryUrl;
            yyParsedNodes = 0;
            //double StartTime=System::the->realTime(); // if a release and debug are mixed strdup followed by free causes a segfault on windows
            //char *file = strdup(tryUrl->url());
            //fprintf(stderr,"parsing %s: ", basename(file));
            //free(file);
            //char *tmp = (char *)malloc(12);
            //free(tmp);
            fflush(stderr);
#if HAVE_LIBPNG
            yyrestart(NULL);
#else
            yyrestart(YYIN);
#endif
            yyparse();
            //fprintf(stderr,"%g s\n",System::the->realTime() -StartTime);

            yyNodeTypes = 0;
            yyDocument = 0;

            result = yyParsedNodes;
            yyParsedNodes = 0;

#if HAVE_LIBPNG
            tryUrl->gzclose();
#else
            tryUrl->fclose();
#endif
        }
        YYIN = 0;
    }

    return result;
}

//

bool VrmlScene::loadFromString(const char *vrmlString)
{
    VrmlNamespace *newScope = new VrmlNamespace();
    VrmlMFNode *newNodes = readString(vrmlString, newScope);
    if (newNodes)
    {
        replaceWorld(*newNodes, newScope, 0, 0);
        delete newNodes;
        return true;
    }
    return false;
}

// Read VRML from a string and return the (valid) nodes.

VrmlMFNode *VrmlScene::readString(const char *vrmlString,
                                  VrmlNamespace *ns, Doc *relative)
{
    VrmlMFNode *result = 0;

    if (vrmlString != 0)
    {
        yyNodeTypes = ns;
        yyDocument = relative;
        yyParsedNodes = 0;

        // set input to be from string
        yyin = 0;
        yystring((char *)vrmlString);

        yyparse();

        yyNodeTypes = 0;
        result = yyParsedNodes;
        yyParsedNodes = 0;
    }

    return result;
}

// Load VRML from an application-provided callback function

bool VrmlScene::loadFromFunction(LoadCB cb, const char *url)
{
    Doc *doc = url ? new Doc(url, 0) : 0;
    VrmlNamespace *ns = new VrmlNamespace();
    VrmlMFNode *newNodes = readFunction(cb, doc, ns);

    if (newNodes)
    {
        replaceWorld(*newNodes, ns, doc, 0);
        delete newNodes;
        return true;
    }
    if (doc)
        delete doc;
    return false;
}

// Read VRML from a cb and return the (valid) nodes.

VrmlMFNode *VrmlScene::readFunction(LoadCB cb, Doc *url, VrmlNamespace *ns)
{
    VrmlMFNode *result = 0;

    if (cb != 0)
    {
        yyNodeTypes = ns;
        yyParsedNodes = 0;
        yyDocument = url;

        // set input to be from cb
        yyfunction(cb);

        yyparse();

        yyDocument = 0;
        yyNodeTypes = 0;
        result = yyParsedNodes;
        yyParsedNodes = 0;
    }

    return result;
}

// Read a PROTO from a URL to get the implementation of an EXTERNPROTO.
// This should read only PROTOs and return when the first/specified PROTO
// is read...

VrmlNodeType *VrmlScene::readPROTO(VrmlMFString *urls, Doc *relative, int parentId)
{
    // This is a problem. The nodeType of the EXTERNPROTO has a namespace
    // that refers back to this namespace (protos), which will be invalid
    // after we exit this function. I guess it needs to be allocated and
    // ref counted too...
    VrmlNamespace *protos = new VrmlNamespace(parentId); // Leek, but better than deleting
    // because this would destroy the just read PROTO
    Doc urlDoc;
    VrmlNodeType *def = 0;
    int i, n = urls->size();

    System::the->debug("readPROTO\n");

    for (i = 0; i < n; ++i)
    {
        //System::the->debug("Trying to read url '%s'\n", urls->get(i));
        urlDoc.seturl(urls->get(i), relative);
        VrmlMFNode *kids = VrmlScene::readWrl(&urlDoc, protos);
        if (kids)
            delete kids;

        // Grab the specified PROTO, or the first one.
        auto whichProto = urlDoc.urlModifier();
        if (!whichProto.empty())
            def = (VrmlNodeType *)protos->findType(whichProto.c_str() + 1);
        else
            def = (VrmlNodeType *)protos->firstType();

        if (def)
        {
            def->setActualUrl(urlDoc.url().c_str());
            break;
        }
        else if (i < n - 1 && strncmp(urls->get(i), "urn:", 4))
            System::the->warn("Couldn't read EXTERNPROTO url '%s': %s\n",
                              urls->get(i), strerror(errno));
    }

    return def;
}

// Write the current scene to a file.
// Need to save the PROTOs/EXTERNPROTOs too...

bool VrmlScene::save(const char *url)
{
    bool success = false;
    Doc save(url);
    ostream &os = save.outputStream();

    if (os)
    {
        os << "#VRML V2.0 utf8\n";
        os << d_nodes;
        success = true;
    }

    return success;
}

//
// Script node API functions
//
const char *VrmlScene::getName() { return "LibVRML97"; }

const char *VrmlScene::getVersion()
{
    static char vs[32];
    sprintf(vs, "%d.%d.%d", LIBVRML_MAJOR_VERSION, LIBVRML_MINOR_VERSION, LIBVRML_MICRO_VERSION);
    return vs;
}

double VrmlScene::getFrameRate() { return d_frameRate; }

// Queue an event to load URL/nodes (async so it can be called from a node)

void VrmlScene::queueLoadUrl(VrmlMFString *url, VrmlMFString *parameters)
{
    if (!d_pendingNodes && !d_pendingUrl)
    {
        d_pendingUrl = url->clone()->toMFString();
        d_pendingParameters = parameters->clone()->toMFString();
    }
}

void VrmlScene::queueReplaceNodes(VrmlMFNode *nodes, VrmlNamespace *ns)
{
    if (!d_pendingNodes && !d_pendingUrl)
    {
        d_pendingNodes = nodes->clone()->toMFNode();
        d_pendingScope = ns;
    }
}

// Event processing. Current events are in the array
// d_eventMem[d_firstEvent,d_lastEvent). If d_firstEvent == d_lastEvent,
// the queue is empty. There is a fixed maximum number of events. If we
// are so far behind that the queue is filled, the oldest events get
// overwritten.

void VrmlScene::queueEvent(double timeStamp,
                           VrmlField *value,
                           VrmlNode *toNode,
                           const char *toEventIn)
{
    Event *e = &d_eventMem[d_lastEvent];
    e->timeStamp = timeStamp;
    e->value = value;
    e->toNode = toNode;
    e->toEventIn = toEventIn;
    d_lastEvent = (d_lastEvent + 1) % MAXEVENTS;

    // If the event queue is full, discard the oldest (in terms of when it
    // was put on the queue, not necessarily in terms of earliest timestamp).
    if (d_lastEvent == d_firstEvent)
    {
        cerr << "UUps, Event queue is full!!!!" << endl;
        e = &d_eventMem[d_lastEvent];
        delete e->value;
        e->value = (VrmlField *)0xdeadbeefdeadbeef;
        d_firstEvent = (d_firstEvent + 1) % MAXEVENTS;
    }
}

// Any events waiting to be distributed?

bool VrmlScene::eventsPending()
{
    return d_firstEvent != d_lastEvent;
}

// Discard all pending events

void VrmlScene::flushEvents()
{
    while (d_firstEvent != d_lastEvent)
    {
        Event *e = &d_eventMem[d_firstEvent];
        d_firstEvent = (d_firstEvent + 1) % MAXEVENTS;
        delete e->value;
        e->value = (VrmlField *)0xffffbeef;
    }
}

// Called by the viewer when the cursor passes over, clicks, drags, or
// releases a sensitive object (an Anchor or another grouping node with
// an enabled TouchSensor child).

void VrmlScene::sensitiveEvent(void *object,
                               double timeStamp,
                               bool isOver, bool isActive,
                               double *point, double *M)
{
    VrmlNode *n = (VrmlNode *)object;

    if (n)
    {
        //cerr << "event for " << n->name() << endl;
        VrmlNodeAnchor *a = n->as<VrmlNodeAnchor>();
        if (a)
        {
            // This should really be (isOver && !isActive && n->wasActive)
            // (ie, button up over the anchor after button down over the anchor)
            if (isActive && isOver)
            {
                d_sensorEventQueue->addEvent(n, timeStamp, isOver, isActive, point);
                a->activate();
                //System::the->inform("");
            }
            else if (isOver)
            {
                const char *description = a->description();
                const char *url = a->url();
                if (description && url)
                    System::the->inform("%s (%s)", description, url);
                else if (description || url)
                    System::the->inform("%s", description ? description : url);
                //else
                //System::the->inform("");
            }
            //else
            //System::the->inform("");
        }

        // The parent grouping node is registered for Touch/Drag Sensors
        else
        {
            //cerr << "local Event for Node " << n->name() <<"|" <<  isOver <<"|" << isActive <<"|" << point[0] <<"|" << timeStamp <<endl;
            d_sensorEventQueue->addEvent(n, timeStamp, isOver, isActive, point);
            VrmlNodeGroup *g = n->as<VrmlNodeGroup>();
            if (g)
            {
                //System::the->inform("");
                g->activate(timeStamp, isOver, isActive, point, M);
                setModified();
            }
        }
    }

    //else
    //System::the->inform("");
}

void VrmlScene::remoteSensitiveEvent(void *object,
                                     double timeStamp,
                                     bool isOver, bool isActive,
                                     double *point, double *M)
{
    VrmlNode *n = (VrmlNode *)object;

    if (n)
    {
        //cerr << "event for " << n->name() << endl;
        VrmlNodeAnchor *a = n->as<VrmlNodeAnchor>();
        if (a)
        {
            // This should really be (isOver && !isActive && n->wasActive)
            // (ie, button up over the anchor after button down over the anchor)
            if (isActive && isOver)
            {
                a->activate();
                //System::the->inform("");
            }
            else if (isOver)
            {
                const char *description = a->description();
                const char *url = a->url();
                if (description && url)
                    System::the->inform("%s (%s)", description, url);
                else if (description || url)
                    System::the->inform("%s", description ? description : url);
                //else
                //System::the->inform("");
            }
            //else
            //System::the->inform("");
        }

        // The parent grouping node is registered for Touch/Drag Sensors
        else
        {
            VrmlNodeGroup *g = n->as<VrmlNodeGroup>();
            if (g)
            {
                //cerr << "remote Event for Node " << n->name() <<"|" <<  isOver <<"|" << isActive <<"|" << point[0] <<"|" << timeStamp <<endl;
                //System::the->inform("");
                g->activate(timeStamp, isOver, isActive, point, M);
                setModified();
            }
        }
    }

    //else
    //System::the->inform("");
}

//
// The update method is where the events are processed. It should be
// called after each frame is rendered.
//
bool VrmlScene::update(double timeStamp)
{
    if (timeStamp <= 0.0)
        timeStamp = System::the->time();
    VrmlSFTime now(timeStamp);

    if (theCOVER)
        theCOVER->update(timeStamp);
    d_deltaTime = DEFAULT_DELTA;

    // Update each of the timers.
    VrmlNodeList::iterator i, end = d_timers->end();
    for (i = d_timers->begin(); i != end; ++i)
    {
        VrmlNodeTimeSensor *t = (*i)->as<VrmlNodeTimeSensor>();
        if (t)
            t->update(now);
    }

    // Update each of the clips.
    // try, now done in render method (uwe w.)
    // now done again here
    end = d_audioClips->end();
    for (i = d_audioClips->begin(); i != end; ++i)
    {
        VrmlNodeAudioClip *c = (*i)->as<VrmlNodeAudioClip>();
        if (c)
            c->update(now);
    }

    // Update each of the scripts.
    end = d_scripts->end();
    for (i = d_scripts->begin(); i != end; ++i)
    {
        VrmlNodeScript *s = (*i)->as<VrmlNodeScript>();
        if (s)
            s->update(now);
    }

    // Update each of the movies.
    end = d_movies->end();
    for (i = d_movies->begin(); i != end; ++i)
    {
        VrmlNodeMovieTexture *m = (*i)->as<VrmlNodeMovieTexture>();
        if (m)
            m->update(now);
    }

    bool eventsProcessed = false;

    // Pass along events to their destinations
    while (d_firstEvent != d_lastEvent && !d_pendingUrl && !d_pendingNodes)
    {
        eventsProcessed = true;

        Event *e = &d_eventMem[d_firstEvent];

        // Ensure that the node is in the scene graph
        VrmlNode *n = e->toNode;
        if (this != n->scene())
        {
            System::the->debug("VrmlScene::update: %s::%s is not in the scene graph yet.\n",
                               n->nodeType()->getName(), n->name());
            n->addToScene((VrmlScene *)this, urlDoc()->url().c_str());
        }
        n->eventIn(e->timeStamp, e->toEventIn, e->value);
        //fprintf(stderr, "VrmlScene::eventIn: %s::%s\n", n->nodeType()->getName(), n->name());
        // this needs to change if event values are shared...

        if (e == &d_eventMem[d_firstEvent])
        { // if d_firstEvent has changed this event has already been deleted, then we should not do so again
            d_firstEvent = (d_firstEvent + 1) % MAXEVENTS;
            delete e->value;
            e->value = (VrmlField *)0xddddbeef;
        }
    }

    if (d_pendingNodes)
    {
        replaceWorld(*d_pendingNodes, d_pendingScope);
        delete d_pendingNodes;
        d_pendingNodes = 0;
        d_pendingScope = 0;
    }
    else if (d_pendingUrl)
    {

        d_sensorEventQueue->sendEvents(); // empty sensor Queue (the load Event might still be in the queue and we want
        // the remote sites to get this event before we start loading
        // empty events
        flushEvents();
        (void)loadUrl(d_pendingUrl, d_pendingParameters);
        delete d_pendingUrl;
        delete d_pendingParameters;
        d_pendingUrl = 0;
        d_pendingParameters = 0;
    }

    d_sensorEventQueue->update();

    // Signal a redisplay if necessary
    return eventsProcessed;
}

bool VrmlScene::headlightOn()
{
    VrmlNodeNavigationInfo *navInfo = bindableNavigationInfoTop();
    if (navInfo)
        return navInfo->headlightOn();
    return true;
}

// Draw this scene into the specified viewer

void VrmlScene::render(Viewer *viewer)
{
    //
    if (d_newView)
    {
        viewer->resetUserNavigation();
        d_newView = false;
    }

    // Default viewpoint parameters
    float position[3] = { 0.0, 0.0, 10.0 };
    float orientation[4] = { 0.0, 0.0, 1.0, 0.0 };
    float field = 0.785398f;
    float avatarSize = 1.6f;
    float avatarWidth = 0.25f;
    float stepSize = 0.75f;
    float visibilityLimit = 0.0;
    float scaleFactor = 1000;

    static const char *vpType = "free";
    VrmlNodeViewpoint *vp = bindableViewpointTop();
    if (vp)
    {

        field = vp->fieldOfView();
        vpType = vp->type();

        if (vp->lastBind)
        {
            vp->getLastPosition(position, orientation);
        }
        else
        {
            vp->getPosition(position, orientation);
        }
        //cerr << " position "<<  position[0] << ";"<<  position[1] << ";"<<  position[2] << ";"<<  endl;
        //cerr << " orientation "<<  orientation[0] << ";"<<  orientation[1] << ";"<<  orientation[2] << ";"<<orientation[3] << ";"<<  endl;

        //vp->inverseTransform(viewer);
    }

    VrmlNodeNavigationInfo *ni = bindableNavigationInfoTop();
    if (ni)
    {
        avatarWidth = ni->avatarSize()[0];
        avatarSize = ni->avatarSize()[1] * 120; //64;
        if (avatarSize == 0)
            avatarSize = 160;
        if (avatarWidth == 0)
            avatarWidth = 0.25;
        visibilityLimit = ni->visibilityLimit();

        // disable and introduce a scale of 1000 into VrmlBaseMat
        // reverted this change becaus we need the same units for VRML and COVISE objects!!
        // default scale is now 1000
        if (ni->scale() > 0)
        {
            // wa have a real scale!!
            if (ni->lastBind)
            {
                scaleFactor = ni->lastScale();
            }
            else
            {
                scaleFactor = ni->scale();
            }
        }
        avatarWidth = ni->avatarSize()[0] * 1000; // VRML is m opencover in mm
        stepSize = ni->avatarSize()[2] * 1000; // VRML is m opencover in mm

        System::the->setNavigationStepSize(stepSize);
        if (enableLights)
        {
            if (headlightEnabled != ni->headlightOn())
            {
                if (ni->headlightOn())
                {
                    System::the->setHeadlight(true);
                    headlightEnabled = true;
                }
                else
                {
                    System::the->setHeadlight(false);
                    headlightEnabled = false;
                }
            }
        }
        if (oldNi != ni)
        {
            System::the->setNearFar(ni->getNear(), ni->getFar());
            char **navTypes = ni->navTypes();
            if ((navTypes) && (navTypes[0]))
            {

                System::the->setNavigationType(navTypes[0]);
                System::the->setNavigationDriveSpeed(ni->speed());
            }
        }
        oldNi = ni;
    }

    if (resetVPFlag) // resetViewpointHack
    {
        cerr << "reset Viewpoint" << endl;
        resetVPFlag = false;
        viewer->setViewpoint(position, orientation, -100000,
                             avatarSize, visibilityLimit, vpType, scaleFactor);
    }

    viewer->setViewpoint(position, orientation, field,
                         avatarSize, visibilityLimit, vpType, scaleFactor);

    // Set background.
    VrmlNodeBackground *bg = bindableBackgroundTop();
    if (bg)
    { // Should be transformed by the accumulated rotations above ...
        bg->renderBindable(viewer);
    }
    else
        viewer->insertBackground(); // Default background

    // Fog
    VrmlNodeFog *f = bindableFogTop();
    if (f)
    {
        viewer->setFog(f->color(), f->visibilityRange(), f->fogType());
    }

    // Activate the headlight.
    // ambient is supposed to be 0 according to the spec...
    /* if ( headlightOn() )
    {
      float rgb[3] = { 1.0, 1.0, 1.0 };
      float xyz[3] = { 0.0, 0.0, -1.0 };
      float ambient = 0.3;

      viewer->insertDirLight( ambient, 1.0, rgb, xyz );
    }*/

    // Top level object
    viewer->beginObject(0, 0, NULL);

    // Do the scene-level lights (Points and Spots)
    /*VrmlNodeList::iterator li, end = d_scopedLights->end();
   for (li = d_scopedLights->begin(); li != end; ++li)
     {
       VrmlNodeLight* x = (*li)->toLight();
       if (x) x->renderScoped( viewer );
     }*/

    // Render the top level group
    d_nodes.render(viewer);

    viewer->endObject();

    // This is actually one frame late...
    d_frameRate = viewer->getFrameRate();

    clearModified();

    if (cache)
        cache->save();

    // If any events were generated during render (ugly...) do an update
    if (eventsPending())
        setDelta(0.0);
}

//
//  Bindable children node stacks. For the CS purists out there, these
//  aren't really stacks as they allow arbitrary elements to be removed
//  (not just the top).
//

VrmlNode *VrmlScene::bindableTop(BindStack stack)
{
    return (stack == 0 || stack->empty()) ? 0 : stack->front();
}

void VrmlScene::bindablePush(BindStack stack, VrmlNode *node)
{
    node->reference();
    bindableRemove(stack, node); // Remove any existing reference
    stack->push_front(node->reference());
    node->dereference();
    setModified();
}

void VrmlScene::bindableRemove(BindStack stack, VrmlNode *node)
{
    if (stack)
    {
        VrmlNodeList::iterator i;

        for (i = stack->begin(); i != stack->end(); ++i)
            if (*i == node)
            {
                if (*i)
                    (*i)->dereference();
                stack->erase(i);
                setModified();
                break;
            }
    }
}

// Remove all entries from the stack

void VrmlScene::bindableRemoveAll(BindStack stack)
{
    /*   VrmlNodeList::iterator i; -=warnings  */
    while (stack->size())
    {
        VrmlNode *n = stack->front();
        stack->pop_front();
        n->dereference();
    }
    /* macht Probleme, wenn das dereference einen destructor aufruft, der den
  stack aendert
  for (i = stack->begin(); i != stack->end(); ++i )
   {
      if(*i)
         (*i)->dereference();
   }
   stack->erase(stack->begin(), stack->end());*/
}

// The nodes in the "set of all nodes of this type" lists are not
// ref'd/deref'd because they are only added to and removed from
// the lists in their constructors and destructors.

// Bindable children nodes (stacks)
// Define for each Type:
//    add/remove to complete list of nodes of this type
//    VrmlNodeType *bindableTypeTop();
//    void bindablePush(VrmlNodeType *);
//    void bindableRemove(VrmlNodeType *);

// Background

void VrmlScene::addBackground(VrmlNodeBackground *n)
{
    d_backgrounds->push_back(n);
}

void VrmlScene::removeBackground(VrmlNodeBackground *n)
{
    d_backgrounds->remove(n);
}

VrmlNodeBackground *VrmlScene::bindableBackgroundTop()
{
    VrmlNode *b = bindableTop(d_backgroundStack);
    return b ? b->as<VrmlNodeBackground>() : 0;
}

void VrmlScene::bindablePush(VrmlNodeBackground *n)
{
    bindablePush(d_backgroundStack, n);
}

void VrmlScene::bindableRemove(VrmlNodeBackground *n)
{
    bindableRemove(d_backgroundStack, n);
}

// Fog

void VrmlScene::addFog(VrmlNodeFog *n)
{
    d_fogs->push_back(n);
}

void VrmlScene::removeFog(VrmlNodeFog *n)
{
    d_fogs->remove(n);
}

VrmlNodeFog *VrmlScene::bindableFogTop()
{
    VrmlNode *f = bindableTop(d_fogStack);
    return f ? f->as<VrmlNodeFog>() : 0;
}

void VrmlScene::bindablePush(VrmlNodeFog *n)
{
    bindablePush(d_fogStack, n);
}

void VrmlScene::bindableRemove(VrmlNodeFog *n)
{
    bindableRemove(d_fogStack, n);
}

// NavigationInfo
void VrmlScene::addNavigationInfo(VrmlNodeNavigationInfo *n)
{
    d_navigationInfos->push_back(n);
}

void VrmlScene::removeNavigationInfo(VrmlNodeNavigationInfo *n)
{
    d_navigationInfos->remove(n);
}

VrmlNodeNavigationInfo *VrmlScene::bindableNavigationInfoTop()
{
    VrmlNode *n = bindableTop(d_navigationInfoStack);
    return n ? n->as<VrmlNodeNavigationInfo>() : 0;
}

void VrmlScene::bindablePush(VrmlNodeNavigationInfo *n)
{
    bindablePush(d_navigationInfoStack, n);
}

void VrmlScene::bindableRemove(VrmlNodeNavigationInfo *n)
{
    bindableRemove(d_navigationInfoStack, n);
}

// Viewpoint
void VrmlScene::addViewpoint(VrmlNodeViewpoint *n)
{
    d_viewpoints->push_back(n);
    // add viewpoint to menu
    System::the->addViewpoint(this, n);
}

void VrmlScene::removeViewpoint(VrmlNodeViewpoint *n)
{
    System::the->removeViewpoint(this, n);
    d_viewpoints->remove(n);
    bindableRemove(n);
}

VrmlNodeViewpoint *VrmlScene::bindableViewpointTop()
{
    VrmlNode *t = bindableTop(d_viewpointStack);
    return t ? t->as<VrmlNodeViewpoint>() : 0;
}

void VrmlScene::bindablePush(VrmlNodeViewpoint *n)
{
    float position[3] = { 0.0, 0.0, 10.0 };
    float orientation[4] = { 0.0, 0.0, 1.0, 0.0 };
    VrmlNodeViewpoint *current = bindableViewpointTop();
    if (current)
    {
        current->recalcLast();
        current->getLastPosition(position, orientation);

        double M[16];
        current->inverseTransform(M);
        double toWorld[16];
        Minvert(toWorld, M);

        double toVP[16];
        n->inverseTransform(toVP);
        double both[16];
        Mmult(both, toWorld, toVP);

        VM(position, both, position);

        Mrotation(M, orientation);
        MM(M, both);
        MgetRot(orientation, &orientation[3], M);
        // get rotation axis and angle from M

        /*    fprintf(stderr, "bindablePush: pos=(%f %f %f) ori=(%f %f %f %f)\n",
         position[0], position[1], position[2],
         orientation[0], orientation[1], orientation[2], orientation[3]);
*/
        n->setLastViewpointPosition(position, orientation);
    }
    bindablePush(d_viewpointStack, n);
    d_newView = true;

    System::the->setViewpoint(this, n);
}

void VrmlScene::bindableRemove(VrmlNodeViewpoint *n)
{
    bindableRemove(d_viewpointStack, n);
    d_newView = true;
}

// Bind to the next viewpoint in the list

void VrmlScene::nextViewpoint()
{
    VrmlNodeViewpoint *vp = bindableViewpointTop();
    VrmlNodeList::iterator i;

    for (i = d_viewpoints->begin(); i != d_viewpoints->end(); ++i)
        if ((*i) == vp)
        {
            if (++i == d_viewpoints->end())
                i = d_viewpoints->begin();

            VrmlSFBool flag(true);
            if ((*i) && (vp = (*i)->as<VrmlNodeViewpoint>()) != 0)
                vp->eventIn(System::the->time(), "set_bind", &flag);

            return;
        }
}

void VrmlScene::prevViewpoint()
{
    VrmlNodeViewpoint *vp = bindableViewpointTop();
    VrmlNodeList::iterator i;

    for (i = d_viewpoints->begin(); i != d_viewpoints->end(); ++i)
        if ((*i) == vp)
        {
            if (i == d_viewpoints->begin())
                i = d_viewpoints->end();

            VrmlSFBool flag(true);
            if (*(--i) && (vp = (*i)->as<VrmlNodeViewpoint>()) != 0)
                vp->eventIn(System::the->time(), "set_bind", &flag);

            return;
        }
}

int VrmlScene::nViewpoints() { return (int)d_viewpoints->size(); }

void VrmlScene::getViewpoint(int nvp, const char **namep, const char **descriptionp)
{
    VrmlNodeList::iterator i;
    int n;

    *namep = *descriptionp = 0;
    for (i = d_viewpoints->begin(), n = 0; i != d_viewpoints->end(); ++i, ++n)
        if (n == nvp)
        {
            *namep = (*i)->name();
            *descriptionp = ((VrmlNodeViewpoint *)(*i))->description();
            return;
        }
}

void VrmlScene::setViewpoint(const char *name, const char *description)
{
    VrmlNodeList::iterator i;

    for (i = d_viewpoints->begin(); i != d_viewpoints->end(); ++i)
        if (strcmp(name, (*i)->name()) == 0 && strcmp(description, ((VrmlNodeViewpoint *)(*i))->description()) == 0)
        {
            VrmlNodeViewpoint *vp;
            VrmlSFBool flag(true);
            if ((vp = (VrmlNodeViewpoint *)*i) != 0)
                vp->eventIn(System::the->time(), "set_bind", &flag);
            return;
        }
}

void VrmlScene::setViewpoint(int nvp)
{
    VrmlNodeList::iterator i;
    int j = 0;

    for (i = d_viewpoints->begin(); i != d_viewpoints->end(); ++i)
    {
        if (j == nvp)
        {
            VrmlNodeViewpoint *vp;
            VrmlSFBool flag(true);
            if ((vp = (VrmlNodeViewpoint *)*i) != 0)
                vp->eventIn(System::the->time(), "set_bind", &flag);
            return;
        }
        ++j;
    }
}

// The nodes in these lists are not ref'd/deref'd because they
// are only added to and removed from the lists in their constructors
// and destructors.

// Scene-level distance-scoped lights

void VrmlScene::addScopedLight(VrmlNodeLight *light)
{
    d_scopedLights->push_back(light);
}

void VrmlScene::removeScopedLight(VrmlNodeLight *light)
{
    d_scopedLights->remove(light);
}

// Movies

void VrmlScene::addMovie(VrmlNodeMovieTexture *movie)
{
    d_movies->push_back(movie);
}

void VrmlScene::removeMovie(VrmlNodeMovieTexture *movie)
{
    d_movies->remove(movie);
}

// Scripts

void VrmlScene::addScript(VrmlNodeScript *script)
{
    d_scripts->push_back(script);
}

void VrmlScene::removeScript(VrmlNodeScript *script)
{
    d_scripts->remove(script);
}

// TimeSensors

void VrmlScene::addTimeSensor(VrmlNodeTimeSensor *timer)
{
    d_timers->push_back(timer);
}

void VrmlScene::removeTimeSensor(VrmlNodeTimeSensor *timer)
{
    d_timers->remove(timer);
}

// AudioClips

void VrmlScene::addAudioClip(VrmlNodeAudioClip *audio_clip)
{
    d_audioClips->push_back(audio_clip);
}

void VrmlScene::removeAudioClip(VrmlNodeAudioClip *audio_clip)
{
    d_audioClips->remove(audio_clip);
}

void VrmlScene::storeCachedInline(const char *url, const char *pathname, const Viewer::Object d_viewerObject)
{
    if (System::the->getCacheMode() != System::CACHE_CREATE
            && System::the->getCacheMode() != System::CACHE_REWRITE)
        return;

    std::string cachefile = System::the->getCacheName(url, pathname);
    if (cachefile.empty())
        return;
    std::cerr << "Cache store: " << pathname << " -> " << cachefile << std::endl;
    System::the->storeInline(cachefile.c_str(), d_viewerObject);
}

Viewer::Object VrmlScene::getCachedInline(const char *url, const char *pathname)
{
    if (System::the->getCacheMode() == System::CACHE_DISABLE
            || System::the->getCacheMode() == System::CACHE_REWRITE)
        return 0L;

    std::string cachefile = System::the->getCacheName(url, pathname);
    if (cachefile.empty())
    {
		if (pathname != NULL)
		{
			std::cerr << "Cache reject: no cachefile name: " << pathname << " -> " << cachefile << std::endl;
		}
        return 0L;
    }

#ifdef _WIN32
    struct _stat sbufInline, sbufCached;
    int ret = _stat(cachefile.c_str(), &sbufCached);
#else
    struct stat sbufInline, sbufCached;
    int ret = stat(cachefile.c_str(), &sbufCached);
#endif
    if (ret != 0)
    {
        std::cerr << "Cache reject: failed to stat cache: " << pathname << " -> " << cachefile << std::endl;
        return 0L;
    }

    if (System::the->getCacheMode() != System::CACHE_USEOLD)
    {
#ifdef _WIN32
        ret = _stat(pathname, &sbufInline);
#else
        ret = stat(pathname, &sbufInline);
#endif
        if (ret != 0)
        {
            std::cerr << "Cache reject: failed to stat Inline: " << pathname << " -> " << cachefile << std::endl;
            return 0L;
        }

#ifdef __APPLE__
#define st_mtim st_mtimespec
#endif

#ifdef WIN32
		if (sbufInline.st_mtime > sbufCached.st_mtime)
#else
        if (sbufInline.st_mtim.tv_sec > sbufCached.st_mtim.tv_sec
                || (sbufInline.st_mtim.tv_sec == sbufCached.st_mtim.tv_sec && sbufInline.st_mtim.tv_nsec > sbufCached.st_mtim.tv_nsec))
#endif
        {
            std::cerr << "Cache reject: too old: " << pathname << " -> " << cachefile << std::endl;
            return 0L;
        }
    }

    std::cerr << "Cache load: " << pathname << " -> " << cachefile << std::endl;
    return System::the->getInline(cachefile.c_str());
}

InlineCache::InlineCache(const char *vrmlfile)
{
    char *tmpName = coDirectory::canonical(vrmlfile);
    fileBase = coDirectory::fileOf(tmpName);
    directory = coDirectory::dirOf(tmpName);
    delete[] tmpName;

    std::stringstream str;
    str << directory << "/cache_" << fileBase << ".cache";
    cacheIndexName = str.str();
    //std::cerr << "InlineCache(file=" << vrmlfile << ") -> " << cacheIndexName << std::endl;
    FILE *fp = fopen(cacheIndexName.c_str(), "r");
    char buf[1000];
    char fn[1000];
    char url[1000];
    int time;
    if (fp)
    {
        while (!feof(fp))
        {
            char *retval_fgets;
            size_t retval_sscanf;
            retval_fgets = fgets(buf, 1000, fp);
            if (retval_fgets == NULL)
            {
                if (!feof(fp))
                {
                    std::cerr << "InlineCache::InlineCache: fgets failed" << std::endl;
                    fclose(fp);
                    return;
                }
            }
            else
            {
                retval_sscanf = sscanf(buf, "%s %s %d", url, fn, &time);
                if (retval_sscanf != 3)
                {
                    std::cerr << "InlineCache::InlineCache: sscanf failed" << std::endl;
                    fclose(fp);
                    return;
                }
                cacheList.push_back(cacheEntry(url, fn, time));
            }
        }
        fclose(fp);
    }
    modified = false;
}

InlineCache::~InlineCache()
{

    delete[] directory;
    delete[] fileBase;
}
void InlineCache::save()
{
    if (modified)
    {
        modified = false;
        FILE *fp = fopen(cacheIndexName.c_str(), "w");
        if (fp)
        {
            list<cacheEntry>::iterator i, end = cacheList.end();
            for (i = cacheList.begin(); i != end; ++i)
            {
                cacheEntry t = (*i);
                fprintf(fp, "%s %s %d\n", t.url, t.fileName, t.time);
            }
            fclose(fp);
        }
    }
}

cacheEntry *InlineCache::findEntry(const char *url)
{
    list<cacheEntry>::iterator i, end = cacheList.end();
    for (i = cacheList.begin(); i != end; ++i)
    {
        cacheEntry t = (*i);
        if (strcmp(i->url, url) == 0)
            return (&(*i));
    }
    return NULL;
}

cacheEntry::cacheEntry(const char *u, const char *fn, int t, Viewer::Object o)
{
    time = t;
    obj = o;
    url = new char[strlen(u) + 1];
    strcpy(url, u);
    fileName = new char[strlen(fn) + 1];
    strcpy(fileName, fn);
}

cacheEntry::cacheEntry(const cacheEntry &ce)
{
    time = ce.time;
    obj = ce.obj;
    url = new char[strlen(ce.url) + 1];
    strcpy(url, ce.url);
    fileName = new char[strlen(ce.fileName) + 1];
    strcpy(fileName, ce.fileName);
}

cacheEntry::~cacheEntry()
{
    delete[] url;
    delete[] fileName;
}
