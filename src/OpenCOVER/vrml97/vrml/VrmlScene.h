/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlScene.h
//  Construct a VRML scene graph from a VRML file or string
//

#ifndef _VRMLSCENE_
#define _VRMLSCENE_

#include "config.h"

// The loaders fill in a Group node
#include "VrmlNodeGroup.h"
#include "Viewer.h"

#include <list>

namespace vrml
{

class VrmlNode;

// List of nodes
//class VrmlNodeList; Why doesn't this work?
// I would rather not include <list> in here.
typedef std::list<VrmlNode *> VrmlNodeList;

extern VRMLEXPORT int enableLights;

class buttonSpecCell;
class Doc;
class Viewer;
class coEventQueue;

// PROTO definitions
class VrmlNamespace;
class VrmlNodeType;

// Bindable children node types
class VrmlNodeBackground;
class VrmlNodeFog;
class VrmlNodeNavigationInfo;
class VrmlNodeViewpoint;

// Lists of other node types needed for rendering or updating the scene
class VrmlNodeAudioClip;
class VrmlNodeLight;
class VrmlNodeMovieTexture;
class VrmlNodeScript;
class VrmlNodeTimeSensor;
class VrmlScene;

class cacheEntry
{
public:
    cacheEntry(const char *url, const char *fileName, int t, Viewer::Object obj = 0L);
    cacheEntry(const cacheEntry &);
    ~cacheEntry();
    char *fileName;
    Viewer::Object obj;
    char *url;
    int time;
};

class InlineCache
{
public:
    InlineCache(const char *vrmlfile);
    ~InlineCache();
    char *directory;
    char *fileBase;
    std::list<cacheEntry> cacheList;
    cacheEntry *findEntry(const char *url);
    void save();
    bool modified;
};

class VrmlScene
{

public:
    // These are available without a scene object
    static VrmlMFNode *readWrl(VrmlMFString *url, Doc *relative, VrmlNamespace *ns, bool *encrypted = NULL);
    static VrmlMFNode *readWrl(Doc *url, VrmlNamespace *ns, bool *encrypted = NULL);
    static VrmlMFNode *readString(const char *vrmlString, VrmlNamespace *ns, Doc *relative = NULL);

    typedef int (*LoadCB)(char *buf, int bufSize);
    static VrmlMFNode *readFunction(LoadCB cb, Doc *url, VrmlNamespace *ns);

    static VrmlNodeType *readPROTO(VrmlMFString *url, Doc *relative = 0);

    //
    VrmlScene(const char *url = 0, const char *localCopy = 0);
    virtual ~VrmlScene(); // Allow overriding of Script API methods

    // Destroy world (just passes destroy request up to client)
    void destroyWorld();

    // Replace world with nodes, recording url as the source URL.
    void replaceWorld(VrmlMFNode &nodes, VrmlNamespace *ns,
                      Doc *url = 0, Doc *urlLocal = 0);

    void clearRelativeURL();

    // Add world with nodes, recording url as the source URL.
    void addWorld(VrmlMFNode &nodes, VrmlNamespace *ns,
                  Doc *url = 0, Doc *urlLocal = 0);

    // A way to let the app know when a world is loaded, changed, etc.
    typedef void (*SceneCB)(int reason);

    // Valid reasons for scene callback (need more...)
    enum
    {
        DESTROY_WORLD,
        REPLACE_WORLD
    };

    void addWorldChangedCallback(SceneCB);

    // Load a generic file (possibly non-VRML)
    bool loadUrl(VrmlMFString *url, VrmlMFString *parameters = 0, bool replace = true);

    // Load a VRML file
    bool load(const char *url, const char *localCopy = 0, bool replace = true);

    // Load a VRML string
    bool loadFromString(const char *string);

    // Load VRML from an application-provided callback function
    bool loadFromFunction(LoadCB, const char *url = 0);

    // Save the scene to a file
    bool save(const char *url);

    // URL the current scene was loaded from
    Doc *urlDoc()
    {
        return d_url;
    }

    // Types and node names defined in this scope
    VrmlNamespace *scope()
    {
        return d_namespace;
    }

    // Queue an event to load URL/nodes (async so it can be called from a node)
    void queueLoadUrl(VrmlMFString *url, VrmlMFString *parameters);
    void queueReplaceNodes(VrmlMFNode *nodes, VrmlNamespace *ns);

    void sensitiveEvent(void *object, double timeStamp,
                        bool isOver, bool isActive, double *point, double *M);
    void remoteSensitiveEvent(void *object, double timeStamp,
                              bool isOver, bool isActive, double *point, double *M);

    // Queue an event for a given node
    void queueEvent(double timeStamp,
                    VrmlField *value,
                    VrmlNode *toNode, const char *toEventIn);

    bool eventsPending();

    void flushEvents();

    // Script node API support functions. Can be overridden if desired.
    virtual const char *getName();
    virtual const char *getVersion();
    double getFrameRate();

    // Returns true if scene needs to be re-rendered
    bool update(double timeStamp = -1.0);

    void render(Viewer *);

    // Indicate that the scene state has changed, need to re-render
    void setModified()
    {
        d_modified = true;
    }
    void clearModified()
    {
        d_modified = false;
    }
    bool isModified()
    {
        return d_modified;
    }

    // Time until next update needed
    void setDelta(double d)
    {
        if (d < d_deltaTime)
            d_deltaTime = d;
    }
    double getDelta()
    {
        return d_deltaTime;
    }

    // Bindable children nodes can be referenced via a list or bindable stacks.
    // Define for each bindableType:
    //    addBindableType(bindableType *);
    //    removeBindableType(bindableType *);
    //    bindableType *bindableTypeTop();
    //    void bindablePush(VrmlNodeType *);
    //    void bindableRemove(VrmlNodeType *);

    // Background
    void addBackground(VrmlNodeBackground *);
    void removeBackground(VrmlNodeBackground *);
    VrmlNodeBackground *bindableBackgroundTop();
    void bindablePush(VrmlNodeBackground *);
    void bindableRemove(VrmlNodeBackground *);

    // Fog
    void addFog(VrmlNodeFog *);
    void removeFog(VrmlNodeFog *);
    VrmlNodeFog *bindableFogTop();
    void bindablePush(VrmlNodeFog *);
    void bindableRemove(VrmlNodeFog *);

    // NavigationInfo
    void addNavigationInfo(VrmlNodeNavigationInfo *);
    void removeNavigationInfo(VrmlNodeNavigationInfo *);
    VrmlNodeNavigationInfo *bindableNavigationInfoTop();
    void bindablePush(VrmlNodeNavigationInfo *);
    void bindableRemove(VrmlNodeNavigationInfo *);

    // Viewpoint
    void addViewpoint(VrmlNodeViewpoint *);
    void removeViewpoint(VrmlNodeViewpoint *);
    VrmlNodeViewpoint *bindableViewpointTop();
    void bindablePush(VrmlNodeViewpoint *);
    void bindableRemove(VrmlNodeViewpoint *);

    // Viewpoint navigation
    void nextViewpoint();
    void prevViewpoint();
    int nViewpoints();
    void getViewpoint(int, const char **, const char **);
    void setViewpoint(const char *, const char *);
    void setViewpoint(int);

    // Other (non-bindable) node types that the scene needs access to:

    // Scene-scoped lights
    void addScopedLight(VrmlNodeLight *);
    void removeScopedLight(VrmlNodeLight *);

    // Scripts
    void addScript(VrmlNodeScript *);
    void removeScript(VrmlNodeScript *);

    // TimeSensors
    void addTimeSensor(VrmlNodeTimeSensor *);
    void removeTimeSensor(VrmlNodeTimeSensor *);

    // AudioClips
    void addAudioClip(VrmlNodeAudioClip *);
    void removeAudioClip(VrmlNodeAudioClip *);

    // MovieTextures
    void addMovie(VrmlNodeMovieTexture *);
    void removeMovie(VrmlNodeMovieTexture *);

    VrmlNode *getRoot()
    {
        return &d_nodes;
    }
    void resetViewpoint()
    {
        resetVPFlag = true;
    }

    coEventQueue *getIncomingSensorEventQueue()
    {
        return d_incomingSensorEventQueue;
    }
    coEventQueue *getSensorEventQueue()
    {
        return d_sensorEventQueue;
    }

    void storeCachedInline(const char *name, const Viewer::Object d_viewerObject);
    Viewer::Object getCachedInline(const char *name);

    bool wasEncrypted() const;

protected:
    bool headlightOn();
    void doCallbacks(int reason);

    void setMenuVisible(bool);
    // Document URL
    Doc *d_url;
    Doc *d_urlLocal;

    // Scene graph
    VrmlNodeGroup d_nodes;

    // Nodes and node types defined in this scope
    VrmlNamespace *d_namespace;

    // Need render
    bool d_modified;

    // New viewpoint has been bound
    bool d_newView;

    // Time until next update
    double d_deltaTime;

    // Allow requests to load urls, nodes to wait until events are processed
    VrmlMFString *d_pendingUrl;
    VrmlMFString *d_pendingParameters;

    VrmlMFNode *d_pendingNodes;
    VrmlNamespace *d_pendingScope;

    // Functions to call when world is changed
    typedef std::list<SceneCB> SceneCBList;
    SceneCBList d_sceneCallbacks;
    // frame rate
    double d_frameRate;

    // Bindable children nodes (pseudo-stacks - allow random access deletion).
    typedef VrmlNodeList *BindStack;

    // Generic bindable children stack operations
    VrmlNode *bindableTop(BindStack);
    void bindablePush(BindStack, VrmlNode *);
    void bindableRemove(BindStack, VrmlNode *);
    void bindableRemoveAll(BindStack);

    //   Background
    VrmlNodeList *d_backgrounds; // All backgrounds
    BindStack d_backgroundStack; // Background stack

    //   Fog
    VrmlNodeList *d_fogs; // All fog nodes
    BindStack d_fogStack; // Fog stack

    //   NavigationInfo
    VrmlNodeList *d_navigationInfos; // All navigation info nodes
    BindStack d_navigationInfoStack; // Navigation info stack

    //   Viewpoint
    VrmlNodeList *d_viewpoints; // All viewpoint nodes
    BindStack d_viewpointStack; // Viewpoint stack

    // An event has a value and a destination, and is associated with a time
    typedef struct
    {
        double timeStamp;
        VrmlField *value;
        VrmlNode *toNode;
        const char *toEventIn;
    } Event;

    // For each scene can have a limited number of pending events.
    // Repeatedly allocating/freeing events is slow (it would be
    // nice to get rid of the field cloning, too), and if there are
    // so many events pending, we are probably running too slow to
    // handle them effectively anyway.
    // The event queue ought to be sorted by timeStamp...
    //static const int MAXEVENTS = 400; MSVC++5 doesn't like this.
    enum
    {
        MAXEVENTS = 4000
    };
    Event d_eventMem[MAXEVENTS];
    int d_firstEvent, d_lastEvent;

    // Scene-scoped lights (PointLights and SpotLights)
    VrmlNodeList *d_scopedLights;

    // Scripts in this scene
    VrmlNodeList *d_scripts;

    // Time sensors in this scene
    VrmlNodeList *d_timers;

    // Audio clips in this scene
    VrmlNodeList *d_audioClips;

    // Movies in this scene
    VrmlNodeList *d_movies;

    // ARSensors in this scene
    VrmlNodeList *d_arSensors;

    // old Navigationinfo to check if it changed
    VrmlNodeNavigationInfo *oldNi;

    bool resetVPFlag;

    // event queues
    coEventQueue *d_sensorEventQueue;
    coEventQueue *d_incomingSensorEventQueue;

    InlineCache *cache;

private:
    bool d_WasEncrypted;
};
}
#endif // _VRMLSCENE_
