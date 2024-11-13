/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  ScriptJS.cpp
//  javascript script objects.
//  This needs a major clean up...
//

#include "config.h"
#if HAVE_JAVASCRIPT
#include "ScriptJS.h"

#include "Doc.h"
#include "MathUtils.h"
#include "System.h"
#include "VrmlNodeScript.h"
#include "VrmlNamespace.h"
#include "VrmlNodeType.h"
#include "VrmlScene.h"

#include "VrmlSFBool.h"
#include "VrmlSFColor.h"
#include "VrmlSFColorRGBA.h"
#include "VrmlSFDouble.h"
#include "VrmlSFFloat.h"
#include "VrmlSFImage.h"
#include "VrmlSFInt.h"
#include "VrmlSFNode.h"
#include "VrmlSFRotation.h"
#include "VrmlSFString.h"
#include "VrmlSFTime.h"
#include "VrmlSFVec2d.h"
#include "VrmlSFVec3d.h"
#include "VrmlSFVec2f.h"
#include "VrmlSFVec3f.h"

#include "VrmlMFColor.h"
#include "VrmlMFColorRGBA.h"
#include "VrmlMFDouble.h"
#include "VrmlMFFloat.h"
#include "VrmlMFInt.h"
#include "VrmlMFNode.h"
#include "VrmlMFRotation.h"
#include "VrmlMFString.h"
#include "VrmlMFTime.h"
#include "VrmlMFVec2d.h"
#include "VrmlMFVec3d.h"
#include "VrmlMFVec2f.h"
#include "VrmlMFVec3f.h"

#include <string>
using std::string;
#include <sstream>
using std::ostringstream;
#include <iostream>
using std::cerr;
using std::endl;

// This is nominally a private include but I want to subclass Arrays...
#include <vrml97/js/jsarray.h>

//#define MAX_HEAP_BYTES 4L * 1024L * 1024L
//#define STACK_CHUNK_BYTES 4024L
#define MAX_HEAP_BYTES 8L * 1024L * 1024L
#define STACK_CHUNK_BYTES 8192L
namespace vrml
{
JSRuntime *ScriptJS::rt = 0; // Javascript runtime singleton object
int ScriptJS::nInstances = 0; // Number of distinct script objects

// Global class and functions

static JSClass globalClass = {
    "global", 0,
    JS_PropertyStub, JS_PropertyStub, JS_PropertyStub, JS_PropertyStub,
    JS_EnumerateStub, JS_ResolveStub, JS_ConvertStub, JS_FinalizeStub,
    NULL, NULL, NULL, NULL, NULL, NULL, 0, 0
};

static JSBool Print(JSContext *, JSObject *, uintN, jsval *, jsval *);
static JSBool PrintNl(JSContext *, JSObject *, uintN, jsval *, jsval *);
static VrmlField *jsvalToVrmlField(JSContext *cx,
                                   jsval v,
                                   VrmlField::VrmlFieldType expectType);

static JSBool vec3f_getProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp);
static JSBool vec3f_setProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp);
static void finalize(JSContext *cx, JSObject *obj);

static JSClass SFVec3fClass = {
    "SFVec3f", JSCLASS_HAS_PRIVATE,
    JS_PropertyStub, JS_PropertyStub, vec3f_getProperty, vec3f_setProperty,
    JS_EnumerateStub, JS_ResolveStub, JS_ConvertStub, finalize,
    NULL, NULL, NULL, NULL, NULL, NULL, 0, 0
};

static JSBool vec3d_getProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp);
static JSBool vec3d_setProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp);
static void finalize(JSContext *cx, JSObject *obj);

static JSClass SFVec3dClass = {
    "SFVec3d", JSCLASS_HAS_PRIVATE,
    JS_PropertyStub, JS_PropertyStub, vec3d_getProperty, vec3d_setProperty,
    JS_EnumerateStub, JS_ResolveStub, JS_ConvertStub, finalize,
    NULL, NULL, NULL, NULL, NULL, NULL, 0, 0
};

static JSFunctionSpec globalFunctions[] = {
    { "print", Print, 0, 0, 0 },
    { "printnl", PrintNl, 0, 0, 0 },
    { 0, 0, 0, 0, 0 }
};

// Browser class and functions

static JSClass browserClass = {
    "Browser", 0,
    JS_PropertyStub, JS_PropertyStub, JS_PropertyStub, JS_PropertyStub,
    JS_EnumerateStub, JS_ResolveStub, JS_ConvertStub, JS_FinalizeStub,
    NULL, NULL, NULL, NULL, NULL, NULL, 0, 0
};

static JSBool getName(JSContext *, JSObject *, uintN, jsval *, jsval *);
static JSBool getVersion(JSContext *, JSObject *, uintN, jsval *, jsval *);
static JSBool getCurrentSpeed(JSContext *, JSObject *, uintN, jsval *, jsval *);
static JSBool getCurrentFrameRate(JSContext *, JSObject *, uintN, jsval *, jsval *);
static JSBool getWorldURL(JSContext *, JSObject *, uintN, jsval *, jsval *);
static JSBool replaceWorld(JSContext *, JSObject *, uintN, jsval *, jsval *);
static JSBool createVrmlFromString(JSContext *, JSObject *, uintN, jsval *, jsval *);
static JSBool createVrmlFromURL(JSContext *, JSObject *, uintN, jsval *, jsval *);

static JSBool addRoute(JSContext *, JSObject *, uintN, jsval *, jsval *);
static JSBool deleteRoute(JSContext *, JSObject *, uintN, jsval *, jsval *);
static JSBool loadURL(JSContext *, JSObject *, uintN, jsval *, jsval *);
static JSBool setDescription(JSContext *, JSObject *, uintN, jsval *, jsval *);
static JSBool setFunctionState(JSContext *, JSObject *, uintN, jsval *, jsval *);
static JSBool setSyncMode(JSContext *, JSObject *, uintN, jsval *, jsval *);
static JSBool browserSystem(JSContext *, JSObject *, uintN, jsval *, jsval *);
static JSBool setFunctionValue(JSContext *, JSObject *, uintN, jsval *, jsval *);
static JSBool getConfigEntry(JSContext *, JSObject *, uintN, jsval *, jsval *);
static JSBool requestMasterState(JSContext *, JSObject *, uintN, jsval *, jsval *);
static JSBool isMaster(JSContext *, JSObject *, uintN, jsval *, jsval *);

static JSBool mffloatlength(JSContext *cx, JSObject *obj, uintN argc, jsval *argv, jsval *rval);
static JSBool mffloat_getProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp);
static JSBool mffloat_setProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp);

static JSFunctionSpec browserFunctions[] = {
    { "getName", getName, 0, 0, 0 },
    { "getVersion", getVersion, 0, 0, 0 },
    { "getCurrentSpeed", getCurrentSpeed, 0, 0, 0 },
    { "getCurrentFrameRate", getCurrentFrameRate, 0, 0, 0 },
    { "getWorldURL", getWorldURL, 0, 0, 0 },
    { "replaceWorld", replaceWorld, 0, 0, 0 },
    { "createVrmlFromString", createVrmlFromString, 0, 0, 0 },
    { "createVrmlFromURL", createVrmlFromURL, 0, 0, 0 },
    { "addRoute", addRoute, 0, 0, 0 },
    { "deleteRoute", deleteRoute, 0, 0, 0 },
    { "loadURL", loadURL, 0, 0, 0 },
    { "setDescription", setDescription, 0, 0, 0 },
    { "setFunctionState", setFunctionState, 0, 0, 0 },
    { "setFunctionValue", setFunctionValue, 0, 0, 0 },
    { "getConfigEntry", getConfigEntry, 0, 0, 0 },
    { "requestMasterState", requestMasterState, 0, 0, 0 },
    { "isMaster", isMaster, 0, 0, 0 },
    { "setSyncMode", setSyncMode, 0, 0, 0 },
    { "system", browserSystem, 0, 0, 0 },
    { 0, 0, 0, 0, 0 }
};

static JSBool mfint32length(JSContext *cx, JSObject *obj, uintN argc, jsval *argv, jsval *rval);

/*
static JSBool
MF_setProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
      VrmlField *f = (VrmlField *)JS_GetPrivate( cx, obj );
      if (! f) return JS_FALSE;
      switch(f->fieldType())
      {
        case VrmlField::MFFLOAT:
          {
  cerr << "MFFLOAT_setProperty" << id << endl;
}
}
cerr << "MF_setProperty" << endl;
return JS_TRUE;
}*/

static void ErrorReporter(JSContext *, const char *, JSErrorReport *);

// Construct from inline script

ScriptJS::ScriptJS(VrmlNodeScript *node, const char *source)
    : d_node(node)
    , d_cx(0)
    , d_globalObj(0)
    , d_browserObj(0)
{
    static bool firstTime = true;
    static bool sMaybeGC = true;
    static bool sForceGC = false;
    if (firstTime)
    {
        sMaybeGC = System::the->getConfigState("COVER.Plugin.Vrml97.MaybeGC", true);
        sForceGC = System::the->getConfigState("COVER.Plugin.Vrml97.ForceGC", false);
        fprintf(stderr, "Vrml97 ScriptJS: ");
        if (sForceGC)
            fprintf(stderr, "Forced garbage collection\n");
        else if (sMaybeGC)
            fprintf(stderr, "Garbage collection allowed\n");
        else
            fprintf(stderr, "No garbage collection\n");
        firstTime = false;
    }
    maybeGC = sMaybeGC;
    forceGC = sForceGC;
    if (!rt)
    {
        long int max_heap = System::the->getMaxHeapBytes();
        if (max_heap == -1)
            max_heap = MAX_HEAP_BYTES;

        rt = JS_Init(max_heap);
    }

    ++nInstances;

    if (rt)
        d_cx = JS_NewContext(rt, STACK_CHUNK_BYTES);

    if (d_cx)
    {
        JS_SetErrorReporter(d_cx, ErrorReporter);

        // Define the global objects (builtins, Browser, SF*, MF*) ...
        d_globalObj = JS_NewObject(d_cx, &globalClass, 0, 0);
        JS_InitStandardClasses(d_cx, d_globalObj);
        JS_DefineFunctions(d_cx, d_globalObj, globalFunctions);

        // VRML-like TRUE, FALSE syntax
        if (!JS_DefineProperty(d_cx, d_globalObj, "FALSE",
                               BOOLEAN_TO_JSVAL(false), 0, 0,
                               JSPROP_READONLY | JSPROP_PERMANENT))
            System::the->error("JS_DefineProp FALSE failed\n");
        if (!JS_DefineProperty(d_cx, d_globalObj, "TRUE",
                               BOOLEAN_TO_JSVAL(true), 0, 0,
                               JSPROP_READONLY | JSPROP_PERMANENT))
            System::the->error("JS_DefineProp TRUE failed\n");

        if (!JS_DefineProperty(d_cx, d_globalObj, "_script",
                               PRIVATE_TO_JSVAL(this), 0, 0,
                               JSPROP_READONLY | JSPROP_PERMANENT))
            System::the->error("JS_DefineProp _script failed\n");

        // Browser object
        d_browserObj = JS_DefineObject(d_cx, d_globalObj, "Browser",
                                       &browserClass, 0, 0);
        if (!JS_DefineProperty(d_cx, d_browserObj, "_script",
                               PRIVATE_TO_JSVAL(this), 0, 0,
                               JSPROP_READONLY | JSPROP_PERMANENT))
            System::the->error("JS_DefineProp _script failed\n");

        if (!JS_DefineFunctions(d_cx, d_browserObj, browserFunctions))
            System::the->error("JS_DefineFunctions failed\n");

        // Define SF*/MF* classes
        defineAPI();

        // Define field/eventOut vars for this script
        defineFields();

        /* These should indicate source location for diagnostics. */
        char *filename = 0;
        uintN lineno = 0;

        jsval rval;
        if (!JS_EvaluateScript(d_cx, d_globalObj, source, (uintN)strlen(source),
                               filename, lineno, &rval))
            System::the->error("JS_EvaluateScript failed\n");
    }
}

ScriptJS::~ScriptJS()
{
    JS_DestroyContext(d_cx);
    if (--nInstances == 0)
    {
        JS_Finish(rt);
        rt = NULL;
    }
}

static double s_timeStamp; // go away...

// Run a specified script

void ScriptJS::activate(double timeStamp,
                        const char *fname,
                        int argc,
                        const VrmlField *argv[])
{
    if (!d_cx)
        return;

    jsval fval, rval;

    if (!JS_LookupProperty(d_cx, d_globalObj, fname, &fval))
        System::the->error("JS_LookupProperty %s failed\n", fname);

    else if (!JSVAL_IS_VOID(fval))
    {
        jsval *jsargvl;
        jsargvl = new jsval[argc];

        d_timeStamp = timeStamp;
        s_timeStamp = timeStamp; // this won't work for long...
        int i;
        // convert VrmlField*'s to (gc-protected) jsvals
        for (i = 0; i < argc; ++i)
            jsargvl[i] = argv[i] ? vrmlFieldToJSVal(argv[i]->fieldType(), argv[i], true) : JSVAL_NULL;

        if (!JS_CallFunctionValue(d_cx, d_globalObj, fval, argc, jsargvl, &rval))
            System::the->error("JS_CallFunctionName(%s) failed\n", fname);

        // Free up args
        for (i = 0; i < argc; ++i)
            if (JSVAL_IS_GCTHING(jsargvl[i]))
                JS_RemoveRoot(d_cx, JSVAL_TO_GCTHING(jsargvl[i]));
        delete[] jsargvl;
    }
    // call the garbage collector
    if (forceGC)
        JS_GC(d_cx);
    else if (maybeGC)
        JS_MaybeGC(d_cx);
}

// Get a handle to the scene from a ScriptJS
VrmlScene *ScriptJS::browser() { return d_node->browser(); }

// Get a handle to the ScriptJS from the global object
static ScriptJS *objToScript(JSContext *cx, JSObject *obj)
{
    jsval val;
    if (!JS_LookupProperty(cx, obj, "_script", &val))
        System::the->error("JS_LookupProperty _script failed\n");

    else if (!JSVAL_IS_INT(val)) // Actually JSVAL_IS_PRIVATE
        System::the->error("JS_LookupProperty _script failed- NULL _script\n");

    return (ScriptJS *)JSVAL_TO_PRIVATE(val);
}

// Get a pointer to an eventOut from an object
static char *objToEventOut(JSContext *cx, JSObject *obj)
{
    jsval val;
    if (JS_LookupProperty(cx, obj, "_eventOut", &val) && JSVAL_IS_INT(val)) // Actually JSVAL_IS_PRIVATE
        return (char *)JSVAL_TO_PRIVATE(val);
    return 0;
}

// Check whether we are modifying a prop on an eventOut object, and if so,
// notify the script.

static void checkEventOut(JSContext *cx, JSObject *obj, VrmlField *val)
{
    char *eventOut = 0;
    ScriptJS *script = 0;

    if ((eventOut = objToEventOut(cx, obj)) != 0 && (script = objToScript(cx, JS_GetParent(cx, obj))) != 0)
    {
        VrmlNodeScript *scriptNode = script->scriptNode();
        scriptNode->setEventOut(eventOut, val);
    }
}

// data converters

static JSBool floatsToJSArray(int n, float *f, JSContext *cx, jsval *rval)
{
    const int MAX_FIXED = 20;
    jsval jsfixed[MAX_FIXED];
    jsval *jsvec = (n > MAX_FIXED) ? new jsval[n] : jsfixed;

    int i;
    for (i = 0; i < n; ++i)
    {
        jsdouble *d = JS_NewDouble(cx, f[i]);
        if (d && JS_AddRoot(cx, d))
            jsvec[i] = DOUBLE_TO_JSVAL(d);
        else
            break;
    }

    if (i == n)
    {
        JSObject *arr = JS_NewArrayObject(cx, n, jsvec);
        if (arr)
            *rval = OBJECT_TO_JSVAL(arr);
    }

    for (int j = 0; j < i; ++j)
        JS_RemoveRoot(cx, JSVAL_TO_GCTHING(jsvec[j]));
    if (n > MAX_FIXED)
        delete[] jsvec;
    return (i == n) ? JS_TRUE : JS_FALSE;
}

static JSBool doublesToJSArray(int n, double *d, JSContext *cx, jsval *rval)
{
    const int MAX_FIXED = 20;
    jsval jsfixed[MAX_FIXED];
    jsval *jsvec = (n > MAX_FIXED) ? new jsval[n] : jsfixed;

    int i;
    for (i = 0; i < n; ++i)
    {
        jsdouble *dd = JS_NewDouble(cx, d[i]);
        if (dd && JS_AddRoot(cx, dd))
            jsvec[i] = DOUBLE_TO_JSVAL(dd);
        else
            break;
    }

    if (i == n)
    {
        JSObject *arr = JS_NewArrayObject(cx, n, jsvec);
        if (arr)
            *rval = OBJECT_TO_JSVAL(arr);
    }

    for (int j = 0; j < i; ++j)
        JS_RemoveRoot(cx, JSVAL_TO_GCTHING(jsvec[j]));
    if (n > MAX_FIXED)
        delete[] jsvec;
    return (i == n) ? JS_TRUE : JS_FALSE;
}

// Generic VrmlField finalize

static void
finalize(JSContext *cx, JSObject *obj)
{
    VrmlField *f = (VrmlField *)JS_GetPrivate(cx, obj);
    if (f)
        delete f;
    JS_SetPrivate(cx, obj, 0);
}

static JSBool
toString(JSContext *cx, JSObject *obj, uintN, jsval *, jsval *rval)
{
    VrmlField *f = (VrmlField *)JS_GetPrivate(cx, obj);
    if (f)
    {
        ostringstream os;
        os << (*f) << '\0';
        string ss = os.str();
        JSString *s = JS_NewStringCopyZ(cx, ss.c_str());
        if (s)
        {
            *rval = STRING_TO_JSVAL(s);
            return JS_TRUE;
        }
    }
    return JS_FALSE;
}

//
//  Classes precede the methods since the methods may access
//  objects of the various classes.
//

// SFColor class

static JSBool
SFColorCons(JSContext *cx, JSObject *obj,
            uintN argc, jsval *argv, jsval *)
{
    jsdouble r = 0.0, g = 0.0, b = 0.0;
    for (unsigned int i = 0; i < argc; ++i)
        switch (i)
        {
        case 0:
            JS_ValueToNumber(cx, argv[i], &r);
            break;
        case 1:
            JS_ValueToNumber(cx, argv[i], &g);
            break;
        case 2:
            JS_ValueToNumber(cx, argv[i], &b);
            break;
        default: /* Too many args */
            break;
        }

    VrmlSFColor *c = new VrmlSFColor((float)r, (float)g, (float)b);
    JS_SetPrivate(cx, obj, c);
    return JS_TRUE;
}

// SFColorRGBA class

static JSBool
SFColorRGBACons(JSContext *cx, JSObject *obj,
                uintN argc, jsval *argv, jsval *)
{
    jsdouble r = 0.0, g = 0.0, b = 0.0, a = 1.0;
    for (unsigned int i = 0; i < argc; ++i)
        switch (i)
        {
        case 0:
            JS_ValueToNumber(cx, argv[i], &r);
            break;
        case 1:
            JS_ValueToNumber(cx, argv[i], &g);
            break;
        case 2:
            JS_ValueToNumber(cx, argv[i], &b);
            break;
        case 3:
            JS_ValueToNumber(cx, argv[i], &a);
            break;
        default: /* Too many args */
            break;
        }

    VrmlSFColorRGBA *c = new VrmlSFColorRGBA((float)r, (float)g, (float)b, (float)a);
    JS_SetPrivate(cx, obj, c);
    return JS_TRUE;
}

/*
static JSBool
SFFloatCons(JSContext *cx, JSObject *obj,
       uintN argc, jsval *argv, jsval *)
{
  jsdouble va = 0.0;
  JS_ValueToNumber( cx, argv[9], &va );
  for (unsigned int i=0; i<argc; ++i)
    switch (i) {
    case 0: JS_ValueToNumber( cx, argv[i], &r ); break;
    case 1: JS_ValueToNumber( cx, argv[i], &g ); break;
case 2: JS_ValueToNumber( cx, argv[i], &b ); break;
default:  break;// Too many args
}

VrmlSFColor *c = new VrmlSFFloat( v );
JS_SetPrivate(cx, obj, c);
return JS_TRUE;
}*/

static JSBool
color_getProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
    VrmlSFColor *c = (VrmlSFColor *)JS_GetPrivate(cx, obj);

    if (c && JSVAL_IS_INT(id))
        switch (JSVAL_TO_INT(id))
        {
        case 0:
            *vp = DOUBLE_TO_JSVAL(JS_NewDouble(cx, c->r()));
            break;
        case 1:
            *vp = DOUBLE_TO_JSVAL(JS_NewDouble(cx, c->g()));
            break;
        case 2:
            *vp = DOUBLE_TO_JSVAL(JS_NewDouble(cx, c->b()));
            break;
        default:
            return JS_FALSE;
        }
    return JS_TRUE;
}

static JSBool
color_setProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
    VrmlSFColor *c = (VrmlSFColor *)JS_GetPrivate(cx, obj);
    jsdouble d;

    if (c && JSVAL_IS_INT(id) && JSVAL_TO_INT(id) >= 0 && JSVAL_TO_INT(id) < 3 && JS_ValueToNumber(cx, *vp, &d))
    {
        c->get()[JSVAL_TO_INT(id)] = (float)d; // very nice...
        checkEventOut(cx, obj, c);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSClass SFColorClass = {
    "SFColor", JSCLASS_HAS_PRIVATE,
    JS_PropertyStub, JS_PropertyStub, color_getProperty, color_setProperty,
    JS_EnumerateStub, JS_ResolveStub, JS_ConvertStub, finalize,
    NULL, NULL, NULL, NULL, NULL, NULL, 0, 0
};

static JSBool
colorRGBA_getProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
    VrmlSFColorRGBA *c = (VrmlSFColorRGBA *)JS_GetPrivate(cx, obj);

    if (c && JSVAL_IS_INT(id))
        switch (JSVAL_TO_INT(id))
        {
        case 0:
            *vp = DOUBLE_TO_JSVAL(JS_NewDouble(cx, c->r()));
            break;
        case 1:
            *vp = DOUBLE_TO_JSVAL(JS_NewDouble(cx, c->g()));
            break;
        case 2:
            *vp = DOUBLE_TO_JSVAL(JS_NewDouble(cx, c->b()));
            break;
        case 3:
            *vp = DOUBLE_TO_JSVAL(JS_NewDouble(cx, c->a()));
            break;
        default:
            return JS_FALSE;
        }
    return JS_TRUE;
}

static JSBool
colorRGBA_setProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
    VrmlSFColorRGBA *c = (VrmlSFColorRGBA *)JS_GetPrivate(cx, obj);
    jsdouble d;

    if (c && JSVAL_IS_INT(id) && JSVAL_TO_INT(id) >= 0 && JSVAL_TO_INT(id) < 4 && JS_ValueToNumber(cx, *vp, &d))
    {
        c->get()[JSVAL_TO_INT(id)] = (float)d; // very nice...
        checkEventOut(cx, obj, c);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSClass SFColorRGBAClass = {
    "SFColorRGBA", JSCLASS_HAS_PRIVATE,
    JS_PropertyStub, JS_PropertyStub, colorRGBA_getProperty, colorRGBA_setProperty,
    JS_EnumerateStub, JS_ResolveStub, JS_ConvertStub, finalize,
    NULL, NULL, NULL, NULL, NULL, NULL, 0, 0
};

static JSBool
mffloatlength(JSContext *cx, JSObject *obj, uintN /*argc*/, jsval * /*argv*/, jsval *rval)
{

    VrmlMFFloat *mf = (VrmlMFFloat *)JS_GetPrivate(cx, obj);
    if (mf)
        *rval = INT_TO_JSVAL(mf->size());
    else
        return JS_FALSE;
    return JS_TRUE;
}

static JSBool
mfint32length(JSContext *cx, JSObject *obj, uintN /*argc*/, jsval * /*argv*/, jsval *rval)
{

    VrmlMFInt *mf = (VrmlMFInt *)JS_GetPrivate(cx, obj);
    if (mf)
        *rval = INT_TO_JSVAL(mf->size());
    else
        return JS_FALSE;
    return JS_TRUE;
}

static JSBool
mffloat_setProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
    VrmlMFFloat *mf = (VrmlMFFloat *)JS_GetPrivate(cx, obj);
    jsdouble d;
    if (mf && JSVAL_IS_INT(id) && JS_ValueToNumber(cx, *vp, &d))
    {
        mf->setSingle(JSVAL_TO_INT(id), (float)d);
        checkEventOut(cx, obj, mf);
    }
    else
        return JS_FALSE;
    return JS_TRUE;
}

static JSBool
mffloat_getProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
    VrmlMFFloat *mf = (VrmlMFFloat *)JS_GetPrivate(cx, obj);
    if (mf && JSVAL_IS_INT(id))
        *vp = DOUBLE_TO_JSVAL(JS_NewDouble(cx, (*mf)[JSVAL_TO_INT(id)]));
    return JS_TRUE;
}

static JSClass MFFloatClass = {
    "MFFloat", JSCLASS_HAS_PRIVATE,
    JS_PropertyStub, JS_PropertyStub, mffloat_getProperty, mffloat_setProperty,
    JS_EnumerateStub, JS_ResolveStub, JS_ConvertStub, finalize,
    NULL, NULL, NULL, NULL, NULL, NULL, 0, 0
};

static JSBool
mfdouble_setProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
    VrmlMFDouble *mf = (VrmlMFDouble *)JS_GetPrivate(cx, obj);
    jsdouble d;
    if (mf && JSVAL_IS_INT(id) && JS_ValueToNumber(cx, *vp, &d))
    {
        mf->setSingle(JSVAL_TO_INT(id), (double)d);
        checkEventOut(cx, obj, mf);
    }
    else
        return JS_FALSE;
    return JS_TRUE;
}

static JSBool
mfdouble_getProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
    VrmlMFDouble *mf = (VrmlMFDouble *)JS_GetPrivate(cx, obj);
    if (mf && JSVAL_IS_INT(id))
        *vp = DOUBLE_TO_JSVAL(JS_NewDouble(cx, (*mf)[JSVAL_TO_INT(id)]));
    return JS_TRUE;
}

static JSClass MFDoubleClass = {
    "MFDouble", JSCLASS_HAS_PRIVATE,
    JS_PropertyStub, JS_PropertyStub, mfdouble_getProperty, mfdouble_setProperty,
    JS_EnumerateStub, JS_ResolveStub, JS_ConvertStub, finalize,
    NULL, NULL, NULL, NULL, NULL, NULL, 0, 0
};

static JSClass MFTimeClass = {
    "MFTime", JSCLASS_HAS_PRIVATE,
    JS_PropertyStub, JS_PropertyStub, mfdouble_getProperty, mfdouble_setProperty,
    JS_EnumerateStub, JS_ResolveStub, JS_ConvertStub, finalize,
    NULL, NULL, NULL, NULL, NULL, NULL, 0, 0
};

// SFImage class
// This is ugly because the spec requires an MFInt array for the pixel data
// and because giving write access to the image size parameters can cause
// the library code to crash unless they are validated somehow...

static JSBool
SFImageCons(JSContext *cx, JSObject *obj,
            uintN argc, jsval *argv, jsval *)
{
    int w = 0, h = 0, nc = 0;
    unsigned char *pixels = 0;
    jsdouble x;

    for (unsigned int i = 0; i < argc; ++i)
        switch (i)
        {
        case 0:
            JS_ValueToNumber(cx, argv[i], &x);
            w = (int)x;
            break;
        case 1:
            JS_ValueToNumber(cx, argv[i], &x);
            h = (int)x;
            break;
        case 2:
            JS_ValueToNumber(cx, argv[i], &x);
            nc = (int)x;
            break;
        case 3: /* MFInt image data */
            if (JSVAL_IS_OBJECT(argv[i]) && JS_IsArrayObject(cx, JSVAL_TO_OBJECT(argv[i])))
            {
                jsuint len;
                JS_GetArrayLength(cx, JSVAL_TO_OBJECT(argv[i]), &len);
                if ((int)len != w * h)
                    return JS_FALSE;
                pixels = new unsigned char[nc * w * h];
                unsigned char *pp = pixels;
                for (int ii = 0; ii < (int)len; ++ii, pp += nc)
                {
                    static unsigned int byteMask[] = { 0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000 };
                    jsval elt;
                    JS_GetElement(cx, JSVAL_TO_OBJECT(argv[i]), (jsint)ii, &elt);
                    if (JSVAL_IS_NUMBER(elt))
                    {
                        jsdouble factor;
                        JS_ValueToNumber(cx, elt, &factor);
                        for (int j = 0; j < nc; ++j)
                            pp[j] = (((int)factor) & byteMask[j]) >> (8 * j);
                    }
                }
            }
            break;
        default: /* Too many args */
            break;
        }

    VrmlSFImage *c = new VrmlSFImage(w, h, nc, pixels);
    JS_SetPrivate(cx, obj, c);
    return JS_TRUE;
}

static JSBool
image_getProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
    VrmlSFImage *c = (VrmlSFImage *)JS_GetPrivate(cx, obj);
    if (!c)
        return JS_FALSE;
    if (JSVAL_IS_INT(id))
        switch (JSVAL_TO_INT(id))
        {
        case 0:
            *vp = INT_TO_JSVAL(c->width());
            break;
        case 1:
            *vp = INT_TO_JSVAL(c->height());
            break;
        case 2:
            *vp = INT_TO_JSVAL(c->nComponents());
            break;
        case 3:
            break; // *vp = convert pixels to MFInt...
        default:
            return JS_FALSE;
        }

    return JS_TRUE;
}

static JSBool
image_setProperty(JSContext *, JSObject *, jsval, jsval *)
{
    // ...
    return JS_FALSE;
}

static JSClass SFImageClass = {
    "SFImage", JSCLASS_HAS_PRIVATE,
    JS_PropertyStub, JS_PropertyStub, image_getProperty, image_setProperty,
    JS_EnumerateStub, JS_ResolveStub, JS_ConvertStub, finalize,
    NULL, NULL, NULL, NULL, NULL, NULL, 0, 0
};

// SFNode class

static JSBool
SFNodeCons(JSContext *cx, JSObject *obj,
           uintN argc, jsval *argv, jsval *)
{
    JSString *str;

    if (argc == 1 && JSVAL_IS_STRING(argv[0]) && (str = JSVAL_TO_STRING(argv[0])) != 0)
    {
        // If the nodes created contain scripts, they might want to access the
        // namespace...
        VrmlNamespace ns;
        VrmlMFNode *nodes = VrmlScene::readString(JS_GetStringBytes(str), &ns);
        if (!nodes || nodes->size() == 0)
            return JS_FALSE;

        // If there are multiple top-level nodes, wrap them in a Group. SPEC?
        VrmlNode *n;
        if (nodes->size() == 1)
            n = nodes->get()[0];
        else
        {
            VrmlNodeGroup *g = new VrmlNodeGroup();
            g->addChildren(*nodes);
            n = g;
        }

        VrmlSFNode *sfnode = new VrmlSFNode(n);
        JS_SetPrivate(cx, obj, sfnode);
        delete nodes;
        return JS_TRUE;
    }

    return JS_FALSE;
}

// SFNode getProperty reads eventOut values, setProperty sends eventIns.

static JSBool
node_getProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
    VrmlSFNode *n = (VrmlSFNode *)JS_GetPrivate(cx, obj);
    JSString *str = 0;
    ScriptJS *script = 0;

    if (n && n->get() && JSVAL_IS_STRING(id) && ((str = JSVAL_TO_STRING(id)) != 0) && (script = objToScript(cx, JS_GetParent(cx, obj))) != 0)
    {
        char *eventOut = JS_GetStringBytes(str);
        const VrmlField *fieldVal = n->get()->getEventOut(eventOut);

        if (fieldVal == NULL)
        {
            fieldVal = n->get()->getField(eventOut);
        }
        // convert event out value to jsval...
        if (fieldVal)
        {
            *vp = script->vrmlFieldToJSVal(fieldVal->fieldType(), fieldVal, false);
            return JS_TRUE;
        }

        System::the->error("unrecognized exposedField/eventOut: node.getProperty(%s)\n", JS_GetStringBytes(str));
    }

    return JS_FALSE;
}

static JSBool
node_setProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
    VrmlSFNode *sfn = (VrmlSFNode *)JS_GetPrivate(cx, obj);
    VrmlNode *n = sfn ? sfn->get() : 0;

    if (n && JSVAL_IS_STRING(id))
    {
        JSString *str = JSVAL_TO_STRING(id);
        char *eventIn = str ? JS_GetStringBytes(str) : 0;
        char *setEventIn = NULL;
        VrmlField::VrmlFieldType expect;
        expect = n->nodeType()->hasEventIn(eventIn);
        if (!expect)
        {
            VrmlNodeScript *scriptNode;
            if ((scriptNode = n->as<VrmlNodeScript>()))
            {
                // Handle exposedFields
                expect = scriptNode->hasExposedField(eventIn);
                if (!expect)
                    expect = scriptNode->hasEventIn(eventIn);
            }
        }
        if (!expect)
        {
            setEventIn = new char[strlen(eventIn) + 5];
            strcpy(setEventIn, "set_");
            strcat(setEventIn, eventIn);
            expect = n->nodeType()->hasEventIn(setEventIn);
            VrmlNodeScript *scriptNode;
            if ((scriptNode = n->as<VrmlNodeScript>()))
            {
                if (!expect)
                    expect = scriptNode->hasEventIn(setEventIn);
            }
        }

        if (expect && eventIn) // convert vp to field, send eventIn to n
        {

            VrmlField *f = jsvalToVrmlField(cx, *vp, expect);
// This should only happen if directOutput is set...
#ifdef DEBUG
            cout << "ScriptJS::node_setProperty sending " << eventIn
                 << " (" << (*f) << ") to "
                 << n->nodeType()->getName() << "::"
                 << n->name() << endl;
#endif

            // the timestamp should be stored as a global property and
            // looked up via obj somehow...
            //ScriptJS *script = objToScript( cx, JS_GetParent( cx, obj ));
            //if (script && script->browser())
            //  script->browser()->queueEvent( s_timeStamp, f, n, eventIn );
            n->setField(eventIn, *f);
            if (n->scene())
                n->scene()->queueEvent(s_timeStamp, f, n, eventIn);
        }

        checkEventOut(cx, obj, sfn);
        if (setEventIn)
            delete[] setEventIn;
    }
    return JS_TRUE;
}

static JSClass SFNodeClass = {
    "SFNode", JSCLASS_HAS_PRIVATE,
    JS_PropertyStub, JS_PropertyStub, node_getProperty, node_setProperty,
    JS_EnumerateStub, JS_ResolveStub, JS_ConvertStub, finalize,
    NULL, NULL, NULL, NULL, NULL, NULL, 0, 0
};

// SFRotation class

static JSBool
SFRotationCons(JSContext *cx, JSObject *obj,
               uintN argc, jsval *argv, jsval *)
{
    jsdouble x = 0.0, y = 1.0, z = 0.0, angle = 0.0;
    //extern JSClass SFVec3fClass;

    if (argc > 0)
    {
        if (JSVAL_IS_NUMBER(argv[0]))
        {
            JS_ValueToNumber(cx, argv[0], &x);
            if (argc > 1)
                JS_ValueToNumber(cx, argv[1], &y);
            if (argc > 2)
                JS_ValueToNumber(cx, argv[2], &z);
            if (argc > 3)
                JS_ValueToNumber(cx, argv[3], &angle);
        }
        else if (argc == 2 && JSVAL_IS_OBJECT(argv[0]) && &SFVec3fClass == JS_GetClass(JSVAL_TO_OBJECT(argv[0])))
        {
            JSObject *v1obj = JSVAL_TO_OBJECT(argv[0]);
            VrmlSFVec3f *v1 = (VrmlSFVec3f *)JS_GetPrivate(cx, v1obj);
            if (JSVAL_IS_NUMBER(argv[1]))
            {
                x = v1->x(); // axis/angle
                y = v1->y();
                z = v1->z();
                JS_ValueToNumber(cx, argv[1], &angle);
            }
            else if (JSVAL_IS_OBJECT(argv[1]) && &SFVec3fClass == JS_GetClass(JSVAL_TO_OBJECT(argv[1])))
            {
                JSObject *v2obj = JSVAL_TO_OBJECT(argv[1]);
                VrmlSFVec3f *v2 = (VrmlSFVec3f *)JS_GetPrivate(cx, v2obj);
                VrmlSFVec3f v3(*v1); // find rotation that takes v1 to v2
                v3.cross(v2);
                x = v3.x();
                y = v3.y();
                z = v3.z();
                angle = acos(v1->dot(v2) / (v1->length() * v2->length()));
            }
            else
                return JS_FALSE;
        }
        else
            return JS_FALSE;
    }

    VrmlSFRotation *v = new VrmlSFRotation((float)x, (float)y, (float)z, (float)angle);
    JS_SetPrivate(cx, obj, v);
    return JS_TRUE;
}

static JSBool
rot_getProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
    VrmlSFRotation *v = (VrmlSFRotation *)JS_GetPrivate(cx, obj);
    if (v && JSVAL_IS_INT(id))
        switch (JSVAL_TO_INT(id))
        {
        case 0:
            *vp = DOUBLE_TO_JSVAL(JS_NewDouble(cx, v->x()));
            break;
        case 1:
            *vp = DOUBLE_TO_JSVAL(JS_NewDouble(cx, v->y()));
            break;
        case 2:
            *vp = DOUBLE_TO_JSVAL(JS_NewDouble(cx, v->z()));
            break;
        case 3:
            *vp = DOUBLE_TO_JSVAL(JS_NewDouble(cx, v->r()));
            break;
        default:
            return JS_FALSE;
        }
    return JS_TRUE;
}

static JSBool
rot_setProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
    VrmlSFRotation *v = (VrmlSFRotation *)JS_GetPrivate(cx, obj);
    jsdouble d;

    if (v && JSVAL_IS_INT(id) && JSVAL_TO_INT(id) >= 0 && JSVAL_TO_INT(id) < 4 && JS_ValueToNumber(cx, *vp, &d))
    {
        v->get()[JSVAL_TO_INT(id)] = (float)d;
        checkEventOut(cx, obj, v);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSClass SFRotationClass = {
    "SFRotation", JSCLASS_HAS_PRIVATE,
    JS_PropertyStub, JS_PropertyStub, rot_getProperty, rot_setProperty,
    JS_EnumerateStub, JS_ResolveStub, JS_ConvertStub, finalize,
    NULL, NULL, NULL, NULL, NULL, NULL, 0, 0
};

// SFVec2d class

static JSBool
SFVec2dCons(JSContext *cx, JSObject *obj,
            uintN argc, jsval *argv, jsval *)
{
    jsdouble x = 0.0, y = 0.0;
    if (argc > 0)
        JS_ValueToNumber(cx, argv[0], &x);
    if (argc > 1)
        JS_ValueToNumber(cx, argv[1], &y);
    VrmlSFVec2d *v = new VrmlSFVec2d((double)x, (double)y);
    JS_SetPrivate(cx, obj, v);
    return JS_TRUE;
}

static JSBool
vec2d_getProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
    VrmlSFVec2d *v = (VrmlSFVec2d *)JS_GetPrivate(cx, obj);
    if (v && JSVAL_IS_INT(id))
        switch (JSVAL_TO_INT(id))
        {
        case 0:
            *vp = DOUBLE_TO_JSVAL(JS_NewDouble(cx, v->x()));
            break;
        case 1:
            *vp = DOUBLE_TO_JSVAL(JS_NewDouble(cx, v->y()));
            break;
        default:
            return JS_FALSE;
        }
    return JS_TRUE;
}

static JSBool
vec2d_setProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
    VrmlSFVec2d *v = (VrmlSFVec2d *)JS_GetPrivate(cx, obj);
    jsdouble d;

    if (v && JSVAL_IS_INT(id) && JSVAL_TO_INT(id) >= 0 && JSVAL_TO_INT(id) < 2 && JS_ValueToNumber(cx, *vp, &d))
    {
        v->get()[JSVAL_TO_INT(id)] = (double)d;
        checkEventOut(cx, obj, v);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSClass SFVec2dClass = {
    "SFVec2d", JSCLASS_HAS_PRIVATE,
    JS_PropertyStub, JS_PropertyStub, vec2d_getProperty, vec2d_setProperty,
    JS_EnumerateStub, JS_ResolveStub, JS_ConvertStub, finalize,
    NULL, NULL, NULL, NULL, NULL, NULL, 0, 0
};

// SFVec2f class

static JSBool
SFVec2fCons(JSContext *cx, JSObject *obj,
            uintN argc, jsval *argv, jsval *)
{
    jsdouble x = 0.0, y = 0.0;
    if (argc > 0)
        JS_ValueToNumber(cx, argv[0], &x);
    if (argc > 1)
        JS_ValueToNumber(cx, argv[1], &y);
    VrmlSFVec2f *v = new VrmlSFVec2f((float)x, (float)y);
    JS_SetPrivate(cx, obj, v);
    return JS_TRUE;
}

static JSBool
vec2f_getProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
    VrmlSFVec2f *v = (VrmlSFVec2f *)JS_GetPrivate(cx, obj);
    if (v && JSVAL_IS_INT(id))
        switch (JSVAL_TO_INT(id))
        {
        case 0:
            *vp = DOUBLE_TO_JSVAL(JS_NewDouble(cx, v->x()));
            break;
        case 1:
            *vp = DOUBLE_TO_JSVAL(JS_NewDouble(cx, v->y()));
            break;
        default:
            return JS_FALSE;
        }
    return JS_TRUE;
}

static JSBool
vec2f_setProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
    VrmlSFVec2f *v = (VrmlSFVec2f *)JS_GetPrivate(cx, obj);
    jsdouble d;

    if (v && JSVAL_IS_INT(id) && JSVAL_TO_INT(id) >= 0 && JSVAL_TO_INT(id) < 2 && JS_ValueToNumber(cx, *vp, &d))
    {
        v->get()[JSVAL_TO_INT(id)] = (float)d;
        checkEventOut(cx, obj, v);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSClass SFVec2fClass = {
    "SFVec2f", JSCLASS_HAS_PRIVATE,
    JS_PropertyStub, JS_PropertyStub, vec2f_getProperty, vec2f_setProperty,
    JS_EnumerateStub, JS_ResolveStub, JS_ConvertStub, finalize,
    NULL, NULL, NULL, NULL, NULL, NULL, 0, 0
};

// SFVec3d class

static JSBool
SFVec3dCons(JSContext *cx, JSObject *obj,
            uintN argc, jsval *argv, jsval *)
{
    jsdouble x = 0.0, y = 0.0, z = 0.0;
    if (argc > 0)
        JS_ValueToNumber(cx, argv[0], &x);
    if (argc > 1)
        JS_ValueToNumber(cx, argv[1], &y);
    if (argc > 2)
        JS_ValueToNumber(cx, argv[2], &z);
    VrmlSFVec3d *v = new VrmlSFVec3d((double)x, (double)y, (double)z);
    JS_SetPrivate(cx, obj, v);
    return JS_TRUE;
}

static JSBool
vec3d_getProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
    VrmlSFVec3d *v = (VrmlSFVec3d *)JS_GetPrivate(cx, obj);
    if (v && JSVAL_IS_INT(id))
        switch (JSVAL_TO_INT(id))
        {
        case 0:
            *vp = DOUBLE_TO_JSVAL(JS_NewDouble(cx, v->x()));
            break;
        case 1:
            *vp = DOUBLE_TO_JSVAL(JS_NewDouble(cx, v->y()));
            break;
        case 2:
            *vp = DOUBLE_TO_JSVAL(JS_NewDouble(cx, v->z()));
            break;
        default:
            return JS_FALSE;
        }
    return JS_TRUE;
}

static JSBool
vec3d_setProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
    VrmlSFVec3d *v = (VrmlSFVec3d *)JS_GetPrivate(cx, obj);
    jsdouble d;

    if (v && JSVAL_IS_INT(id) && JSVAL_TO_INT(id) >= 0 && JSVAL_TO_INT(id) < 3 && JS_ValueToNumber(cx, *vp, &d))
    {
        v->get()[JSVAL_TO_INT(id)] = (float)d;
        checkEventOut(cx, obj, v);
        return JS_TRUE;
    }

    return JS_FALSE;
}

// SFVec3f class

static JSBool
SFVec3fCons(JSContext *cx, JSObject *obj,
            uintN argc, jsval *argv, jsval *)
{
    jsdouble x = 0.0, y = 0.0, z = 0.0;
    if (argc > 0)
        JS_ValueToNumber(cx, argv[0], &x);
    if (argc > 1)
        JS_ValueToNumber(cx, argv[1], &y);
    if (argc > 2)
        JS_ValueToNumber(cx, argv[2], &z);
    VrmlSFVec3f *v = new VrmlSFVec3f((float)x, (float)y, (float)z);
    JS_SetPrivate(cx, obj, v);
    return JS_TRUE;
}

static JSBool
vec3f_getProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
    VrmlSFVec3f *v = (VrmlSFVec3f *)JS_GetPrivate(cx, obj);
    if (v && JSVAL_IS_INT(id))
        switch (JSVAL_TO_INT(id))
        {
        case 0:
            *vp = DOUBLE_TO_JSVAL(JS_NewDouble(cx, v->x()));
            break;
        case 1:
            *vp = DOUBLE_TO_JSVAL(JS_NewDouble(cx, v->y()));
            break;
        case 2:
            *vp = DOUBLE_TO_JSVAL(JS_NewDouble(cx, v->z()));
            break;
        default:
            return JS_FALSE;
        }
    return JS_TRUE;
}

static JSBool
vec3f_setProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
    VrmlSFVec3f *v = (VrmlSFVec3f *)JS_GetPrivate(cx, obj);
    jsdouble d;

    if (v && JSVAL_IS_INT(id) && JSVAL_TO_INT(id) >= 0 && JSVAL_TO_INT(id) < 3 && JS_ValueToNumber(cx, *vp, &d))
    {
        v->get()[JSVAL_TO_INT(id)] = (float)d;
        checkEventOut(cx, obj, v);
        return JS_TRUE;
    }

    return JS_FALSE;
}

// MFColor class

static JSBool
MFColorCons(JSContext *cx, JSObject *obj,
            uintN argc, jsval *argv, jsval *rval)
{
    obj = JS_NewArrayObject(cx, (jsint)argc, argv);
    *rval = OBJECT_TO_JSVAL(obj);
    return obj == 0 ? JS_FALSE : JS_TRUE;
}

static JSClass MFColorClass = {
    "MFColor", 0,
    JS_PropertyStub, JS_PropertyStub, JS_PropertyStub, JS_PropertyStub,
    JS_EnumerateStub, JS_ResolveStub, JS_ConvertStub, JS_FinalizeStub,
    NULL, NULL, NULL, NULL, NULL, NULL, 0, 0
};

// MFColorRGBA class

static JSBool
MFColorRGBACons(JSContext *cx, JSObject *obj,
                uintN argc, jsval *argv, jsval *rval)
{
    obj = JS_NewArrayObject(cx, (jsint)argc, argv);
    *rval = OBJECT_TO_JSVAL(obj);
    return obj == 0 ? JS_FALSE : JS_TRUE;
}

static JSClass MFColorRGBAClass = {
    "MFColorRGBA", 0,
    JS_PropertyStub, JS_PropertyStub, JS_PropertyStub, JS_PropertyStub,
    JS_EnumerateStub, JS_ResolveStub, JS_ConvertStub, JS_FinalizeStub,
    NULL, NULL, NULL, NULL, NULL, NULL, 0, 0
};

// MFFloat class

static JSBool
MFFloatCons(JSContext *cx, JSObject *obj,
            uintN argc, jsval *argv, jsval *rval)
{
    obj = JS_NewArrayObject(cx, (jsint)argc, argv);
    VrmlMFFloat *mf = new VrmlMFFloat();
    JS_SetPrivate(cx, obj, mf);
    *rval = OBJECT_TO_JSVAL(obj);
    return obj == 0 ? JS_FALSE : JS_TRUE;
}

// MFInt32 class

static JSBool
mfint_getProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
    VrmlMFInt *mf = (VrmlMFInt *)JS_GetPrivate(cx, obj);
    if (mf && JSVAL_IS_INT(id))
        *vp = INT_TO_JSVAL((*mf)[JSVAL_TO_INT(id)]);
    return JS_TRUE;
}

static JSBool
mfint_setProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
    VrmlMFInt *mf = (VrmlMFInt *)JS_GetPrivate(cx, obj);
    int d;
    if (mf && JSVAL_IS_INT(id))
    {
        d = JSVAL_TO_INT(*vp);
        mf->setSingle(JSVAL_TO_INT(id), d);
        checkEventOut(cx, obj, mf);
    }
    else
        return JS_FALSE;
    return JS_TRUE;
}

static JSBool
MFInt32Cons(JSContext *cx, JSObject *obj,
            uintN argc, jsval *argv, jsval *rval)
{
    obj = JS_NewArrayObject(cx, (jsint)argc, argv);
    *rval = OBJECT_TO_JSVAL(obj);
    return obj == 0 ? JS_FALSE : JS_TRUE;
}

static JSClass MFInt32Class = {
    "MFInt32", JSCLASS_HAS_PRIVATE,
    JS_PropertyStub, JS_PropertyStub, mfint_getProperty, mfint_setProperty,
    JS_EnumerateStub, JS_ResolveStub, JS_ConvertStub, JS_FinalizeStub,
    NULL, NULL, NULL, NULL, NULL, NULL, 0, 0
};

// MFNode class

static JSBool
MFNodeCons(JSContext *cx, JSObject *obj,
           uintN argc, jsval *argv, jsval *rval)
{
    obj = JS_NewArrayObject(cx, (jsint)argc, argv);
    *rval = OBJECT_TO_JSVAL(obj);
    return obj == 0 ? JS_FALSE : JS_TRUE;
}

static JSClass MFNodeClass = {
    "MFNode", 0,
    JS_PropertyStub, JS_PropertyStub, JS_PropertyStub, JS_PropertyStub,
    JS_EnumerateStub, JS_ResolveStub, JS_ConvertStub, JS_FinalizeStub,
    NULL, NULL, NULL, NULL, NULL, NULL, 0, 0
};

// MFRotation class

static JSBool
MFRotationCons(JSContext *cx, JSObject *obj,
               uintN argc, jsval *argv, jsval *rval)
{
    obj = JS_NewArrayObject(cx, (jsint)argc, argv);
    *rval = OBJECT_TO_JSVAL(obj);
    return obj == 0 ? JS_FALSE : JS_TRUE;
}

static JSClass MFRotationClass = {
    "MFRotation", 0,
    JS_PropertyStub, JS_PropertyStub, JS_PropertyStub, JS_PropertyStub,
    JS_EnumerateStub, JS_ResolveStub, JS_ConvertStub, JS_FinalizeStub,
    NULL, NULL, NULL, NULL, NULL, NULL, 0, 0
};

// MFString class

static JSBool
MFStringCons(JSContext *cx, JSObject *obj,
             uintN argc, jsval *argv, jsval *rval)
{
    obj = JS_NewArrayObject(cx, (jsint)argc, argv);
    *rval = OBJECT_TO_JSVAL(obj);
    return obj == 0 ? JS_FALSE : JS_TRUE;
}

static JSClass MFStringClass = {
    "MFString", 0,
    JS_PropertyStub, JS_PropertyStub, JS_PropertyStub, JS_PropertyStub,
    JS_EnumerateStub, JS_ResolveStub, JS_ConvertStub, JS_FinalizeStub,
    NULL, NULL, NULL, NULL, NULL, NULL, 0, 0
};

// MFTime class

static JSBool
MFTimeCons(JSContext *cx, JSObject *obj,
           uintN argc, jsval *argv, jsval *rval)
{
    obj = JS_NewArrayObject(cx, (jsint)argc, argv);
    VrmlMFTime *mf = new VrmlMFTime();
    JS_SetPrivate(cx, obj, mf);
    *rval = OBJECT_TO_JSVAL(obj);
    return obj == 0 ? JS_FALSE : JS_TRUE;
}

// MFVec2d class

static JSBool
mfvec2d_getProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
    VrmlMFVec2d *mf = (VrmlMFVec2d *)JS_GetPrivate(cx, obj);
    if (mf && JSVAL_IS_INT(id))
    {

        JSObject *sfobj = JS_NewObject(cx, &SFVec3fClass, 0, obj);
        if (!sfobj)
            return JSVAL_NULL;
        double *val = mf->get();
        val += 3 * JSVAL_TO_INT(id);
        JS_SetPrivate(cx, sfobj, new VrmlSFVec2d(val[0], val[1]));

        *vp = OBJECT_TO_JSVAL(sfobj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
mfvec2d_setProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
    VrmlMFVec2d *mf = (VrmlMFVec2d *)JS_GetPrivate(cx, obj);
    if (mf && JSVAL_IS_INT(id))
    {
        VrmlSFVec2d *sfobj = (VrmlSFVec2d *)JS_GetPrivate(cx, JSVAL_TO_OBJECT(*vp));
        if (sfobj != NULL)
        {
            mf->setSingle(JSVAL_TO_INT(id), sfobj->get());
            checkEventOut(cx, obj, mf);
        }
    }
    else
        return JS_FALSE;
    return JS_TRUE;
}

static JSBool
MFVec2dCons(JSContext *cx, JSObject *obj,
            uintN argc, jsval *argv, jsval *rval)
{
    obj = JS_NewArrayObject(cx, (jsint)argc, argv);
    *rval = OBJECT_TO_JSVAL(obj);
    return obj == 0 ? JS_FALSE : JS_TRUE;
}

static JSClass MFVec2dClass = {
    "MFVec2d", 0,
    JS_PropertyStub, JS_PropertyStub, mfvec2d_getProperty, mfvec2d_setProperty,
    JS_EnumerateStub, JS_ResolveStub, JS_ConvertStub, JS_FinalizeStub,
    NULL, NULL, NULL, NULL, NULL, NULL, 0, 0
};

// MFVec2f class

static JSBool
mfvec2f_getProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
    VrmlMFVec3f *mf = (VrmlMFVec3f *)JS_GetPrivate(cx, obj);
    if (mf && JSVAL_IS_INT(id))
    {

        JSObject *sfobj = JS_NewObject(cx, &SFVec3fClass, 0, obj);
        if (!sfobj)
            return JSVAL_NULL;
        float *val = mf->get();
        val += 2 * JSVAL_TO_INT(id);
        JS_SetPrivate(cx, sfobj, new VrmlSFVec2f(val[0], val[1]));

        *vp = OBJECT_TO_JSVAL(sfobj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
mfvec2f_setProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
    VrmlMFVec2f *mf = (VrmlMFVec2f *)JS_GetPrivate(cx, obj);
    if (mf && JSVAL_IS_INT(id))
    {
        VrmlSFVec2f *sfobj = (VrmlSFVec2f *)JS_GetPrivate(cx, JSVAL_TO_OBJECT(*vp));
        if (sfobj != NULL)
        {
            mf->setSingle(JSVAL_TO_INT(id), sfobj->get());
            checkEventOut(cx, obj, mf);
        }
    }
    else
        return JS_FALSE;
    return JS_TRUE;
}

static JSBool
MFVec2fCons(JSContext *cx, JSObject *obj,
            uintN argc, jsval *argv, jsval *rval)
{
    obj = JS_NewArrayObject(cx, (jsint)argc, argv);
    *rval = OBJECT_TO_JSVAL(obj);
    return obj == 0 ? JS_FALSE : JS_TRUE;
}

static JSClass MFVec2fClass = {
    "MFVec2f", 0,
    JS_PropertyStub, JS_PropertyStub, mfvec2f_getProperty, mfvec2f_setProperty,
    JS_EnumerateStub, JS_ResolveStub, JS_ConvertStub, JS_FinalizeStub,
    NULL, NULL, NULL, NULL, NULL, NULL, 0, 0
};

// MFVec3d class

static JSBool
mfvec3d_getProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
    VrmlMFVec3d *mf = (VrmlMFVec3d *)JS_GetPrivate(cx, obj);
    if (mf && JSVAL_IS_INT(id))
    {

        JSObject *sfobj = JS_NewObject(cx, &SFVec3fClass, 0, obj);
        if (!sfobj)
            return JSVAL_NULL;
        double *val = mf->get();
        val += 3 * JSVAL_TO_INT(id);
        JS_SetPrivate(cx, sfobj, new VrmlSFVec3d(val[0], val[1], val[2]));

        *vp = OBJECT_TO_JSVAL(sfobj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
mfvec3d_setProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
    VrmlMFVec3d *mf = (VrmlMFVec3d *)JS_GetPrivate(cx, obj);
    if (mf && JSVAL_IS_INT(id))
    {
        VrmlSFVec3d *sfobj = (VrmlSFVec3d *)JS_GetPrivate(cx, JSVAL_TO_OBJECT(*vp));
        if (sfobj != NULL)
        {
            mf->setSingle(JSVAL_TO_INT(id), sfobj->get());
            checkEventOut(cx, obj, mf);
        }
    }
    else
        return JS_FALSE;
    return JS_TRUE;
}

static JSBool
MFVec3dCons(JSContext *cx, JSObject *obj,
            uintN argc, jsval *argv, jsval *rval)
{
    obj = JS_NewArrayObject(cx, (jsint)argc, argv);
    *rval = OBJECT_TO_JSVAL(obj);
    return obj == 0 ? JS_FALSE : JS_TRUE;
}

static JSClass MFVec3dClass = {
    "MFVec3d", 0,
    JS_PropertyStub, JS_PropertyStub, mfvec3d_getProperty, mfvec3d_setProperty,
    JS_EnumerateStub, JS_ResolveStub, JS_ConvertStub, JS_FinalizeStub,
    NULL, NULL, NULL, NULL, NULL, NULL, 0, 0
};

// MFVec3f class

static JSBool
mfvec3f_getProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
    VrmlMFVec3f *mf = (VrmlMFVec3f *)JS_GetPrivate(cx, obj);
    if (mf && JSVAL_IS_INT(id))
    {

        JSObject *sfobj = JS_NewObject(cx, &SFVec3fClass, 0, obj);
        if (!sfobj)
            return JSVAL_NULL;
        float *val = mf->get();
        val += 3 * JSVAL_TO_INT(id);
        JS_SetPrivate(cx, sfobj, new VrmlSFVec3f(val[0], val[1], val[2]));

        *vp = OBJECT_TO_JSVAL(sfobj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
mfvec3f_setProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
    VrmlMFVec3f *mf = (VrmlMFVec3f *)JS_GetPrivate(cx, obj);
    if (mf && JSVAL_IS_INT(id))
    {
        VrmlSFVec3f *sfobj = (VrmlSFVec3f *)JS_GetPrivate(cx, JSVAL_TO_OBJECT(*vp));
        if (sfobj != NULL)
        {
            mf->setSingle(JSVAL_TO_INT(id), sfobj->get());
            checkEventOut(cx, obj, mf);
        }
    }
    else
        return JS_FALSE;
    return JS_TRUE;
}

static JSBool
MFVec3fCons(JSContext *cx, JSObject *obj,
            uintN argc, jsval *argv, jsval *rval)
{
    obj = JS_NewArrayObject(cx, (jsint)argc, argv);
    *rval = OBJECT_TO_JSVAL(obj);
    return obj == 0 ? JS_FALSE : JS_TRUE;
}

static JSClass MFVec3fClass = {
    "MFVec3f", 0,
    JS_PropertyStub, JS_PropertyStub, mfvec3f_getProperty, mfvec3f_setProperty,
    JS_EnumerateStub, JS_ResolveStub, JS_ConvertStub, JS_FinalizeStub,
    NULL, NULL, NULL, NULL, NULL, NULL, 0, 0
};

// VrmlMatrix class

static JSBool
VrmlMatrixCons(JSContext *cx, JSObject *obj,
               uintN /*argc*/, jsval * /*argv*/, jsval *)
{
    double *m = new double[16];
    JS_SetPrivate(cx, obj, m);
    cerr << "new Matrix" << endl;
    return JS_TRUE;
}

static JSBool
matrix_getProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
    cerr << "matrix_getProperty" << endl;
    double *m = (double *)JS_GetPrivate(cx, obj);
    if (m && JSVAL_IS_INT(id))
        switch (JSVAL_TO_INT(id))
        {
        case 0:
            if (doublesToJSArray(16, m, cx, vp) == JS_FALSE)
                return JS_FALSE;
            break;
        default:
            return JS_FALSE;
        }
    cerr << "matrix_getProperty successfull" << endl;
    return JS_TRUE;
}

static JSBool
matrix_setProperty(JSContext *cx, JSObject *obj, jsval id, jsval *vp)
{
    cerr << "matrix_setProperty" << endl;
    double *m = (double *)JS_GetPrivate(cx, obj);
    jsdouble d;

    if (m && JSVAL_IS_INT(id) && JSVAL_TO_INT(id) >= 0 && JSVAL_TO_INT(id) < 16 && JS_ValueToNumber(cx, *vp, &d))
    {
        m[JSVAL_TO_INT(id)] = d;
        cerr << "matrix_setProperty successfull" << endl;
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSClass VrmlMatrixClass = {
    "VrmlMatrix", JSCLASS_HAS_PRIVATE,
    JS_PropertyStub, JS_PropertyStub, matrix_getProperty, matrix_setProperty,
    JS_EnumerateStub, JS_ResolveStub, JS_ConvertStub, finalize,
    NULL, NULL, NULL, NULL, NULL, NULL, 0, 0
};

//
//  Methods
//

// SFColor methods

static JSBool
color_setHSV(JSContext *cx, JSObject *obj, uintN argc, jsval *argv,
             jsval *)
{
    jsdouble h, s, v;
    VrmlSFColor *c = (VrmlSFColor *)JS_GetPrivate(cx, obj);
    if (c && argc == 3 && JS_ValueToNumber(cx, argv[0], &h) && JS_ValueToNumber(cx, argv[1], &s) && JS_ValueToNumber(cx, argv[2], &v))
    {
        c->setHSV((float)h, (float)s, (float)v);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
color_getHSV(JSContext *cx, JSObject *obj, uintN, jsval *,
             jsval *rval)
{
    VrmlSFColor *c = (VrmlSFColor *)JS_GetPrivate(cx, obj);
    if (c)
    {
        float hsv[3];
        c->getHSV(hsv[0], hsv[1], hsv[2]);
        return floatsToJSArray(3, hsv, cx, rval);
    }
    return JS_FALSE;
}

// SFColorRGBA methods

static JSBool
colorRGBA_setHSV(JSContext *cx, JSObject *obj, uintN argc, jsval *argv,
                 jsval *)
{
    jsdouble h, s, v;
    VrmlSFColorRGBA *c = (VrmlSFColorRGBA *)JS_GetPrivate(cx, obj);
    if (c && argc == 3 && JS_ValueToNumber(cx, argv[0], &h) && JS_ValueToNumber(cx, argv[1], &s) && JS_ValueToNumber(cx, argv[2], &v))
    {
        c->setHSV((float)h, (float)s, (float)v);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
colorRGBA_getHSV(JSContext *cx, JSObject *obj, uintN, jsval *,
                 jsval *rval)
{
    VrmlSFColorRGBA *c = (VrmlSFColorRGBA *)JS_GetPrivate(cx, obj);
    if (c)
    {
        float hsv[3];
        c->getHSV(hsv[0], hsv[1], hsv[2]);
        return floatsToJSArray(3, hsv, cx, rval);
    }
    return JS_FALSE;
}

// SFRotation methods

static JSBool
rot_getAxis(JSContext *cx, JSObject *obj, uintN, jsval *,
            jsval *rval)
{
    VrmlSFRotation *r = (VrmlSFRotation *)JS_GetPrivate(cx, obj);
    if (r)
        return floatsToJSArray(3, r->get(), cx, rval);
    return JS_FALSE;
}

static JSBool
rot_inverse(JSContext *cx, JSObject *obj, uintN, jsval *,
            jsval *rval)
{
    VrmlSFRotation *r = (VrmlSFRotation *)JS_GetPrivate(cx, obj);
    JSObject *robj = 0;

    if (r && (robj = JS_NewObject(cx, &SFRotationClass, 0, JS_GetParent(cx, obj))) != 0)
    {
        VrmlSFRotation *r2 = (VrmlSFRotation *)(r->clone());
        r2->invert();
        JS_SetPrivate(cx, robj, r2);
        *rval = OBJECT_TO_JSVAL(robj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
rot_multiply(JSContext *cx, JSObject *obj, uintN argc, jsval *argv,
             jsval *rval)
{
    VrmlSFRotation *r1 = (VrmlSFRotation *)JS_GetPrivate(cx, obj);
    VrmlSFRotation *r2 = 0;
    JSObject *r3obj = 0;

    // Verify that we have 2 SFRotation's && create a result
    if (r1 && argc > 0 && JSVAL_IS_OBJECT(argv[0]) && &SFRotationClass == JS_GetClass(JSVAL_TO_OBJECT(argv[0])) && (r2 = (VrmlSFRotation *)JS_GetPrivate(cx, JSVAL_TO_OBJECT(argv[0]))) && (((VrmlField *)r2)->fieldType() == VrmlField::SFROTATION) && (r3obj = JS_NewObject(cx, &SFRotationClass, 0, JS_GetParent(cx, obj))) != 0)
    {
        VrmlSFRotation *r3 = (VrmlSFRotation *)(r1->clone());
        r3->multiply(r2);
        JS_SetPrivate(cx, r3obj, r3);
        *rval = OBJECT_TO_JSVAL(r3obj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
rot_multVec(JSContext *cx, JSObject *obj, uintN argc, jsval *argv,
            jsval *rval)
{
    VrmlSFRotation *r1 = (VrmlSFRotation *)JS_GetPrivate(cx, obj);
    VrmlSFVec3f *v2 = 0;
    JSObject *v3obj = 0;

    // Verify that we have an SFVec3f arg && create a result
    if (r1 && argc > 0 && JSVAL_IS_OBJECT(argv[0]) && &SFVec3fClass == JS_GetClass(JSVAL_TO_OBJECT(argv[0])) && (v2 = (VrmlSFVec3f *)JS_GetPrivate(cx, JSVAL_TO_OBJECT(argv[0]))) && (((VrmlField *)v2)->fieldType() == VrmlField::SFVEC3F) && (v3obj = JS_NewObject(cx, &SFVec3fClass, 0, JS_GetParent(cx, obj))) != 0)
    {
        VrmlSFVec3f *v3 = new VrmlSFVec3f;
        double M[16];
        Mrotation(M, r1->get());
        VM(v3->get(), M, v2->get());
        JS_SetPrivate(cx, v3obj, v3);
        *rval = OBJECT_TO_JSVAL(v3obj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

// Should this create a new object or modify obj?...

static JSBool
rot_setAxis(JSContext *cx, JSObject *obj, uintN argc, jsval *argv,
            jsval *rval)
{
    VrmlSFRotation *r1 = (VrmlSFRotation *)JS_GetPrivate(cx, obj);
    VrmlSFVec3f *v2 = 0;

    // Verify that we have an SFVec3f arg && create a result
    if (r1 && argc > 0 && JSVAL_IS_OBJECT(argv[0]) && &SFVec3fClass == JS_GetClass(JSVAL_TO_OBJECT(argv[0])) && (v2 = (VrmlSFVec3f *)JS_GetPrivate(cx, JSVAL_TO_OBJECT(argv[0]))) && (((VrmlField *)v2)->fieldType() == VrmlField::SFVEC3F))
    {
        r1->set(v2->x(), v2->y(), v2->z(), r1->r());
        *rval = OBJECT_TO_JSVAL(obj); // return the same object
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
rot_slerp(JSContext *cx, JSObject *obj, uintN argc, jsval *argv,
          jsval *rval)
{
    VrmlSFRotation *r1 = (VrmlSFRotation *)JS_GetPrivate(cx, obj);
    VrmlSFRotation *r2 = 0;
    JSObject *r3obj = 0;

    // Verify that we have 2 SFRotation's, a number, & create a result
    if (r1 && argc > 1 && JSVAL_IS_OBJECT(argv[0]) && &SFRotationClass == JS_GetClass(JSVAL_TO_OBJECT(argv[0])) && (r2 = (VrmlSFRotation *)JS_GetPrivate(cx, JSVAL_TO_OBJECT(argv[0]))) && (((VrmlField *)r2)->fieldType() == VrmlField::SFROTATION) && JSVAL_IS_NUMBER(argv[1]) && (r3obj = JS_NewObject(cx, &SFRotationClass, 0, JS_GetParent(cx, obj))) != 0)
    {
        VrmlSFRotation *r3 = (VrmlSFRotation *)(r1->clone());
        jsdouble factor;
        JS_ValueToNumber(cx, argv[1], &factor);
        r3->slerp(r2, (float)factor);
        JS_SetPrivate(cx, r3obj, r3);
        *rval = OBJECT_TO_JSVAL(r3obj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

// SFVec2d methods

static JSBool
vec2d_add(JSContext *cx, JSObject *obj, uintN argc, jsval *argv, jsval *rval)
{
    VrmlSFVec2d *v1 = (VrmlSFVec2d *)JS_GetPrivate(cx, obj);
    VrmlSFVec2d *v2 = 0;
    JSObject *v3obj = 0;

    // Verify that we have 2 SFVec2d's && create a result
    if (v1 && argc > 0 && JSVAL_IS_OBJECT(argv[0]) && &SFVec2dClass == JS_GetClass(JSVAL_TO_OBJECT(argv[0])) && (v2 = (VrmlSFVec2d *)JS_GetPrivate(cx, JSVAL_TO_OBJECT(argv[0]))) && (((VrmlField *)v2)->fieldType() == VrmlField::SFVEC2D) && (v3obj = JS_NewObject(cx, &SFVec2dClass, 0, JS_GetParent(cx, obj))) != 0)
    {
        VrmlSFVec2d *v3 = (VrmlSFVec2d *)(v1->clone());
        v3->add(v2);
        JS_SetPrivate(cx, v3obj, v3);
        *rval = OBJECT_TO_JSVAL(v3obj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
vec2d_divide(JSContext *cx, JSObject *obj, uintN argc, jsval *argv, jsval *rval)
{
    VrmlSFVec2d *v1 = (VrmlSFVec2d *)JS_GetPrivate(cx, obj);
    jsdouble d = 0.0;
    JSObject *v3obj = 0;

    // Verify that we have a number && create a result
    if (v1 && argc > 0 && JS_ValueToNumber(cx, argv[0], &d) && (v3obj = JS_NewObject(cx, &SFVec2dClass, 0, JS_GetParent(cx, obj))) != 0)
    {
        VrmlSFVec2d *v3 = (VrmlSFVec2d *)(v1->clone());
        v3->divide(d);
        JS_SetPrivate(cx, v3obj, v3);
        *rval = OBJECT_TO_JSVAL(v3obj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
vec2d_dot(JSContext *cx, JSObject *obj, uintN argc, jsval *argv, jsval *rval)
{
    VrmlSFVec2d *v1 = (VrmlSFVec2d *)JS_GetPrivate(cx, obj);
    VrmlSFVec2d *v2 = 0;
    JSObject *v3obj = 0;

    // Verify that we have 2 SFVec2d's && create a result
    if (v1 && argc > 0 && JSVAL_IS_OBJECT(argv[0]) && &SFVec2dClass == JS_GetClass(JSVAL_TO_OBJECT(argv[0])) && (v2 = (VrmlSFVec2d *)JS_GetPrivate(cx, JSVAL_TO_OBJECT(argv[0]))) && (((VrmlField *)v2)->fieldType() == VrmlField::SFVEC2D) && (v3obj = JS_NewObject(cx, &SFVec2dClass, 0, JS_GetParent(cx, obj))) != 0)
    {
        VrmlSFVec2d *v3 = (VrmlSFVec2d *)(v1->clone());
        v3->dot(v2);
        JS_SetPrivate(cx, v3obj, v3);
        *rval = OBJECT_TO_JSVAL(v3obj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
vec2d_length(JSContext *cx, JSObject *obj, uintN, jsval *, jsval *rval)
{
    VrmlSFVec2d *v = (VrmlSFVec2d *)JS_GetPrivate(cx, obj);
    if (v)
    {
        *rval = DOUBLE_TO_JSVAL(JS_NewDouble(cx, v->length()));
        return JS_TRUE;
    }
    return JS_FALSE;
}

static JSBool
vec2d_multiply(JSContext *cx, JSObject *obj, uintN argc, jsval *argv, jsval *rval)
{
    VrmlSFVec2d *v1 = (VrmlSFVec2d *)JS_GetPrivate(cx, obj);
    jsdouble d = 0.0;
    JSObject *v3obj = 0;

    // Verify that we have a number && create a result
    if (v1 && argc > 0 && JS_ValueToNumber(cx, argv[0], &d) && (v3obj = JS_NewObject(cx, &SFVec2dClass, 0, JS_GetParent(cx, obj))) != 0)
    {
        VrmlSFVec2d *v3 = (VrmlSFVec2d *)(v1->clone());
        v3->multiply(d);
        JS_SetPrivate(cx, v3obj, v3);
        *rval = OBJECT_TO_JSVAL(v3obj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
vec2d_negate(JSContext *cx, JSObject *obj, uintN, jsval *, jsval *rval)
{
    VrmlSFVec2d *v1 = (VrmlSFVec2d *)JS_GetPrivate(cx, obj);
    JSObject *v3obj = 0;

    if (v1 && (v3obj = JS_NewObject(cx, &SFVec2dClass, 0, JS_GetParent(cx, obj))) != 0)
    {
        VrmlSFVec2d *v3 = (VrmlSFVec2d *)(v1->clone());
        v3->multiply(-1.0);
        JS_SetPrivate(cx, v3obj, v3);
        *rval = OBJECT_TO_JSVAL(v3obj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
vec2d_normalize(JSContext *cx, JSObject *obj, uintN, jsval *, jsval *rval)
{
    VrmlSFVec2d *v1 = (VrmlSFVec2d *)JS_GetPrivate(cx, obj);
    JSObject *v3obj = 0;

    if (v1 && (v3obj = JS_NewObject(cx, &SFVec2dClass, 0, JS_GetParent(cx, obj))) != 0)
    {
        VrmlSFVec2d *v3 = (VrmlSFVec2d *)(v1->clone());
        v3->normalize();
        JS_SetPrivate(cx, v3obj, v3);
        *rval = OBJECT_TO_JSVAL(v3obj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
vec2d_subtract(JSContext *cx, JSObject *obj, uintN argc, jsval *argv, jsval *rval)
{
    VrmlSFVec2d *v1 = (VrmlSFVec2d *)JS_GetPrivate(cx, obj);
    VrmlSFVec2d *v2 = 0;
    JSObject *v3obj = 0;

    // Verify that we have 2 SFVec2d's && create a result
    if (v1 && argc > 0 && JSVAL_IS_OBJECT(argv[0]) && &SFVec2dClass == JS_GetClass(JSVAL_TO_OBJECT(argv[0])) && (v2 = (VrmlSFVec2d *)JS_GetPrivate(cx, JSVAL_TO_OBJECT(argv[0]))) && (((VrmlField *)v2)->fieldType() == VrmlField::SFVEC2D) && (v3obj = JS_NewObject(cx, &SFVec2dClass, 0, JS_GetParent(cx, obj))) != 0)
    {
        VrmlSFVec2d *v3 = (VrmlSFVec2d *)(v1->clone());
        v3->subtract(v2);
        JS_SetPrivate(cx, v3obj, v3);
        *rval = OBJECT_TO_JSVAL(v3obj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

// SFVec2f methods

static JSBool
vec2f_add(JSContext *cx, JSObject *obj, uintN argc, jsval *argv, jsval *rval)
{
    VrmlSFVec2f *v1 = (VrmlSFVec2f *)JS_GetPrivate(cx, obj);
    VrmlSFVec2f *v2 = 0;
    JSObject *v3obj = 0;

    // Verify that we have 2 SFVec2f's && create a result
    if (v1 && argc > 0 && JSVAL_IS_OBJECT(argv[0]) && &SFVec2fClass == JS_GetClass(JSVAL_TO_OBJECT(argv[0])) && (v2 = (VrmlSFVec2f *)JS_GetPrivate(cx, JSVAL_TO_OBJECT(argv[0]))) && (((VrmlField *)v2)->fieldType() == VrmlField::SFVEC2F) && (v3obj = JS_NewObject(cx, &SFVec2fClass, 0, JS_GetParent(cx, obj))) != 0)
    {
        VrmlSFVec2f *v3 = (VrmlSFVec2f *)(v1->clone());
        v3->add(v2);
        JS_SetPrivate(cx, v3obj, v3);
        *rval = OBJECT_TO_JSVAL(v3obj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
vec2f_divide(JSContext *cx, JSObject *obj, uintN argc, jsval *argv, jsval *rval)
{
    VrmlSFVec2f *v1 = (VrmlSFVec2f *)JS_GetPrivate(cx, obj);
    jsdouble d = 0.0;
    JSObject *v3obj = 0;

    // Verify that we have a number && create a result
    if (v1 && argc > 0 && JS_ValueToNumber(cx, argv[0], &d) && (v3obj = JS_NewObject(cx, &SFVec2fClass, 0, JS_GetParent(cx, obj))) != 0)
    {
        VrmlSFVec2f *v3 = (VrmlSFVec2f *)(v1->clone());
        v3->divide((float)d);
        JS_SetPrivate(cx, v3obj, v3);
        *rval = OBJECT_TO_JSVAL(v3obj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
vec2f_dot(JSContext *cx, JSObject *obj, uintN argc, jsval *argv, jsval *rval)
{
    VrmlSFVec2f *v1 = (VrmlSFVec2f *)JS_GetPrivate(cx, obj);
    VrmlSFVec2f *v2 = 0;
    JSObject *v3obj = 0;

    // Verify that we have 2 SFVec2f's && create a result
    if (v1 && argc > 0 && JSVAL_IS_OBJECT(argv[0]) && &SFVec2fClass == JS_GetClass(JSVAL_TO_OBJECT(argv[0])) && (v2 = (VrmlSFVec2f *)JS_GetPrivate(cx, JSVAL_TO_OBJECT(argv[0]))) && (((VrmlField *)v2)->fieldType() == VrmlField::SFVEC2F) && (v3obj = JS_NewObject(cx, &SFVec2fClass, 0, JS_GetParent(cx, obj))) != 0)
    {
        VrmlSFVec2f *v3 = (VrmlSFVec2f *)(v1->clone());
        v3->dot(v2);
        JS_SetPrivate(cx, v3obj, v3);
        *rval = OBJECT_TO_JSVAL(v3obj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
vec2f_length(JSContext *cx, JSObject *obj, uintN, jsval *, jsval *rval)
{
    VrmlSFVec2f *v = (VrmlSFVec2f *)JS_GetPrivate(cx, obj);
    if (v)
    {
        *rval = DOUBLE_TO_JSVAL(JS_NewDouble(cx, v->length()));
        return JS_TRUE;
    }
    return JS_FALSE;
}

static JSBool
vec2f_multiply(JSContext *cx, JSObject *obj, uintN argc, jsval *argv, jsval *rval)
{
    VrmlSFVec2f *v1 = (VrmlSFVec2f *)JS_GetPrivate(cx, obj);
    jsdouble d = 0.0;
    JSObject *v3obj = 0;

    // Verify that we have a number && create a result
    if (v1 && argc > 0 && JS_ValueToNumber(cx, argv[0], &d) && (v3obj = JS_NewObject(cx, &SFVec2fClass, 0, JS_GetParent(cx, obj))) != 0)
    {
        VrmlSFVec2f *v3 = (VrmlSFVec2f *)(v1->clone());
        v3->multiply((float)d);
        JS_SetPrivate(cx, v3obj, v3);
        *rval = OBJECT_TO_JSVAL(v3obj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
vec2f_negate(JSContext *cx, JSObject *obj, uintN, jsval *, jsval *rval)
{
    VrmlSFVec2f *v1 = (VrmlSFVec2f *)JS_GetPrivate(cx, obj);
    JSObject *v3obj = 0;

    if (v1 && (v3obj = JS_NewObject(cx, &SFVec2fClass, 0, JS_GetParent(cx, obj))) != 0)
    {
        VrmlSFVec2f *v3 = (VrmlSFVec2f *)(v1->clone());
        v3->multiply(-1.0);
        JS_SetPrivate(cx, v3obj, v3);
        *rval = OBJECT_TO_JSVAL(v3obj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
vec2f_normalize(JSContext *cx, JSObject *obj, uintN, jsval *, jsval *rval)
{
    VrmlSFVec2f *v1 = (VrmlSFVec2f *)JS_GetPrivate(cx, obj);
    JSObject *v3obj = 0;

    if (v1 && (v3obj = JS_NewObject(cx, &SFVec2fClass, 0, JS_GetParent(cx, obj))) != 0)
    {
        VrmlSFVec2f *v3 = (VrmlSFVec2f *)(v1->clone());
        v3->normalize();
        JS_SetPrivate(cx, v3obj, v3);
        *rval = OBJECT_TO_JSVAL(v3obj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
vec2f_subtract(JSContext *cx, JSObject *obj, uintN argc, jsval *argv, jsval *rval)
{
    VrmlSFVec2f *v1 = (VrmlSFVec2f *)JS_GetPrivate(cx, obj);
    VrmlSFVec2f *v2 = 0;
    JSObject *v3obj = 0;

    // Verify that we have 2 SFVec2f's && create a result
    if (v1 && argc > 0 && JSVAL_IS_OBJECT(argv[0]) && &SFVec2fClass == JS_GetClass(JSVAL_TO_OBJECT(argv[0])) && (v2 = (VrmlSFVec2f *)JS_GetPrivate(cx, JSVAL_TO_OBJECT(argv[0]))) && (((VrmlField *)v2)->fieldType() == VrmlField::SFVEC2F) && (v3obj = JS_NewObject(cx, &SFVec2fClass, 0, JS_GetParent(cx, obj))) != 0)
    {
        VrmlSFVec2f *v3 = (VrmlSFVec2f *)(v1->clone());
        v3->subtract(v2);
        JS_SetPrivate(cx, v3obj, v3);
        *rval = OBJECT_TO_JSVAL(v3obj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

// SFVec3f methods

static JSBool
vec3f_add(JSContext *cx, JSObject *obj, uintN argc, jsval *argv, jsval *rval)
{
    VrmlSFVec3f *v1 = (VrmlSFVec3f *)JS_GetPrivate(cx, obj);
    VrmlSFVec3f *v2 = 0;
    JSObject *v3obj = 0;

    // Verify that we have 2 SFVec3f's && create a result
    if (v1 && argc > 0 && JSVAL_IS_OBJECT(argv[0]) && &SFVec3fClass == JS_GetClass(JSVAL_TO_OBJECT(argv[0])) && (v2 = (VrmlSFVec3f *)JS_GetPrivate(cx, JSVAL_TO_OBJECT(argv[0]))) && (((VrmlField *)v2)->fieldType() == VrmlField::SFVEC3F) && (v3obj = JS_NewObject(cx, &SFVec3fClass, 0, JS_GetParent(cx, obj))) != 0)
    {
        VrmlSFVec3f *v3 = (VrmlSFVec3f *)(v1->clone());
        v3->add(v2);
        JS_SetPrivate(cx, v3obj, v3);
        *rval = OBJECT_TO_JSVAL(v3obj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
vec3f_cross(JSContext *cx, JSObject *obj, uintN argc, jsval *argv, jsval *rval)
{
    VrmlSFVec3f *v1 = (VrmlSFVec3f *)JS_GetPrivate(cx, obj);
    VrmlSFVec3f *v2 = 0;
    JSObject *v3obj = 0;

    // Verify that we have 2 SFVec3f's && create a result
    if (v1 && argc > 0 && JSVAL_IS_OBJECT(argv[0]) && &SFVec3fClass == JS_GetClass(JSVAL_TO_OBJECT(argv[0])) && (v2 = (VrmlSFVec3f *)JS_GetPrivate(cx, JSVAL_TO_OBJECT(argv[0]))) && (((VrmlField *)v2)->fieldType() == VrmlField::SFVEC3F) && (v3obj = JS_NewObject(cx, &SFVec3fClass, 0, JS_GetParent(cx, obj))) != 0)
    {
        VrmlSFVec3f *v3 = (VrmlSFVec3f *)(v1->clone());
        v3->cross(v2);
        JS_SetPrivate(cx, v3obj, v3);
        *rval = OBJECT_TO_JSVAL(v3obj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
vec3f_divide(JSContext *cx, JSObject *obj, uintN argc, jsval *argv, jsval *rval)
{
    VrmlSFVec3f *v1 = (VrmlSFVec3f *)JS_GetPrivate(cx, obj);
    jsdouble d = 0.0;
    JSObject *v3obj = 0;

    // Verify that we have a number && create a result
    if (v1 && argc > 0 && JS_ValueToNumber(cx, argv[0], &d) && (v3obj = JS_NewObject(cx, &SFVec3fClass, 0, JS_GetParent(cx, obj))) != 0)
    {
        VrmlSFVec3f *v3 = (VrmlSFVec3f *)(v1->clone());
        v3->divide((float)d);
        JS_SetPrivate(cx, v3obj, v3);
        *rval = OBJECT_TO_JSVAL(v3obj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
vec3f_dot(JSContext *cx, JSObject *obj, uintN argc, jsval *argv, jsval *rval)
{
    VrmlSFVec3f *v1 = (VrmlSFVec3f *)JS_GetPrivate(cx, obj);
    VrmlSFVec3f *v2 = 0;

    // Verify that we have 2 SFVec3f's && create a result
    if (v1 && argc > 0 && JSVAL_IS_OBJECT(argv[0]) && &SFVec3fClass == JS_GetClass(JSVAL_TO_OBJECT(argv[0])) && (v2 = (VrmlSFVec3f *)JS_GetPrivate(cx, JSVAL_TO_OBJECT(argv[0]))) && (((VrmlField *)v2)->fieldType() == VrmlField::SFVEC3F))
    {
        jsdouble *d = JS_NewDouble(cx, v1->dot(v2));
        *rval = DOUBLE_TO_JSVAL(d);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
vec3f_length(JSContext *cx, JSObject *obj, uintN, jsval *, jsval *rval)
{
    VrmlSFVec3f *v = (VrmlSFVec3f *)JS_GetPrivate(cx, obj);
    if (v)
    {
        *rval = DOUBLE_TO_JSVAL(JS_NewDouble(cx, v->length()));
        return JS_TRUE;
    }
    return JS_FALSE;
}

static JSBool
vec3f_multiply(JSContext *cx, JSObject *obj, uintN argc, jsval *argv, jsval *rval)
{
    VrmlSFVec3f *v1 = (VrmlSFVec3f *)JS_GetPrivate(cx, obj);
    jsdouble d = 0.0;
    JSObject *v3obj = 0;

    // Verify that we have a number && create a result
    if (v1 && argc > 0 && JS_ValueToNumber(cx, argv[0], &d) && (v3obj = JS_NewObject(cx, &SFVec3fClass, 0, JS_GetParent(cx, obj))) != 0)
    {
        VrmlSFVec3f *v3 = (VrmlSFVec3f *)(v1->clone());
        v3->multiply((float)d);
        JS_SetPrivate(cx, v3obj, v3);
        *rval = OBJECT_TO_JSVAL(v3obj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
vec3f_negate(JSContext *cx, JSObject *obj, uintN, jsval *, jsval *rval)
{
    VrmlSFVec3f *v1 = (VrmlSFVec3f *)JS_GetPrivate(cx, obj);
    JSObject *v3obj = 0;

    if (v1 && (v3obj = JS_NewObject(cx, &SFVec3fClass, 0, JS_GetParent(cx, obj))) != 0)
    {
        VrmlSFVec3f *v3 = (VrmlSFVec3f *)(v1->clone());
        v3->multiply(-1.0);
        JS_SetPrivate(cx, v3obj, v3);
        *rval = OBJECT_TO_JSVAL(v3obj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
vec3f_normalize(JSContext *cx, JSObject *obj, uintN, jsval *, jsval *rval)
{
    VrmlSFVec3f *v1 = (VrmlSFVec3f *)JS_GetPrivate(cx, obj);
    JSObject *v3obj = 0;

    if (v1 && (v3obj = JS_NewObject(cx, &SFVec3fClass, 0, JS_GetParent(cx, obj))) != 0)
    {
        VrmlSFVec3f *v3 = (VrmlSFVec3f *)(v1->clone());
        v3->normalize();
        JS_SetPrivate(cx, v3obj, v3);
        *rval = OBJECT_TO_JSVAL(v3obj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
vec3f_subtract(JSContext *cx, JSObject *obj, uintN argc, jsval *argv, jsval *rval)
{
    VrmlSFVec3f *v1 = (VrmlSFVec3f *)JS_GetPrivate(cx, obj);
    VrmlSFVec3f *v2 = 0;
    JSObject *v3obj = 0;

    // Verify that we have 2 SFVec3f's && create a result
    if (v1 && argc > 0 && JSVAL_IS_OBJECT(argv[0]) && &SFVec3fClass == JS_GetClass(JSVAL_TO_OBJECT(argv[0])) && (v2 = (VrmlSFVec3f *)JS_GetPrivate(cx, JSVAL_TO_OBJECT(argv[0]))) && (((VrmlField *)v2)->fieldType() == VrmlField::SFVEC3F) && (v3obj = JS_NewObject(cx, &SFVec3fClass, 0, JS_GetParent(cx, obj))) != 0)
    {
        VrmlSFVec3f *v3 = (VrmlSFVec3f *)(v1->clone());
        v3->subtract(v2);
        JS_SetPrivate(cx, v3obj, v3);
        *rval = OBJECT_TO_JSVAL(v3obj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

// SFVec3d methods

static JSBool
vec3d_add(JSContext *cx, JSObject *obj, uintN argc, jsval *argv, jsval *rval)
{
    VrmlSFVec3d *v1 = (VrmlSFVec3d *)JS_GetPrivate(cx, obj);
    VrmlSFVec3d *v2 = 0;
    JSObject *v3obj = 0;

    // Verify that we have 2 SFVec3d's && create a result
    if (v1 && argc > 0 && JSVAL_IS_OBJECT(argv[0]) && &SFVec3dClass == JS_GetClass(JSVAL_TO_OBJECT(argv[0])) && (v2 = (VrmlSFVec3d *)JS_GetPrivate(cx, JSVAL_TO_OBJECT(argv[0]))) && (((VrmlField *)v2)->fieldType() == VrmlField::SFVEC3D) && (v3obj = JS_NewObject(cx, &SFVec3dClass, 0, JS_GetParent(cx, obj))) != 0)
    {
        VrmlSFVec3d *v3 = (VrmlSFVec3d *)(v1->clone());
        v3->add(v2);
        JS_SetPrivate(cx, v3obj, v3);
        *rval = OBJECT_TO_JSVAL(v3obj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
vec3d_cross(JSContext *cx, JSObject *obj, uintN argc, jsval *argv, jsval *rval)
{
    VrmlSFVec3d *v1 = (VrmlSFVec3d *)JS_GetPrivate(cx, obj);
    VrmlSFVec3d *v2 = 0;
    JSObject *v3obj = 0;

    // Verify that we have 2 SFVec3d's && create a result
    if (v1 && argc > 0 && JSVAL_IS_OBJECT(argv[0]) && &SFVec3dClass == JS_GetClass(JSVAL_TO_OBJECT(argv[0])) && (v2 = (VrmlSFVec3d *)JS_GetPrivate(cx, JSVAL_TO_OBJECT(argv[0]))) && (((VrmlField *)v2)->fieldType() == VrmlField::SFVEC3D) && (v3obj = JS_NewObject(cx, &SFVec3dClass, 0, JS_GetParent(cx, obj))) != 0)
    {
        VrmlSFVec3d *v3 = (VrmlSFVec3d *)(v1->clone());
        v3->cross(v2);
        JS_SetPrivate(cx, v3obj, v3);
        *rval = OBJECT_TO_JSVAL(v3obj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
vec3d_divide(JSContext *cx, JSObject *obj, uintN argc, jsval *argv, jsval *rval)
{
    VrmlSFVec3d *v1 = (VrmlSFVec3d *)JS_GetPrivate(cx, obj);
    jsdouble d = 0.0;
    JSObject *v3obj = 0;

    // Verify that we have a number && create a result
    if (v1 && argc > 0 && JS_ValueToNumber(cx, argv[0], &d) && (v3obj = JS_NewObject(cx, &SFVec3dClass, 0, JS_GetParent(cx, obj))) != 0)
    {
        VrmlSFVec3d *v3 = (VrmlSFVec3d *)(v1->clone());
        v3->divide(d);
        JS_SetPrivate(cx, v3obj, v3);
        *rval = OBJECT_TO_JSVAL(v3obj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
vec3d_dot(JSContext *cx, JSObject *obj, uintN argc, jsval *argv, jsval *rval)
{
    VrmlSFVec3d *v1 = (VrmlSFVec3d *)JS_GetPrivate(cx, obj);
    VrmlSFVec3d *v2 = 0;

    // Verify that we have 2 SFVec3d's && create a result
    if (v1 && argc > 0 && JSVAL_IS_OBJECT(argv[0]) && &SFVec3dClass == JS_GetClass(JSVAL_TO_OBJECT(argv[0])) && (v2 = (VrmlSFVec3d *)JS_GetPrivate(cx, JSVAL_TO_OBJECT(argv[0]))) && (((VrmlField *)v2)->fieldType() == VrmlField::SFVEC3D))
    {
        jsdouble *d = JS_NewDouble(cx, v1->dot(v2));
        *rval = DOUBLE_TO_JSVAL(d);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
vec3d_length(JSContext *cx, JSObject *obj, uintN, jsval *, jsval *rval)
{
    VrmlSFVec3d *v = (VrmlSFVec3d *)JS_GetPrivate(cx, obj);
    if (v)
    {
        *rval = DOUBLE_TO_JSVAL(JS_NewDouble(cx, v->length()));
        return JS_TRUE;
    }
    return JS_FALSE;
}

static JSBool
vec3d_multiply(JSContext *cx, JSObject *obj, uintN argc, jsval *argv, jsval *rval)
{
    VrmlSFVec3d *v1 = (VrmlSFVec3d *)JS_GetPrivate(cx, obj);
    jsdouble d = 0.0;
    JSObject *v3obj = 0;

    // Verify that we have a number && create a result
    if (v1 && argc > 0 && JS_ValueToNumber(cx, argv[0], &d) && (v3obj = JS_NewObject(cx, &SFVec3dClass, 0, JS_GetParent(cx, obj))) != 0)
    {
        VrmlSFVec3d *v3 = (VrmlSFVec3d *)(v1->clone());
        v3->multiply(d);
        JS_SetPrivate(cx, v3obj, v3);
        *rval = OBJECT_TO_JSVAL(v3obj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
vec3d_negate(JSContext *cx, JSObject *obj, uintN, jsval *, jsval *rval)
{
    VrmlSFVec3d *v1 = (VrmlSFVec3d *)JS_GetPrivate(cx, obj);
    JSObject *v3obj = 0;

    if (v1 && (v3obj = JS_NewObject(cx, &SFVec3dClass, 0, JS_GetParent(cx, obj))) != 0)
    {
        VrmlSFVec3d *v3 = (VrmlSFVec3d *)(v1->clone());
        v3->multiply(-1.0);
        JS_SetPrivate(cx, v3obj, v3);
        *rval = OBJECT_TO_JSVAL(v3obj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
vec3d_normalize(JSContext *cx, JSObject *obj, uintN, jsval *, jsval *rval)
{
    VrmlSFVec3d *v1 = (VrmlSFVec3d *)JS_GetPrivate(cx, obj);
    JSObject *v3obj = 0;

    if (v1 && (v3obj = JS_NewObject(cx, &SFVec3dClass, 0, JS_GetParent(cx, obj))) != 0)
    {
        VrmlSFVec3d *v3 = (VrmlSFVec3d *)(v1->clone());
        v3->normalize();
        JS_SetPrivate(cx, v3obj, v3);
        *rval = OBJECT_TO_JSVAL(v3obj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
vec3d_subtract(JSContext *cx, JSObject *obj, uintN argc, jsval *argv, jsval *rval)
{
    VrmlSFVec3d *v1 = (VrmlSFVec3d *)JS_GetPrivate(cx, obj);
    VrmlSFVec3d *v2 = 0;
    JSObject *v3obj = 0;

    // Verify that we have 2 SFVec3d's && create a result
    if (v1 && argc > 0 && JSVAL_IS_OBJECT(argv[0]) && &SFVec3dClass == JS_GetClass(JSVAL_TO_OBJECT(argv[0])) && (v2 = (VrmlSFVec3d *)JS_GetPrivate(cx, JSVAL_TO_OBJECT(argv[0]))) && (((VrmlField *)v2)->fieldType() == VrmlField::SFVEC3D) && (v3obj = JS_NewObject(cx, &SFVec3dClass, 0, JS_GetParent(cx, obj))) != 0)
    {
        VrmlSFVec3d *v3 = (VrmlSFVec3d *)(v1->clone());
        v3->subtract(v2);
        JS_SetPrivate(cx, v3obj, v3);
        *rval = OBJECT_TO_JSVAL(v3obj);
        return JS_TRUE;
    }

    return JS_FALSE;
}

// VrmlMatrix methods

static JSBool
VrmlMatrix_multMatrixVec(JSContext *cx, JSObject *obj, uintN argc, jsval *argv, jsval *rval)
{
    double *m = (double *)JS_GetPrivate(cx, obj);
    VrmlSFVec3f *v2 = 0;
    JSObject *v3obj = 0;

    cerr << "VrmlMatrix_multMatrixVec" << endl;
    // Verify that we have 2 SFVec3f's && create a result
    if (m && argc > 0 && JSVAL_IS_OBJECT(argv[0]) && &SFVec3fClass == JS_GetClass(JSVAL_TO_OBJECT(argv[0])) && (v2 = (VrmlSFVec3f *)JS_GetPrivate(cx, JSVAL_TO_OBJECT(argv[0]))) && (((VrmlField *)v2)->fieldType() == VrmlField::SFVEC3F) && (v3obj = JS_NewObject(cx, &SFVec3fClass, 0, JS_GetParent(cx, obj))) != 0)
    {
        VrmlSFVec3f *v3 = new VrmlSFVec3f;
        VM(v3->get(), m, v2->get());
        JS_SetPrivate(cx, v3obj, v3);
        *rval = OBJECT_TO_JSVAL(v3obj);
        cerr << "VrmlMatrix_multMatrixVec successfull" << endl;
        return JS_TRUE;
    }

    return JS_FALSE;
}

static JSBool
VrmlMatrix_setTransform(JSContext *cx, JSObject *obj, uintN argc, jsval *argv, jsval * /*shoud we return something TODOrval*/)
{
    double *m = (double *)JS_GetPrivate(cx, obj);
    VrmlSFVec3f *translation = 0;
    VrmlSFRotation *rotation = 0;
    VrmlSFVec3f *scale = 0;
    VrmlSFRotation *scaleOrientation = 0;
    VrmlSFVec3f *center = 0;

    cerr << "VrmlMatrix_setTransform argc:" << argc << endl;
    // Verify that we have 2 SFVec3f's && create a result
    if (m && argc == 5 && JSVAL_IS_OBJECT(argv[0]) && &SFVec3fClass == JS_GetClass(JSVAL_TO_OBJECT(argv[0])) && (translation = (VrmlSFVec3f *)JS_GetPrivate(cx, JSVAL_TO_OBJECT(argv[0]))) && (((VrmlField *)translation)->fieldType() == VrmlField::SFVEC3F) && JSVAL_IS_OBJECT(argv[1]) && &SFRotationClass == JS_GetClass(JSVAL_TO_OBJECT(argv[1])) && (rotation = (VrmlSFRotation *)JS_GetPrivate(cx, JSVAL_TO_OBJECT(argv[1]))) && (((VrmlField *)rotation)->fieldType() == VrmlField::SFROTATION) && JSVAL_IS_OBJECT(argv[2]) && &SFVec3fClass == JS_GetClass(JSVAL_TO_OBJECT(argv[2])) && (scale = (VrmlSFVec3f *)JS_GetPrivate(cx, JSVAL_TO_OBJECT(argv[2]))) && (((VrmlField *)scale)->fieldType() == VrmlField::SFVEC3F) && JSVAL_IS_OBJECT(argv[3]) && &SFRotationClass == JS_GetClass(JSVAL_TO_OBJECT(argv[3])) && (scaleOrientation = (VrmlSFRotation *)JS_GetPrivate(cx, JSVAL_TO_OBJECT(argv[3]))) && (((VrmlField *)scaleOrientation)->fieldType() == VrmlField::SFROTATION) && JSVAL_IS_OBJECT(argv[4]) && &SFVec3fClass == JS_GetClass(JSVAL_TO_OBJECT(argv[4])) && (center = (VrmlSFVec3f *)JS_GetPrivate(cx, JSVAL_TO_OBJECT(argv[4]))) && (((VrmlField *)center)->fieldType() == VrmlField::SFVEC3F))
    {
        double m2[16];
        Mtrans(m, translation->get());
        Mtrans(m2, center->get());
        MM(m, m2); // XXX: richtig rum?
        if (!FPZERO(rotation->get()[3]))
        {
            Mrotation(m2, rotation->get());
            MM(m, m2);
        }
        if (!FPEQUAL(scale->get()[0], 1.0) || !FPEQUAL(scale->get()[1], 1.0) || !FPEQUAL(scale->get()[2], 1.0))
        {
            if (!FPZERO(scaleOrientation->get()[3]))
            {
                Mrotation(m2, scaleOrientation->get());
                MM(m, m2);
                Mscale(m, scale->get());
                MM(m, m2);
                Mrotation(m2, scaleOrientation->get(), -scaleOrientation->get()[3]);
                MM(m, m2);
            }
            else
            {
                Mscale(m, scale->get());
                MM(m, m2);
            }
        }
        Mtrans(m2, -center->get()[0], -center->get()[1], -center->get()[2]);
        MM(m, m2);
        cerr << "VrmlMatrix_setTransform successfull" << endl;
        return JS_TRUE;
    }

    return JS_FALSE;
}

// Create class objects for the non-scalar field types.
// Move toString to a prototype object?

void ScriptJS::defineAPI()
{
    // SFColor
    static JSPropertySpec SFColorProps[] = {
        { "r", 0, JSPROP_ENUMERATE, 0, 0 },
        { "g", 1, JSPROP_ENUMERATE, 0, 0 },
        { "b", 2, JSPROP_ENUMERATE, 0, 0 },
        { 0, 0, 0, 0, 0 }
    };

    static JSFunctionSpec SFColorMethods[] = {
        /*    name          native          nargs    */
        { "setHSV", color_setHSV, 3, 0, 0 },
        { "getHSV", color_getHSV, 0, 0, 0 },
        { "toString", toString, 0, 0, 0 },
        { 0, 0, 0, 0, 0 }
    };

    if (!JS_InitClass(d_cx, d_globalObj, 0, &SFColorClass,
                      SFColorCons, 0, // constructor function, min arg count
                      SFColorProps, SFColorMethods,
                      0, 0)) // static properties and methods
        System::the->error("JS_InitClass SFColor failed\n");

    static JSPropertySpec SFColorRGBAProps[] = {
        { "r", 0, JSPROP_ENUMERATE, 0, 0 },
        { "g", 1, JSPROP_ENUMERATE, 0, 0 },
        { "b", 2, JSPROP_ENUMERATE, 0, 0 },
        { "a", 3, JSPROP_ENUMERATE, 0, 0 },
        { 0, 0, 0, 0, 0 }
    };

    static JSFunctionSpec SFColorRGBAMethods[] = {
        /*    name          native          nargs    */
        { "setHSV", colorRGBA_setHSV, 3, 0, 0 },
        { "getHSV", colorRGBA_getHSV, 0, 0, 0 },
        { "toString", toString, 0, 0, 0 },
        { 0, 0, 0, 0, 0 }
    };

    if (!JS_InitClass(d_cx, d_globalObj, 0, &SFColorRGBAClass,
                      SFColorRGBACons, 0, // constructor function, min arg count
                      SFColorRGBAProps, SFColorRGBAMethods,
                      0, 0)) // static properties and methods
        System::the->error("JS_InitClass SFColorRGBA failed\n");

    /* // SFFloat
    static JSPropertySpec SFFloatProps[] = {
      {"v",       0,       JSPROP_ENUMERATE},
      {0}
    };

    static JSFunctionSpec SFFloatMethods[] = {
      //    name          native          nargs
      {"toString",        toString,       0},
      {0}
      };

   if (! JS_InitClass( d_cx, d_globalObj, 0, &SFFloatClass,
   SFFloatCons, 0, // constructor function, min arg count
   SFFloatProps, SFFloatMethods,
   0, 0))	  // static properties and methods
   System::the->error("JS_InitClass SFColor failed\n");
   */

    // SFImage
    static JSPropertySpec SFImageProps[] = {
        { "x", 0, JSPROP_ENUMERATE, 0, 0 },
        { "y", 1, JSPROP_ENUMERATE, 0, 0 },
        { "comp", 2, JSPROP_ENUMERATE, 0, 0 },
        { "array", 3, JSPROP_ENUMERATE, 0, 0 },
        { 0, 0, 0, 0, 0 }
    };

    static JSFunctionSpec SFImageMethods[] = {
        /*    name          native          nargs    */
        { "toString", toString, 0, 0, 0 },
        { 0, 0, 0, 0, 0 }
    };

    if (!JS_InitClass(d_cx, d_globalObj, 0, &SFImageClass,
                      SFImageCons, 0, // constructor function, min arg count
                      SFImageProps, SFImageMethods,
                      0, 0)) // static properties and methods
        System::the->error("JS_InitClass SFImage failed\n");

    // SFNode
    static JSFunctionSpec SFNodeMethods[] = {
        /*    name          native          nargs    */
        { "toString", toString, 0, 0, 0 },
        { 0, 0, 0, 0, 0 }
    };

    if (!JS_InitClass(d_cx, d_globalObj, 0, &SFNodeClass,
                      SFNodeCons, 1, // constructor function, min arg count
                      0, SFNodeMethods,
                      0, 0)) // static properties and methods
        System::the->error("JS_InitClass SFNode failed\n");

    // SFRotation
    static JSPropertySpec SFRotationProps[] = {
        { "x", 0, JSPROP_ENUMERATE, 0, 0 },
        { "y", 1, JSPROP_ENUMERATE, 0, 0 },
        { "z", 2, JSPROP_ENUMERATE, 0, 0 },
        { "angle", 3, JSPROP_ENUMERATE, 0, 0 },
        { 0, 0, 0, 0, 0 }
    };
    static JSFunctionSpec SFRotationMethods[] = {
        /*    name          native          nargs    */
        { "getAxis", rot_getAxis, 0, 0, 0 },
        { "inverse", rot_inverse, 0, 0, 0 },
        { "multiply", rot_multiply, 1, 0, 0 },
        { "multVec", rot_multVec, 1, 0, 0 },
        { "setAxis", rot_setAxis, 1, 0, 0 },
        { "slerp", rot_slerp, 2, 0, 0 },
        { "toString", toString, 0, 0, 0 },
        { 0, 0, 0, 0, 0 }
    };

    if (!JS_InitClass(d_cx, d_globalObj, 0, &SFRotationClass,
                      SFRotationCons, 0, // constructor function, min arg count
                      SFRotationProps, SFRotationMethods,
                      0, 0)) // static properties and methods
        System::the->error("JS_InitClass SFRotation failed\n");

    static JSFunctionSpec SFVec2dMethods[] = {
        /*    name          native          nargs    */
        { "add", vec2d_add, 1, 0, 0 },
        { "divide", vec2d_divide, 1, 0, 0 },
        { "dot", vec2d_dot, 1, 0, 0 },
        { "length", vec2d_length, 0, 0, 0 },
        { "multiply", vec2d_multiply, 1, 0, 0 },
        { "negate", vec2d_negate, 0, 0, 0 },
        { "normalize", vec2d_normalize, 0, 0, 0 },
        { "subtract", vec2d_subtract, 1, 0, 0 },
        { "toString", toString, 0, 0, 0 },
        { 0, 0, 0, 0, 0 }
    };

    // SFVec2d
    static JSPropertySpec SFVec2dProps[] = {
        { "x", 0, JSPROP_ENUMERATE, 0, 0 },
        { "y", 1, JSPROP_ENUMERATE, 0, 0 },
        { 0, 0, 0, 0, 0 }
    };

    if (!JS_InitClass(d_cx, d_globalObj, 0, &SFVec2dClass,
                      SFVec2dCons, 0, // constructor function, min arg count
                      SFVec2dProps, SFVec2dMethods,
                      0, 0)) // static properties and methods
        System::the->error("JS_InitClass SFVec2d failed\n");

    // SFVec2f
    static JSPropertySpec SFVec2fProps[] = {
        { "x", 0, JSPROP_ENUMERATE, 0, 0 },
        { "y", 1, JSPROP_ENUMERATE, 0, 0 },
        { 0, 0, 0, 0, 0 }
    };

    static JSFunctionSpec SFVec2fMethods[] = {
        /*    name          native          nargs    */
        { "add", vec2f_add, 1, 0, 0 },
        { "divide", vec2f_divide, 1, 0, 0 },
        { "dot", vec2f_dot, 1, 0, 0 },
        { "length", vec2f_length, 0, 0, 0 },
        { "multiply", vec2f_multiply, 1, 0, 0 },
        { "negate", vec2f_negate, 0, 0, 0 },
        { "normalize", vec2f_normalize, 0, 0, 0 },
        { "subtract", vec2f_subtract, 1, 0, 0 },
        { "toString", toString, 0, 0, 0 },
        { 0, 0, 0, 0, 0 }
    };

    if (!JS_InitClass(d_cx, d_globalObj, 0, &SFVec2fClass,
                      SFVec2fCons, 0, // constructor function, min arg count
                      SFVec2fProps, SFVec2fMethods,
                      0, 0)) // static properties and methods
        System::the->error("JS_InitClass SFVec2f failed\n");

    // SFVec3d
    static JSPropertySpec SFVec3dProps[] = {
        { "x", 0, JSPROP_ENUMERATE, 0, 0 },
        { "y", 1, JSPROP_ENUMERATE, 0, 0 },
        { "z", 2, JSPROP_ENUMERATE, 0, 0 },
        { 0, 0, 0, 0, 0 }
    };

    static JSFunctionSpec SFVec3dMethods[] = {
        /*    name          native          nargs    */
        { "add", vec3d_add, 1, 0, 0 },
        { "cross", vec3d_cross, 1, 0, 0 },
        { "divide", vec3d_divide, 1, 0, 0 },
        { "dot", vec3d_dot, 1, 0, 0 },
        { "length", vec3d_length, 0, 0, 0 },
        { "multiply", vec3d_multiply, 1, 0, 0 },
        { "negate", vec3d_negate, 0, 0, 0 },
        { "normalize", vec3d_normalize, 0, 0, 0 },
        { "subtract", vec3d_subtract, 1, 0, 0 },
        { "toString", toString, 0, 0, 0 },
        { 0, 0, 0, 0, 0 }
    };

    if (!JS_InitClass(d_cx, d_globalObj, 0, &SFVec3dClass,
                      SFVec3dCons, 0, // constructor function, min arg count
                      SFVec3dProps, SFVec3dMethods,
                      0, 0)) // static properties and methods
        System::the->error("JS_InitClass SFVec3d failed\n");

    // SFVec3f
    static JSPropertySpec SFVec3fProps[] = {
        { "x", 0, JSPROP_ENUMERATE, 0, 0 },
        { "y", 1, JSPROP_ENUMERATE, 0, 0 },
        { "z", 2, JSPROP_ENUMERATE, 0, 0 },
        { 0, 0, 0, 0, 0 }
    };
    static JSFunctionSpec SFVec3fMethods[] = {
        /*    name          native          nargs    */
        { "add", vec3f_add, 1, 0, 0 },
        { "cross", vec3f_cross, 1, 0, 0 },
        { "divide", vec3f_divide, 1, 0, 0 },
        { "dot", vec3f_dot, 1, 0, 0 },
        { "length", vec3f_length, 0, 0, 0 },
        { "multiply", vec3f_multiply, 1, 0, 0 },
        { "negate", vec3f_negate, 0, 0, 0 },
        { "normalize", vec3f_normalize, 0, 0, 0 },
        { "subtract", vec3f_subtract, 1, 0, 0 },
        { "toString", toString, 0, 0, 0 },
        { 0, 0, 0, 0, 0 }
    };

    if (!JS_InitClass(d_cx, d_globalObj, 0, &SFVec3fClass,
                      SFVec3fCons, 0, // constructor function, min arg count
                      SFVec3fProps, SFVec3fMethods,
                      0, 0)) // static properties and methods
        System::the->error("JS_InitClass SFVec3f failed\n");

    // VrmlMatrix class
    static JSPropertySpec VrmlMatrixProps[] = {
        { "mat", 0, JSPROP_ENUMERATE, 0, 0 },
        { 0, 0, 0, 0, 0 }
    };
    static JSFunctionSpec VrmlMFFloatMethods[] = {
        /*    name          native                        nargs    */
        { "toString", toString, 0, 0, 0 },
        { "length", mffloatlength, 0, 0, 0 },
        { 0, 0, 0, 0, 0 }
    };
    static JSFunctionSpec VrmlMFInt32Methods[] = {
        /*    name          native                        nargs    */
        { "toString", toString, 0, 0, 0 },
        { "length", mfint32length, 0, 0, 0 },
        { 0, 0, 0, 0, 0 }
    };
    static JSFunctionSpec VrmlMFTimeMethods[] = {
        /*    name          native                        nargs    */
        { "toString", toString, 0, 0, 0 },
        { "length", mffloatlength, 0, 0, 0 },
        { 0, 0, 0, 0, 0 }
    };

    static JSFunctionSpec VrmlMatrixMethods[] = {
        /*    name          native                        nargs    */
        { "setTransform", VrmlMatrix_setTransform, 5, 0, 0 },
        { "multMatrixVec", VrmlMatrix_multMatrixVec, 1, 0, 0 },
        { 0, 0, 0, 0, 0 }
    };

    if (!JS_InitClass(d_cx, d_globalObj, 0, &VrmlMatrixClass,
                      VrmlMatrixCons, 0, // constructor function, min arg count
                      VrmlMatrixProps, VrmlMatrixMethods,
                      0, 0)) // static properties and methods
        System::the->error("JS_InitClass VrmlMatrix failed\n");

    // All the MF* classes are subclassed from Array
    JSObject *array_proto = js_InitArrayClass(d_cx, d_globalObj);

    // MFColor class
    if (!JS_InitClass(d_cx, d_globalObj, array_proto,
                      &MFColorClass,
                      MFColorCons, 0, // constructor function, min arg count
                      0, 0,
                      0, 0)) // static properties and methods
        System::the->error("JS_InitClass MFColor failed\n");

    // MFColorRGBA class
    if (!JS_InitClass(d_cx, d_globalObj, array_proto,
                      &MFColorRGBAClass,
                      MFColorRGBACons, 0, // constructor function, min arg count
                      0, 0,
                      0, 0)) // static properties and methods
        System::the->error("JS_InitClass MFColorRGBA failed\n");

    // MFFloat class
    if (!JS_InitClass(d_cx, d_globalObj, array_proto,
                      &MFFloatClass,
                      MFFloatCons, 0, // constructor function, min arg count
                      0, VrmlMFFloatMethods,
                      0, 0)) // static properties and methods
        System::the->error("JS_InitClass MFFloat failed\n");

    // MFInt32 class
    if (!JS_InitClass(d_cx, d_globalObj, array_proto,
                      &MFInt32Class,
                      MFInt32Cons, 0, // constructor function, min arg count
                      0, VrmlMFInt32Methods,
                      0, 0)) // static properties and methods
        System::the->error("JS_InitClass MFInt32 failed\n");

    // MFNode class
    if (!JS_InitClass(d_cx, d_globalObj, array_proto,
                      &MFNodeClass,
                      MFNodeCons, 0, // constructor function, min arg count
                      0, 0,
                      0, 0)) // static properties and methods
        System::the->error("JS_InitClass MFNode failed\n");

    // MFRotation class
    if (!JS_InitClass(d_cx, d_globalObj, array_proto,
                      &MFRotationClass,
                      MFRotationCons, 0, // constructor function, min arg count
                      0, 0,
                      0, 0)) // static properties and methods
        System::the->error("JS_InitClass MFRotation failed\n");

    // MFString class
    if (!JS_InitClass(d_cx, d_globalObj, array_proto,
                      &MFStringClass,
                      MFStringCons, 0, // constructor function, min arg count
                      0, 0,
                      0, 0)) // static properties and methods
        System::the->error("JS_InitClass MFString failed\n");

    // MFTime class
    if (!JS_InitClass(d_cx, d_globalObj, array_proto,
                      &MFTimeClass,
                      MFTimeCons, 0, // constructor function, min arg count
                      0, VrmlMFTimeMethods,
                      0, 0)) // static properties and methods
        System::the->error("JS_InitClass MFTime failed\n");

    // MFVec2d class
    if (!JS_InitClass(d_cx, d_globalObj, array_proto,
                      &MFVec2dClass,
                      MFVec2dCons, 0, // constructor function, min arg count
                      0, 0,
                      0, 0)) // static properties and methods
        System::the->error("JS_InitClass MFVec2d failed\n");

    // MFVec2f class
    if (!JS_InitClass(d_cx, d_globalObj, array_proto,
                      &MFVec2fClass,
                      MFVec2fCons, 0, // constructor function, min arg count
                      0, 0,
                      0, 0)) // static properties and methods
        System::the->error("JS_InitClass MFVec2f failed\n");

    // MFVec3d class
    if (!JS_InitClass(d_cx, d_globalObj, array_proto,
                      &MFVec3dClass,
                      MFVec3dCons, 0, // constructor function, min arg count
                      0, 0,
                      0, 0)) // static properties and methods
        System::the->error("JS_InitClass MFVec3d failed\n");

    // MFVec3f class
    if (!JS_InitClass(d_cx, d_globalObj, array_proto,
                      &MFVec3fClass,
                      MFVec3fCons, 0, // constructor function, min arg count
                      0, 0,
                      0, 0)) // static properties and methods
        System::the->error("JS_InitClass MFVec3f failed\n");
}

// Convert a VrmlField value to a jsval, optionally protect from gc.

jsval ScriptJS::vrmlFieldToJSVal(VrmlField::VrmlFieldType type,
                                 const VrmlField *f,
                                 bool protect)
{
    switch (type)
    {
    case VrmlField::SFBOOL:
    {
        return BOOLEAN_TO_JSVAL(f ? f->toSFBool()->get() : 0);
    }

    case VrmlField::SFCOLOR:
    {
        JSObject *obj = JS_NewObject(d_cx, &SFColorClass, 0, d_globalObj);
        if (!obj)
            return JSVAL_NULL;
        if (protect)
            JS_AddRoot(d_cx, obj);
        JS_SetPrivate(d_cx, obj, f ? f->clone() : new VrmlSFColor());
        return OBJECT_TO_JSVAL(obj);
    }

    case VrmlField::SFCOLORRGBA:
    {
        JSObject *obj = JS_NewObject(d_cx, &SFColorRGBAClass, 0, d_globalObj);
        if (!obj)
            return JSVAL_NULL;
        if (protect)
            JS_AddRoot(d_cx, obj);
        JS_SetPrivate(d_cx, obj, f ? f->clone() : new VrmlSFColorRGBA());
        return OBJECT_TO_JSVAL(obj);
    }

    case VrmlField::SFDOUBLE:
    {
        jsdouble *d = JS_NewDouble(d_cx, f ? f->toSFDouble()->get() : 0.0);
        if (protect)
            JS_AddRoot(d_cx, d);
        return DOUBLE_TO_JSVAL(d);
    }

    case VrmlField::SFFLOAT:
    {
        jsdouble *d = JS_NewDouble(d_cx, f ? f->toSFFloat()->get() : 0.0);
        if (protect)
            JS_AddRoot(d_cx, d);
        return DOUBLE_TO_JSVAL(d);
    }

    case VrmlField::SFIMAGE:
    {
        JSObject *obj = JS_NewObject(d_cx, &SFImageClass, 0, d_globalObj);
        if (!obj)
            return JSVAL_NULL;
        if (protect)
            JS_AddRoot(d_cx, obj);
        JS_SetPrivate(d_cx, obj, f ? f->clone() : new VrmlSFImage());
        return OBJECT_TO_JSVAL(obj);
    }

    case VrmlField::SFINT32:
    {
        return INT_TO_JSVAL(f ? f->toSFInt()->get() : 0);
    }

    case VrmlField::SFNODE:
    {
        JSObject *obj = JS_NewObject(d_cx, &SFNodeClass, 0, d_globalObj);
        if (!obj)
            return JSVAL_NULL;
        if (protect)
            JS_AddRoot(d_cx, obj);
        JS_SetPrivate(d_cx, obj, f ? f->clone() : new VrmlSFNode());
        return OBJECT_TO_JSVAL(obj);
    }

    case VrmlField::SFROTATION:
    {
        JSObject *obj = JS_NewObject(d_cx, &SFRotationClass, 0, d_globalObj);
        if (!obj)
            return JSVAL_NULL;
        if (protect)
            JS_AddRoot(d_cx, obj);
        JS_SetPrivate(d_cx, obj, f ? f->clone() : new VrmlSFRotation());
        return OBJECT_TO_JSVAL(obj);
    }

    case VrmlField::SFSTRING:
    {
        JSString *s = JS_NewStringCopyZ(d_cx, f ? f->toSFString()->get() : "");
        if (protect)
            JS_AddRoot(d_cx, s);
        return STRING_TO_JSVAL(s);
    }

    case VrmlField::SFTIME:
    {
        jsdouble *d = JS_NewDouble(d_cx, f ? f->toSFTime()->get() : 0);
        if (protect)
            JS_AddRoot(d_cx, d);
        return DOUBLE_TO_JSVAL(d);
    }

    case VrmlField::SFVEC2D:
    {
        JSObject *obj = JS_NewObject(d_cx, &SFVec2dClass, 0, d_globalObj);
        if (!obj)
            return JSVAL_NULL;
        if (protect)
            JS_AddRoot(d_cx, obj);
        JS_SetPrivate(d_cx, obj, f ? f->clone() : new VrmlSFVec2d());
        return OBJECT_TO_JSVAL(obj);
    }

    case VrmlField::SFVEC3D:
    {
        JSObject *obj = JS_NewObject(d_cx, &SFVec3dClass, 0, d_globalObj);
        if (!obj)
            return JSVAL_NULL;
        if (protect)
            JS_AddRoot(d_cx, obj);
        JS_SetPrivate(d_cx, obj, f ? f->clone() : new VrmlSFVec3d());
        return OBJECT_TO_JSVAL(obj);
    }

    case VrmlField::SFVEC2F:
    {
        JSObject *obj = JS_NewObject(d_cx, &SFVec2fClass, 0, d_globalObj);
        if (!obj)
            return JSVAL_NULL;
        if (protect)
            JS_AddRoot(d_cx, obj);
        JS_SetPrivate(d_cx, obj, f ? f->clone() : new VrmlSFVec2f());
        return OBJECT_TO_JSVAL(obj);
    }

    case VrmlField::SFVEC3F:
    {
        JSObject *obj = JS_NewObject(d_cx, &SFVec3fClass, 0, d_globalObj);
        if (!obj)
            return JSVAL_NULL;
        if (protect)
            JS_AddRoot(d_cx, obj);
        JS_SetPrivate(d_cx, obj, f ? f->clone() : new VrmlSFVec3f());
        return OBJECT_TO_JSVAL(obj);
    }

    // MF*
    case VrmlField::MFCOLOR:
    {
        VrmlMFColor *mf = f ? ((VrmlMFColor *)(f->toMFColor())) : 0;
        int i, n = mf ? mf->size() : 0;
        JSObject *obj = JS_NewArrayObject(d_cx, (jsint)n, 0);
        if (!obj)
            return JSVAL_NULL;
        JS_AddRoot(d_cx, obj);
        float *fn = mf ? mf->get() : 0;
        for (i = 0; i < n; ++i, fn += 3)
        {
            JSObject *elt = JS_NewObject(d_cx, &SFColorClass,
                                         0, d_globalObj);
            if (!elt)
                break;
            JS_SetPrivate(d_cx, elt, new VrmlSFColor(fn[0], fn[1], fn[2]));
            JS_DefineElement(d_cx, obj, (jsint)i, OBJECT_TO_JSVAL(elt),
                             JS_PropertyStub, JS_PropertyStub,
                             JSPROP_ENUMERATE);
        }
        if (!protect)
            JS_RemoveRoot(d_cx, obj);
        return OBJECT_TO_JSVAL(obj);
    }

    case VrmlField::MFCOLORRGBA:
    {
        VrmlMFColorRGBA *mf = f ? ((VrmlMFColorRGBA *)(f->toMFColorRGBA())) : 0;
        int i, n = mf ? mf->size() : 0;
        JSObject *obj = JS_NewArrayObject(d_cx, (jsint)n, 0);
        if (!obj)
            return JSVAL_NULL;
        JS_AddRoot(d_cx, obj);
        float *fn = mf ? mf->get() : 0;
        for (i = 0; i < n; ++i, fn += 4)
        {
            JSObject *elt = JS_NewObject(d_cx, &SFColorRGBAClass,
                                         0, d_globalObj);
            if (!elt)
                break;
            JS_SetPrivate(d_cx, elt, new VrmlSFColorRGBA(fn[0], fn[1], fn[2], fn[3]));
            JS_DefineElement(d_cx, obj, (jsint)i, OBJECT_TO_JSVAL(elt),
                             JS_PropertyStub, JS_PropertyStub,
                             JSPROP_ENUMERATE);
        }
        if (!protect)
            JS_RemoveRoot(d_cx, obj);
        return OBJECT_TO_JSVAL(obj);
    }

    case VrmlField::MFDOUBLE:
    {

        JSObject *obj = JS_NewObject(d_cx, &MFDoubleClass, 0, d_globalObj);
        if (!obj)
            return JSVAL_NULL;
        if (protect)
            JS_AddRoot(d_cx, obj);
        JS_SetPrivate(d_cx, obj, f ? f->clone() : new VrmlMFDouble());
        return OBJECT_TO_JSVAL(obj);

        /*VrmlMFDouble *mf = f ? ((VrmlMFDouble*) (f->toMFDouble())) : 0;
         int i, n = mf ? mf->size() : 0;
         JSObject *obj = JS_NewArrayObject( d_cx, (jsint)n, 0 );
         if (! obj) return JSVAL_NULL;
         JS_AddRoot( d_cx, obj );
         double *fn = mf ? mf->get() : 0;
         for (i=0; i<n; ++i, ++fn)
           {
             jsdouble *elt = JS_NewDouble( d_cx, *fn );
             if (! elt) break;
             JS_DefineElement( d_cx, obj, (jsint)i, DOUBLE_TO_JSVAL(elt),
         JS_PropertyStub, MF_setProperty,
         JSPROP_ENUMERATE );
         }
         if (! protect) JS_RemoveRoot( d_cx, obj );
         return OBJECT_TO_JSVAL(obj);*/
    }

    case VrmlField::MFFLOAT:
    {

        JSObject *obj = JS_NewObject(d_cx, &MFFloatClass, 0, d_globalObj);
        if (!obj)
            return JSVAL_NULL;
        if (protect)
            JS_AddRoot(d_cx, obj);
        JS_SetPrivate(d_cx, obj, f ? f->clone() : new VrmlMFFloat());
        return OBJECT_TO_JSVAL(obj);

        /*VrmlMFFloat *mf = f ? ((VrmlMFFloat*) (f->toMFFloat())) : 0;
         int i, n = mf ? mf->size() : 0;
         JSObject *obj = JS_NewArrayObject( d_cx, (jsint)n, 0 );
         if (! obj) return JSVAL_NULL;
         JS_AddRoot( d_cx, obj );
         float *fn = mf ? mf->get() : 0;
         for (i=0; i<n; ++i, ++fn)
           {
             jsdouble *elt = JS_NewDouble( d_cx, *fn );
             if (! elt) break;
             JS_DefineElement( d_cx, obj, (jsint)i, DOUBLE_TO_JSVAL(elt),
         JS_PropertyStub, MF_setProperty,
         JSPROP_ENUMERATE );
         }
         if (! protect) JS_RemoveRoot( d_cx, obj );
         return OBJECT_TO_JSVAL(obj);*/
    }

    case VrmlField::MFINT32:
    {
        JSObject *obj = JS_NewObject(d_cx, &MFInt32Class, 0, d_globalObj);
        if (!obj)
            return JSVAL_NULL;
        if (protect)
            JS_AddRoot(d_cx, obj);
        JS_SetPrivate(d_cx, obj, f ? f->clone() : new VrmlMFInt());
        return OBJECT_TO_JSVAL(obj);
        /*
         VrmlMFInt *mf = f ? ((VrmlMFInt*) (f->toMFInt())) : 0;
         int i, n = mf ? mf->size() : 0;
         JSObject *obj = JS_NewArrayObject( d_cx, (jsint)n, 0 );
         if (! obj) return JSVAL_NULL;
         if (protect) JS_AddRoot( d_cx, obj );
         int *fn = mf ? mf->get() : 0;
         for (i=0; i<n; ++i, ++fn)
          {
            JS_DefineElement( d_cx, obj, (jsint)i, INT_TO_JSVAL(*fn),
                    JS_PropertyStub, JS_PropertyStub,
         JSPROP_ENUMERATE );
         }
         return OBJECT_TO_JSVAL(obj);*/
    }

    case VrmlField::MFNODE:
    {
        VrmlMFNode *mf = f ? ((VrmlMFNode *)(f->toMFNode())) : 0;
        int i, n = mf ? mf->size() : 0;
        JSObject *obj = JS_NewArrayObject(d_cx, (jsint)n, 0);
        if (!obj)
            return JSVAL_NULL;
        JS_AddRoot(d_cx, obj);
        for (i = 0; i < n; ++i)
        {
            JSObject *elt = JS_NewObject(d_cx, &SFNodeClass,
                                         0, d_globalObj);
            if (!elt)
                break;
            JS_SetPrivate(d_cx, elt, new VrmlSFNode(mf->get(i)));
            JS_DefineElement(d_cx, obj, (jsint)i, OBJECT_TO_JSVAL(elt),
                             JS_PropertyStub, JS_PropertyStub,
                             JSPROP_ENUMERATE);
        }
        if (!protect)
            JS_RemoveRoot(d_cx, obj);
        return OBJECT_TO_JSVAL(obj);
    }

    case VrmlField::MFROTATION:
    {
        VrmlMFRotation *mf = f ? ((VrmlMFRotation *)(f->toMFRotation())) : 0;
        int i, n = mf ? mf->size() : 0;
        JSObject *obj = JS_NewArrayObject(d_cx, (jsint)n, 0);
        if (!obj)
            return JSVAL_NULL;
        JS_AddRoot(d_cx, obj);
        float *fn = mf ? mf->get() : 0;
        for (i = 0; i < n; ++i, fn += 4)
        {
            JSObject *elt = JS_NewObject(d_cx, &SFRotationClass,
                                         0, d_globalObj);
            if (!elt)
                break;
            JS_SetPrivate(d_cx, elt,
                          new VrmlSFRotation(fn[0], fn[1], fn[2], fn[3]));
            JS_DefineElement(d_cx, obj, (jsint)i, OBJECT_TO_JSVAL(elt),
                             JS_PropertyStub, JS_PropertyStub,
                             JSPROP_ENUMERATE);
        }
        if (!protect)
            JS_RemoveRoot(d_cx, obj);
        return OBJECT_TO_JSVAL(obj);
    }

    case VrmlField::MFSTRING:
    {
        jsval *jsvec = 0;
        int i, n = 0;
        if (f)
        {
            VrmlMFString *mf = (VrmlMFString *)(f->toMFString());
            n = mf ? mf->size() : 0;
            if (n > 0)
                jsvec = new jsval[n];
            for (i = 0; i < n; ++i)
            {
                JSString *s = JS_NewStringCopyZ(d_cx, mf->get(i));
                jsvec[i] = STRING_TO_JSVAL(s);
            }
        }
        JSObject *obj = JS_NewArrayObject(d_cx, (jsint)n, jsvec);
        if (jsvec)
            delete[] jsvec;
        if (!obj)
            return JSVAL_NULL;
        if (protect)
            JS_AddRoot(d_cx, obj);
        return OBJECT_TO_JSVAL(obj);
    }

    case VrmlField::MFTIME:
    {

        JSObject *obj = JS_NewObject(d_cx, &MFTimeClass, 0, d_globalObj);
        if (!obj)
            return JSVAL_NULL;
        if (protect)
            JS_AddRoot(d_cx, obj);
        JS_SetPrivate(d_cx, obj, f ? f->clone() : new VrmlMFTime());
        return OBJECT_TO_JSVAL(obj);

        /*VrmlMFTime *mf = f ? ((VrmlMFTime*) (f->toMFTime())) : 0;
         int i, n = mf ? mf->size() : 0;
         JSObject *obj = JS_NewArrayObject( d_cx, (jsint)n, 0 );
         if (! obj) return JSVAL_NULL;
         JS_AddRoot( d_cx, obj );
         double *fn = mf ? mf->get() : 0;
         for (i=0; i<n; ++i, ++fn)
           {
             jsdouble *elt = JS_NewDouble( d_cx, *fn );
             if (! elt) break;
             JS_DefineElement( d_cx, obj, (jsint)i, DOUBLE_TO_JSVAL(elt),
         JS_PropertyStub, MF_setProperty,
         JSPROP_ENUMERATE );
         }
         if (! protect) JS_RemoveRoot( d_cx, obj );
         return OBJECT_TO_JSVAL(obj);*/
    }

    case VrmlField::MFVEC2D:
    {
        VrmlMFVec2d *mf = f ? ((VrmlMFVec2d *)(f->toMFVec2d())) : 0;
        int i, n = mf ? mf->size() : 0;
        JSObject *obj = JS_NewArrayObject(d_cx, (jsint)n, 0);
        if (!obj)
            return JSVAL_NULL;
        JS_AddRoot(d_cx, obj);
        double *fn = mf ? mf->get() : 0;
        for (i = 0; i < n; ++i, fn += 2)
        {
            JSObject *elt = JS_NewObject(d_cx, &SFVec2dClass,
                                         0, d_globalObj);
            if (!elt)
                break;
            JS_SetPrivate(d_cx, elt, new VrmlSFVec2d(fn[0], fn[1]));
            JS_DefineElement(d_cx, obj, (jsint)i, OBJECT_TO_JSVAL(elt),
                             JS_PropertyStub, JS_PropertyStub,
                             JSPROP_ENUMERATE);
        }
        if (!protect)
            JS_RemoveRoot(d_cx, obj);
        return OBJECT_TO_JSVAL(obj);
    }

    case VrmlField::MFVEC3D:
    {
        JSObject *obj = JS_NewObject(d_cx, &MFVec3dClass, 0, d_globalObj);
        if (!obj)
            return JSVAL_NULL;
        if (protect)
            JS_AddRoot(d_cx, obj);
        JS_SetPrivate(d_cx, obj, f ? f->clone() : new VrmlMFVec3d());
        return OBJECT_TO_JSVAL(obj);

        /*
         VrmlMFVec3d *mf = f ? ((VrmlMFVec3d*) (f->toMFVec3d())) : 0;
         int i, n = mf ? mf->size() : 0;
         JSObject *obj = JS_NewArrayObject( d_cx, (jsint)n, 0 );
         if (! obj) return JSVAL_NULL;
         JS_AddRoot( d_cx, obj );
         double *fn = mf ? mf->get() : 0;
         for (i=0; i<n; ++i, fn+=3)
          {
            JSObject *elt = JS_NewObject( d_cx, &SFVec3dClass,
                      0, d_globalObj );
         if (! elt) break;
         JS_SetPrivate( d_cx, elt, new VrmlSFVec3d(fn[0], fn[1], fn[2]) );
         JS_DefineElement( d_cx, obj, (jsint)i, OBJECT_TO_JSVAL(elt),
         JS_PropertyStub, JS_PropertyStub,
         JSPROP_ENUMERATE );
         }
         if (! protect) JS_RemoveRoot( d_cx, obj );
         return OBJECT_TO_JSVAL(obj);*/
    }

    case VrmlField::MFVEC2F:
    {
        VrmlMFVec2f *mf = f ? ((VrmlMFVec2f *)(f->toMFVec2f())) : 0;
        int i, n = mf ? mf->size() : 0;
        JSObject *obj = JS_NewArrayObject(d_cx, (jsint)n, 0);
        if (!obj)
            return JSVAL_NULL;
        JS_AddRoot(d_cx, obj);
        float *fn = mf ? mf->get() : 0;
        for (i = 0; i < n; ++i, fn += 2)
        {
            JSObject *elt = JS_NewObject(d_cx, &SFVec2fClass,
                                         0, d_globalObj);
            if (!elt)
                break;
            JS_SetPrivate(d_cx, elt, new VrmlSFVec2f(fn[0], fn[1]));
            JS_DefineElement(d_cx, obj, (jsint)i, OBJECT_TO_JSVAL(elt),
                             JS_PropertyStub, JS_PropertyStub,
                             JSPROP_ENUMERATE);
        }
        if (!protect)
            JS_RemoveRoot(d_cx, obj);
        return OBJECT_TO_JSVAL(obj);
    }

    case VrmlField::MFVEC3F:
    {
        JSObject *obj = JS_NewObject(d_cx, &MFVec3fClass, 0, d_globalObj);
        if (!obj)
            return JSVAL_NULL;
        if (protect)
            JS_AddRoot(d_cx, obj);
        JS_SetPrivate(d_cx, obj, f ? f->clone() : new VrmlMFVec3f());
        return OBJECT_TO_JSVAL(obj);

        /*
         VrmlMFVec3f *mf = f ? ((VrmlMFVec3f*) (f->toMFVec3f())) : 0;
         int i, n = mf ? mf->size() : 0;
         JSObject *obj = JS_NewArrayObject( d_cx, (jsint)n, 0 );
         if (! obj) return JSVAL_NULL;
         JS_AddRoot( d_cx, obj );
         float *fn = mf ? mf->get() : 0;
         for (i=0; i<n; ++i, fn+=3)
          {
            JSObject *elt = JS_NewObject( d_cx, &SFVec3fClass,
                      0, d_globalObj );
         if (! elt) break;
         JS_SetPrivate( d_cx, elt, new VrmlSFVec3f(fn[0], fn[1], fn[2]) );
         JS_DefineElement( d_cx, obj, (jsint)i, OBJECT_TO_JSVAL(elt),
         JS_PropertyStub, JS_PropertyStub,
         JSPROP_ENUMERATE );
         }
         if (! protect) JS_RemoveRoot( d_cx, obj );
         return OBJECT_TO_JSVAL(obj);*/
    }

    default:
        System::the->error("vrmlFieldToJSVal: unhandled type (%d)\n",
                           (int)type);
        return BOOLEAN_TO_JSVAL(false); // oops...
    }
}

// Convert a jsval to a (new) VrmlField

static VrmlField *jsvalToVrmlField(JSContext *cx,
                                   jsval v,
                                   VrmlField::VrmlFieldType expectType)
{
    if (JSVAL_IS_NULL(v) || JSVAL_IS_VOID(v))
        return 0;

    else if (JSVAL_IS_BOOLEAN(v) && expectType == VrmlField::SFBOOL)
        return new VrmlSFBool(JSVAL_TO_BOOLEAN(v) != 0);

    else if (JSVAL_IS_INT(v) && expectType == VrmlField::SFINT32)
        return new VrmlSFInt(JSVAL_TO_INT(v));
    else if (JSVAL_IS_NUMBER(v) && expectType == VrmlField::SFINT32)
    {
        jsdouble d;
        JS_ValueToNumber(cx, v, &d);
        return new VrmlSFInt((int)d);
    }

    else if (JSVAL_IS_NUMBER(v) && expectType == VrmlField::SFFLOAT)
    {
        jsdouble d;
        JS_ValueToNumber(cx, v, &d);
        return new VrmlSFFloat((float)d);
    }

    else if (JSVAL_IS_NUMBER(v) && expectType == VrmlField::SFTIME)
    {
        jsdouble d;
        JS_ValueToNumber(cx, v, &d);
        return new VrmlSFTime(d);
    }

    else if (JSVAL_IS_STRING(v) && expectType == VrmlField::SFSTRING)
    {
        JSString *str = JSVAL_TO_STRING(v);
        return new VrmlSFString(str ? JS_GetStringBytes(str) : "");
    }

    else if (JSVAL_IS_OBJECT(v) && ((expectType == VrmlField::SFCOLOR && &SFColorClass == JS_GetClass(JSVAL_TO_OBJECT(v))) || (expectType == VrmlField::SFCOLORRGBA && &SFColorRGBAClass == JS_GetClass(JSVAL_TO_OBJECT(v))) || (expectType == VrmlField::SFIMAGE && &SFImageClass == JS_GetClass(JSVAL_TO_OBJECT(v))) || (expectType == VrmlField::SFNODE && &SFNodeClass == JS_GetClass(JSVAL_TO_OBJECT(v))) || (expectType == VrmlField::SFROTATION && &SFRotationClass == JS_GetClass(JSVAL_TO_OBJECT(v))) || (expectType == VrmlField::SFVEC2D && &SFVec2dClass == JS_GetClass(JSVAL_TO_OBJECT(v))) || (expectType == VrmlField::SFVEC3D && &SFVec3dClass == JS_GetClass(JSVAL_TO_OBJECT(v))) || (expectType == VrmlField::SFVEC2F && &SFVec2fClass == JS_GetClass(JSVAL_TO_OBJECT(v))) || (expectType == VrmlField::SFVEC3F && &SFVec3fClass == JS_GetClass(JSVAL_TO_OBJECT(v)))))
    {
        VrmlField *f = (VrmlField *)JS_GetPrivate(cx, JSVAL_TO_OBJECT(v));
        if (!f)
            return 0;
        return f->clone();
    }

    // MFColor
    else if (JSVAL_IS_OBJECT(v) && JS_IsArrayObject(cx, JSVAL_TO_OBJECT(v)) && expectType == VrmlField::MFCOLOR)
    {
        jsuint len;
        JS_GetArrayLength(cx, JSVAL_TO_OBJECT(v), &len);
        float *data = new float[3 * len];
        memset(data, 0, 3 * len * sizeof(float));

        for (int i = 0; i < (int)len; ++i)
        {
            jsval elt;
            JS_GetElement(cx, JSVAL_TO_OBJECT(v), (jsint)i, &elt);
            if (JSVAL_IS_OBJECT(elt))
            {
                JSObject *obj = JSVAL_TO_OBJECT(elt);
                VrmlField *vf = obj ? (VrmlField *)JS_GetPrivate(cx, obj) : 0;
                VrmlSFColor *c = vf ? vf->toSFColor() : 0;
                if (c)
                {
                    data[3 * i] = c->get()[0];
                    data[3 * i + 1] = c->get()[1];
                    data[3 * i + 2] = c->get()[2];
                }
            }
        }
        VrmlMFColor *f = new VrmlMFColor((int)len, data);
        delete[] data;
        return f;
    }

    // MFColorRGBA
    else if (JSVAL_IS_OBJECT(v) && JS_IsArrayObject(cx, JSVAL_TO_OBJECT(v)) && expectType == VrmlField::MFCOLORRGBA)
    {
        jsuint len;
        JS_GetArrayLength(cx, JSVAL_TO_OBJECT(v), &len);
        float *data = new float[4 * len];
        memset(data, 0, 4 * len * sizeof(float));

        for (int i = 0; i < (int)len; ++i)
        {
            jsval elt;
            JS_GetElement(cx, JSVAL_TO_OBJECT(v), (jsint)i, &elt);
            if (JSVAL_IS_OBJECT(elt))
            {
                JSObject *obj = JSVAL_TO_OBJECT(elt);
                VrmlField *vf = obj ? (VrmlField *)JS_GetPrivate(cx, obj) : 0;
                VrmlSFColorRGBA *c = vf ? vf->toSFColorRGBA() : 0;
                if (c)
                {
                    data[4 * i] = c->get()[0];
                    data[4 * i + 1] = c->get()[1];
                    data[4 * i + 2] = c->get()[2];
                    data[4 * i + 3] = c->get()[3];
                }
            }
        }
        VrmlMFColorRGBA *f = new VrmlMFColorRGBA((int)len, data);
        delete[] data;
        return f;
    }

    // MFDouble
    else if (JSVAL_IS_OBJECT(v) && JS_IsArrayObject(cx, JSVAL_TO_OBJECT(v)) && expectType == VrmlField::MFDOUBLE)
    {
        jsuint len;
        JS_GetArrayLength(cx, JSVAL_TO_OBJECT(v), &len);
        double *data = new double[len];
        memset(data, 0, len * sizeof(double));

        for (int i = 0; i < (int)len; ++i)
        {
            jsval elt;
            JS_GetElement(cx, JSVAL_TO_OBJECT(v), (jsint)i, &elt);
            if (JSVAL_IS_NUMBER(elt))
            {
                jsdouble factor;
                JS_ValueToNumber(cx, elt, &factor);
                data[i] = factor;
            }
        }
        VrmlMFDouble *f = new VrmlMFDouble((int)len, data);
        delete[] data;
        return f;
    }

    // MFFloat
    else if (JSVAL_IS_OBJECT(v) && JS_IsArrayObject(cx, JSVAL_TO_OBJECT(v)) && expectType == VrmlField::MFFLOAT)
    {
        jsuint len;
        JS_GetArrayLength(cx, JSVAL_TO_OBJECT(v), &len);
        float *data = new float[len];
        memset(data, 0, len * sizeof(float));

        for (int i = 0; i < (int)len; ++i)
        {
            jsval elt;
            JS_GetElement(cx, JSVAL_TO_OBJECT(v), (jsint)i, &elt);
            if (JSVAL_IS_NUMBER(elt))
            {
                jsdouble factor;
                JS_ValueToNumber(cx, elt, &factor);
                data[i] = (float)factor;
            }
        }
        VrmlMFFloat *f = new VrmlMFFloat((int)len, data);
        delete[] data;
        return f;
    }

    // MFInt
    else if (JSVAL_IS_OBJECT(v) && JS_IsArrayObject(cx, JSVAL_TO_OBJECT(v)) && expectType == VrmlField::MFINT32)
    {
        jsuint len;
        JS_GetArrayLength(cx, JSVAL_TO_OBJECT(v), &len);
        int *data = new int[len];
        memset(data, 0, len * sizeof(int));

        for (int i = 0; i < (int)len; ++i)
        {
            jsval elt;
            JS_GetElement(cx, JSVAL_TO_OBJECT(v), (jsint)i, &elt);
            if (JSVAL_IS_INT(elt))
            {
                int factor;
                factor = JSVAL_TO_INT(elt);
                data[i] = (int)factor;
            }
        }
        VrmlMFInt *f = new VrmlMFInt((int)len, data);
        delete[] data;
        return f;
    }

    // MFNode
    else if (JSVAL_IS_OBJECT(v) && JS_IsArrayObject(cx, JSVAL_TO_OBJECT(v)) && expectType == VrmlField::MFNODE)
    {
        jsuint len;
        JS_GetArrayLength(cx, JSVAL_TO_OBJECT(v), &len);
        VrmlMFNode *f = new VrmlMFNode((int)len, 0);
        VrmlNode **nodes = f->get();
        for (int i = 0; i < (int)len; ++i)
        {
            jsval elt;
            JS_GetElement(cx, JSVAL_TO_OBJECT(v), (jsint)i, &elt);
            VrmlSFNode *child = 0;
            if (JSVAL_IS_OBJECT(elt) && &SFNodeClass == JS_GetClass(JSVAL_TO_OBJECT(elt)))
                child = (VrmlSFNode *)JS_GetPrivate(cx, JSVAL_TO_OBJECT(elt));
            nodes[i] = child ? child->get()->reference() : 0;
        }
        return f;
    }

    // MFRotation
    else if (JSVAL_IS_OBJECT(v) && JS_IsArrayObject(cx, JSVAL_TO_OBJECT(v)) && expectType == VrmlField::MFROTATION)
    {
        jsuint len;
        JS_GetArrayLength(cx, JSVAL_TO_OBJECT(v), &len);
        float *data = new float[4 * len];
        memset(data, 0, 4 * len * sizeof(float));

        for (int i = 0; i < (int)len; ++i)
        {
            jsval elt;
            JS_GetElement(cx, JSVAL_TO_OBJECT(v), (jsint)i, &elt);
            if (JSVAL_IS_OBJECT(elt))
            {
                JSObject *obj = JSVAL_TO_OBJECT(elt);
                VrmlField *vf = obj ? (VrmlField *)JS_GetPrivate(cx, obj) : 0;
                VrmlSFRotation *c = vf ? vf->toSFRotation() : 0;
                if (c)
                {
                    data[4 * i] = c->get()[0];
                    data[4 * i + 1] = c->get()[1];
                    data[4 * i + 2] = c->get()[2];
                    data[4 * i + 3] = c->get()[3];
                }
            }
        }
        VrmlMFRotation *f = new VrmlMFRotation((int)len, data);
        delete[] data;
        return f;
    }

    // MFString
    else if (JSVAL_IS_OBJECT(v) && JS_IsArrayObject(cx, JSVAL_TO_OBJECT(v)) && expectType == VrmlField::MFSTRING)
    {
        jsuint len;
        JS_GetArrayLength(cx, JSVAL_TO_OBJECT(v), &len);
        const char **s = new const char *[len];
        for (int i = 0; i < (int)len; ++i)
        {
            jsval elt;
            JS_GetElement(cx, JSVAL_TO_OBJECT(v), (jsint)i, &elt);
            JSString *str = 0;
            if (JSVAL_IS_STRING(elt) && (str = JSVAL_TO_STRING(elt)) != 0)
                s[i] = JS_GetStringBytes(str);
            else
                s[i] = 0;
        }
        VrmlMFString *f = new VrmlMFString((int)len, s);
        delete[] s;
        return f;
    }

    // MFTime
    else if (JSVAL_IS_OBJECT(v) && JS_IsArrayObject(cx, JSVAL_TO_OBJECT(v)) && expectType == VrmlField::MFTIME)
    {
        jsuint len;
        JS_GetArrayLength(cx, JSVAL_TO_OBJECT(v), &len);
        double *data = new double[len];
        memset(data, 0, len * sizeof(double));

        for (int i = 0; i < (int)len; ++i)
        {
            jsval elt;
            JS_GetElement(cx, JSVAL_TO_OBJECT(v), (jsint)i, &elt);
            if (JSVAL_IS_NUMBER(elt))
            {
                jsdouble factor;
                JS_ValueToNumber(cx, elt, &factor);
                data[i] = factor;
            }
        }
        VrmlMFTime *f = new VrmlMFTime((int)len, data);
        delete[] data;
        return f;
    }

    // MFVec2d
    else if (JSVAL_IS_OBJECT(v) && JS_IsArrayObject(cx, JSVAL_TO_OBJECT(v)) && expectType == VrmlField::MFVEC2D)
    {
        jsuint len;
        JS_GetArrayLength(cx, JSVAL_TO_OBJECT(v), &len);
        double *data = new double[2 * len];
        memset(data, 0, 2 * len * sizeof(double));

        for (int i = 0; i < (int)len; ++i)
        {
            jsval elt;
            JS_GetElement(cx, JSVAL_TO_OBJECT(v), (jsint)i, &elt);
            if (JSVAL_IS_OBJECT(elt))
            {
                JSObject *obj = JSVAL_TO_OBJECT(elt);
                VrmlField *vf = obj ? (VrmlField *)JS_GetPrivate(cx, obj) : 0;
                VrmlSFVec2d *c = vf ? vf->toSFVec2d() : 0;
                if (c)
                {
                    data[2 * i] = c->get()[0];
                    data[2 * i + 1] = c->get()[1];
                }
            }
        }
        VrmlMFVec2d *f = new VrmlMFVec2d((int)len, data);
        delete[] data;
        return f;
    }

    // MFVec3d
    else if (JSVAL_IS_OBJECT(v) && JS_GetPrivate(cx, JSVAL_TO_OBJECT(v)) && expectType == VrmlField::MFVEC3D)
    {

        VrmlMFVec3d *mf = (VrmlMFVec3d *)JS_GetPrivate(cx, JSVAL_TO_OBJECT(v));
        VrmlMFVec3d *f = new VrmlMFVec3d(*mf);
        return f;
    }

    // MFVec2f
    else if (JSVAL_IS_OBJECT(v) && JS_IsArrayObject(cx, JSVAL_TO_OBJECT(v)) && expectType == VrmlField::MFVEC2F)
    {
        jsuint len;
        JS_GetArrayLength(cx, JSVAL_TO_OBJECT(v), &len);
        float *data = new float[2 * len];
        memset(data, 0, 2 * len * sizeof(float));

        for (int i = 0; i < (int)len; ++i)
        {
            jsval elt;
            JS_GetElement(cx, JSVAL_TO_OBJECT(v), (jsint)i, &elt);
            if (JSVAL_IS_OBJECT(elt))
            {
                JSObject *obj = JSVAL_TO_OBJECT(elt);
                VrmlField *vf = obj ? (VrmlField *)JS_GetPrivate(cx, obj) : 0;
                VrmlSFVec2f *c = vf ? vf->toSFVec2f() : 0;
                if (c)
                {
                    data[2 * i] = c->get()[0];
                    data[2 * i + 1] = c->get()[1];
                }
            }
        }
        VrmlMFVec2f *f = new VrmlMFVec2f((int)len, data);
        delete[] data;
        return f;
    }

    // MFVec3f
    else if (JSVAL_IS_OBJECT(v) && JS_GetPrivate(cx, JSVAL_TO_OBJECT(v)) && expectType == VrmlField::MFVEC3F)
    {

        VrmlMFVec3f *mf = (VrmlMFVec3f *)JS_GetPrivate(cx, JSVAL_TO_OBJECT(v));
        VrmlMFVec3f *f = new VrmlMFVec3f(*mf);
        return f;
    }

    else
    {
        System::the->error("jsvalToVrmlField: unhandled type (expected %d)\n",
                           expectType);
        JSString *str = JS_ValueToString(cx, v);
        if (str)
            System::the->error("%s", JS_GetStringBytes(str));

        return 0;
    }
}

static JSBool
exposedField_setProperty(JSContext* cx, JSObject* obj, jsval id, jsval* val)
{
    JSString* str = JS_ValueToString(cx, id);
    if (!str)
        return JS_FALSE;
    const char* eventName = JS_GetStringBytes(str);

    // The ScriptJS object pointer is stored as a global property value:
    ScriptJS* script = objToScript(cx, obj);
    if (!script)
        return JS_FALSE;
    VrmlNodeScript* scriptNode = script->scriptNode();

    // Validate the type
    VrmlField::VrmlFieldType t = scriptNode->hasExposedField(eventName);
    if (!t)
        return JS_FALSE;

    // Convert to a vrmlField and set the eventOut value
    VrmlField* f = jsvalToVrmlField(cx, *val, t);
    if (!f)
    {
        System::the->error("Error: invalid type in assignment to eventOut %s\n",
            eventName);
        return JS_FALSE;
    }

    scriptNode->setExposedField(eventName, f);
    if (f)
        delete f;

    /*
    no idea why this should be done.
    val is the value that is assigned to an event out
    _eventOut Properties should only be added to event our fields not to the values that are assigned to an event out...
    it causes fields to be destroyed when assigned to an event out.
    // Don't overwrite the property value.
    if (JSVAL_IS_OBJECT(*val) && JSVAL_TO_OBJECT(*val) != 0 && !JS_DefineProperty(cx, JSVAL_TO_OBJECT(*val), "_eventOut", PRIVATE_TO_JSVAL((intptr_t)eventName), 0, 0, JSPROP_READONLY | JSPROP_PERMANENT))
        System::the->error("JS_DefineProp _eventOut failed\n");
        */

    return JS_TRUE;
}

// Must assign the proper type to eventOuts

static JSBool
eventOut_setProperty(JSContext *cx, JSObject *obj, jsval id, jsval *val)
{
    JSString *str = JS_ValueToString(cx, id);
    if (!str)
        return JS_FALSE;
    const char *eventName = JS_GetStringBytes(str);

    // The ScriptJS object pointer is stored as a global property value:
    ScriptJS *script = objToScript(cx, obj);
    if (!script)
        return JS_FALSE;
    VrmlNodeScript *scriptNode = script->scriptNode();

    // Validate the type
    VrmlField::VrmlFieldType t = scriptNode->hasEventOut(eventName);
    if (!t)
        return JS_FALSE;

    // Convert to a vrmlField and set the eventOut value
    VrmlField *f = jsvalToVrmlField(cx, *val, t);
    if (!f)
    {
        System::the->error("Error: invalid type in assignment to eventOut %s\n",
                           eventName);
        return JS_FALSE;
    }

    scriptNode->setEventOut(eventName, f);
    if (f)
        delete f;

    /*
    no idea why this should be done.
    val is the value that is assigned to an event out
    _eventOut Properties should only be added to event our fields not to the values that are assigned to an event out...
    it causes fields to be destroyed when assigned to an event out.
    // Don't overwrite the property value.
    if (JSVAL_IS_OBJECT(*val) && JSVAL_TO_OBJECT(*val) != 0 && !JS_DefineProperty(cx, JSVAL_TO_OBJECT(*val), "_eventOut", PRIVATE_TO_JSVAL((intptr_t)eventName), 0, 0, JSPROP_READONLY | JSPROP_PERMANENT))
        System::the->error("JS_DefineProp _eventOut failed\n");
        */

    return JS_TRUE;
}

// Define objects corresponding to fields/eventOuts

void ScriptJS::defineFields()
{
    VrmlNodeScript::FieldList::iterator i, end = d_node->fields().end();
    for (i = d_node->fields().begin(); i != end; ++i)
    {
        jsval val = vrmlFieldToJSVal((*i)->type, (*i)->value, false);
#ifdef DEBUG
        if ((*i)->value)
            cout << "field " << (*i)->name << " value "
                 << *((*i)->value) << endl;
        else
            cout << "field " << (*i)->name << " <no value>\n";
#endif
        if ((*i)->exposed)
        {
            if (!JS_DefineProperty(d_cx, d_globalObj, (*i)->name, val,
                //getter, setter, ...
                0, exposedField_setProperty,
                JSPROP_PERMANENT))
                System::the->error("JS_DefineProp %s failed\n", (*i)->name);
        }
        else
        {
        if (!JS_DefineProperty(d_cx, d_globalObj, (*i)->name, val,
                               //getter, setter, ...
                               0, 0,
                               JSPROP_PERMANENT))
            System::the->error("JS_DefineProp %s failed\n", (*i)->name);
        }
    }

    end = d_node->eventOuts().end();
    for (i = d_node->eventOuts().begin(); i != end; ++i)
    {
        jsval val = vrmlFieldToJSVal((*i)->type, (*i)->value, false);
        if (JSVAL_IS_OBJECT(val) && JSVAL_TO_OBJECT(val) != 0 && !JS_DefineProperty(d_cx, JSVAL_TO_OBJECT(val), "_eventOut", PRIVATE_TO_JSVAL((intptr_t)((*i)->name)), 0, 0, JSPROP_READONLY | JSPROP_PERMANENT))
            System::the->error("JS_DefineProp _eventOut failed\n");

        if (!JS_DefineProperty(d_cx, d_globalObj, (*i)->name, val,
                               //getter, setter
                               0, eventOut_setProperty,
                               JSPROP_PERMANENT))
            System::the->error("JS_DefineProp %s failed\n", (*i)->name);
    }
}

// Global functions
static JSBool
Print(JSContext *cx, JSObject *, uintN argc, jsval *argv, jsval *)
{
    uintN i;

    for (i = 0; i < argc; i++)
    {
        JSString *str = JS_ValueToString(cx, argv[i]);
        if (!str)
            return JS_FALSE;
        printf("%s%s", i ? " " : "", JS_GetStringBytes(str));
    }
    putchar('\n');
    return JS_TRUE;
}

// Global functions

static JSBool
PrintNl(JSContext *cx, JSObject *, uintN argc, jsval *argv, jsval *)
{
    uintN i;

    for (i = 0; i < argc; i++)
    {
        JSString *str = JS_ValueToString(cx, argv[i]);
        if (!str)
            return JS_FALSE;
        printf("%s%s", i ? " " : "", JS_GetStringBytes(str));
    }
    return JS_TRUE;
}

// Browser functions

static JSBool getName(JSContext *cx, JSObject *b,
                      uintN, jsval *,
                      jsval *rval)
{
    jsval p;
    if (!JS_GetProperty(cx, b, "_script", &p))
        return JS_FALSE;

    ScriptJS *s = (ScriptJS *)JSVAL_TO_PRIVATE(p);
    if (!s || !s->browser())
        return JS_FALSE;

    const char *name = s->browser()->getName();
    *rval = STRING_TO_JSVAL(JS_InternString(cx, name));
    return JS_TRUE;
}

static JSBool getVersion(JSContext *cx, JSObject *b,
                         uintN, jsval *,
                         jsval *rval)
{
    jsval p;
    if (!JS_GetProperty(cx, b, "_script", &p))
        return JS_FALSE;

    ScriptJS *s = (ScriptJS *)JSVAL_TO_PRIVATE(p);
    if (!s || !s->browser())
        return JS_FALSE;

    const char *version = s->browser()->getVersion();
    *rval = STRING_TO_JSVAL(JS_InternString(cx, version));
    return JS_TRUE;
}

static JSBool getCurrentSpeed(JSContext *cx, JSObject *,
                              uintN, jsval *,
                              jsval *rval)
{
    //...
    *rval = DOUBLE_TO_JSVAL(JS_NewDouble(cx, 0.0));
    return JS_TRUE;
}

static JSBool getCurrentFrameRate(JSContext *cx, JSObject *b,
                                  uintN, jsval *, jsval *rval)
{
    jsval p;
    if (!JS_GetProperty(cx, b, "_script", &p))
        return JS_FALSE;

    ScriptJS *s = (ScriptJS *)JSVAL_TO_PRIVATE(p);
    if (!s || !s->browser())
        return JS_FALSE;

    *rval = DOUBLE_TO_JSVAL(JS_NewDouble(cx, s->browser()->getFrameRate()));
    return JS_TRUE;
}

static JSBool getWorldURL(JSContext *cx, JSObject *b,
                          uintN, jsval *,
                          jsval *rval)
{
    jsval p;
    if (!JS_GetProperty(cx, b, "_script", &p))
        return JS_FALSE;

    ScriptJS *s = (ScriptJS *)JSVAL_TO_PRIVATE(p);
    if (!s || !s->browser())
        return JS_FALSE;

    *rval = STRING_TO_JSVAL(JS_InternString(cx, s->browser()->urlDoc() ? s->browser()->urlDoc()->url().c_str() : ""));
    return JS_TRUE;
}

// No events will be processed after loadURL.

static JSBool loadURL(JSContext *cx, JSObject *b,
                      uintN argc, jsval *argv, jsval *)
{
    if ((int)argc < 1)
        return JS_FALSE;

    jsval p;
    if (!JS_GetProperty(cx, b, "_script", &p))
        return JS_FALSE;

    ScriptJS *s = (ScriptJS *)JSVAL_TO_PRIVATE(p);
    if (!s || !s->browser())
        return JS_FALSE;

    // Get 2 arguments MFString url, MFString parameters
    VrmlMFString *url, *parameters = 0;
    url = (VrmlMFString *)jsvalToVrmlField(cx, argv[0], VrmlField::MFSTRING);
    if (!url)
        return JS_TRUE; // empty URL

    if ((int)argc > 1)
        parameters = (VrmlMFString *)jsvalToVrmlField(cx, argv[1],
                                                      VrmlField::MFSTRING);

    s->browser()->queueLoadUrl(url, parameters);
    return JS_TRUE;
}

// This does return, but no events will be processed after it is called.

static JSBool replaceWorld(JSContext *cx, JSObject *b,
                           uintN argc, jsval *argv, jsval *)
{
    if ((int)argc < 1)
        return JS_FALSE;

    jsval p;
    if (!JS_GetProperty(cx, b, "_script", &p))
        return JS_FALSE;

    ScriptJS *s = (ScriptJS *)JSVAL_TO_PRIVATE(p);
    if (!s || !s->browser())
        return JS_FALSE;

    // Get 1 argument MFNode nodes
    VrmlMFNode *nodes = (VrmlMFNode *)jsvalToVrmlField(cx,
                                                       argv[0],
                                                       VrmlField::MFNODE);
    if (nodes)
    {
        VrmlNamespace *ns = new VrmlNamespace(); // should be stored with nodes...
        s->browser()->queueReplaceNodes(nodes, ns);
    }

    return JS_TRUE;
}

static JSBool createVrmlFromString(JSContext *cx, JSObject *bobj,
                                   uintN argc, jsval *argv, jsval *rval)
{
    JSString *str;

    Doc *relative = NULL;
    jsval p;
    ScriptJS *s = NULL; // egcs warning
    if (JS_GetProperty(cx, bobj, "_script", &p) && (s = (ScriptJS *)JSVAL_TO_PRIVATE(p)) != 0 && s->browser() != 0)
        relative = s->browser()->urlDoc();

    if (argc == 1 && JSVAL_IS_STRING(argv[0]) && (str = JSVAL_TO_STRING(argv[0])) != 0)
    {
        char *vrmlString = JS_GetStringBytes(str);
        VrmlNamespace *newScope = new VrmlNamespace();
        VrmlMFNode *kids;

        if (!(kids = VrmlScene::readString(vrmlString, newScope, relative)))
            return JS_FALSE;

        // Put the children from g into an MFNode and return in rval.
        // should store the namespace as well...
        int i, n = kids->size();
        jsval *jsvec = new jsval[n];
        VrmlNode **k = kids->get();

        for (i = 0; i < n; ++i)
        {
            JSObject *obj = JS_NewObject(cx, &SFNodeClass, 0, bobj);
            if (!obj)
                return JS_FALSE;
            JS_SetPrivate(cx, obj, new VrmlSFNode(k[i] ? k[i]->reference() : 0));
            jsvec[i] = OBJECT_TO_JSVAL(obj);
        }
        *rval = OBJECT_TO_JSVAL(JS_NewArrayObject(cx, (jsint)n, jsvec));
        delete[] jsvec;
        delete kids;
        return JS_TRUE;
    }

    return JS_FALSE;
}

// createVrmlFromURL( MFString url, SFNode node, SFString event )

static JSBool createVrmlFromURL(JSContext *cx, JSObject *b,
                                uintN argc, jsval *argv, jsval *)
{
    if ((int)argc != 3)
        return JS_FALSE;

    Doc *relative = 0;

    jsval p;
    ScriptJS *s = 0; // egcs warning
    if (JS_GetProperty(cx, b, "_script", &p) && (s = (ScriptJS *)JSVAL_TO_PRIVATE(p)) != 0 && s->browser() != 0)
        relative = s->browser()->urlDoc();

    VrmlMFString *url = 0;
    VrmlField *f;
    if (JSVAL_IS_STRING(argv[0]))
    {
        JSString *str = JSVAL_TO_STRING(argv[0]);
        url = new VrmlMFString(JS_GetStringBytes(str));
    }
    else if ((f = jsvalToVrmlField(cx, argv[0], VrmlField::MFSTRING)) != 0)
    {
        url = f->toMFString();
    }

    if (url)
    {
        VrmlSFNode *node = 0; // egcs warning
        JSString *str = 0; // egcs warning

        f = jsvalToVrmlField(cx, argv[1], VrmlField::SFNODE);
        if (f != 0 && (node = f->toSFNode()) != 0 && JSVAL_IS_STRING(argv[2]) && (str = JSVAL_TO_STRING(argv[2])) != 0)
        {
            char *event = JS_GetStringBytes(str);

            VrmlNamespace ns; // this is a problem...
            VrmlMFNode *kids = VrmlScene::readWrl(url, relative, &ns);
            VrmlNode *nn = node->get();

            if (nn && kids)
                nn->eventIn(s_timeStamp, // fix me...
                            event,
                            kids);
            delete node;
            delete kids;
            delete url;
            return JS_TRUE;
        }

        delete url;
        delete f;
    }
    return JS_FALSE;
}

// addRoute(SFNode fromNode, String fromEventOut, SFNode toNode, String toEvent)

static JSBool addRoute(JSContext *cx, JSObject *,
                       uintN argc, jsval *argv,
                       jsval *)
{
    if ((int)argc != 4 || !JSVAL_IS_STRING(argv[1]) || !JSVAL_IS_STRING(argv[3]))
        return JS_FALSE;
    JSString *str = JS_ValueToString(cx, argv[1]);
    if (!str)
        return JS_FALSE;
    char *eventOut = JS_GetStringBytes(str);
    str = JS_ValueToString(cx, argv[3]);
    if (!str)
        return JS_FALSE;
    char *eventIn = JS_GetStringBytes(str);

    if (!eventOut || !eventIn)
        return JS_FALSE;
    VrmlField *f = jsvalToVrmlField(cx, argv[0], VrmlField::SFNODE);
    VrmlSFNode *from = f ? f->toSFNode() : 0;
    f = jsvalToVrmlField(cx, argv[2], VrmlField::SFNODE);
    VrmlSFNode *to = f ? f->toSFNode() : 0;
    if (!to || !to->get() || !from || !from->get())
        return JS_FALSE;

    System::the->debug("addRoute(%s::%s::%s, %s::%s::%s)\n",
                       from->get()->nodeType()->getName(),
                       from->get()->name(),
                       eventOut,
                       to->get()->nodeType()->getName(),
                       to->get()->name(),
                       eventIn);

    (from->get())->addRoute(eventOut, to->get(), eventIn);

    return JS_TRUE;
}

// deleteRoute(SFNode fromNode, String fromEventOut, SFNode toNode, String toEvent)

static JSBool deleteRoute(JSContext *cx, JSObject *,
                          uintN argc, jsval *argv, jsval *)
{
    if ((int)argc != 4 || !JSVAL_IS_STRING(argv[1]) || !JSVAL_IS_STRING(argv[3]))
        return JS_FALSE;
    JSString *str = JS_ValueToString(cx, argv[1]);
    if (!str)
        return JS_FALSE;
    char *eventOut = JS_GetStringBytes(str);
    str = JS_ValueToString(cx, argv[3]);
    if (!str)
        return JS_FALSE;
    char *eventIn = JS_GetStringBytes(str);

    if (!eventOut || !eventIn)
        return JS_FALSE;
    VrmlField *f = jsvalToVrmlField(cx, argv[0], VrmlField::SFNODE);
    VrmlSFNode *from = f ? f->toSFNode() : 0;
    f = jsvalToVrmlField(cx, argv[2], VrmlField::SFNODE);
    VrmlSFNode *to = f ? f->toSFNode() : 0;
    if (!to || !to->get() || !from || !from->get())
        return JS_FALSE;

    (from->get())->deleteRoute(eventOut, to->get(), eventIn);

    return JS_TRUE;
}

static JSBool setDescription(JSContext *cx, JSObject *,
                             uintN argc, jsval *argv, jsval *)
{
    if (argc < 1)
        return JS_FALSE;
    JSString *str = JS_ValueToString(cx, argv[0]);
    if (!str)
        return JS_FALSE;
    System::the->inform("%s", JS_GetStringBytes(str));
    return JS_TRUE;
}

static JSBool browserSystem(JSContext *cx, JSObject *,
                            uintN argc, jsval *argv, jsval *)
{
    if (argc < 1)
        return JS_FALSE;
    JSString *str = JS_ValueToString(cx, argv[0]);
    if (!str)
        return JS_FALSE;
    int retval;
    retval = system(JS_GetStringBytes(str));
    if (retval == -1)
    {
        std::cerr << "browserSystem: system failed" << std::endl;
        return JS_FALSE;
    }
    return JS_TRUE;
}

static JSBool getConfigEntry(JSContext *cx, JSObject *,
                             uintN argc, jsval *argv, jsval *rval)
{
    //jsdouble val=0.0;
    if (argc < 1)
        return JS_FALSE;
    JSString *str = JS_ValueToString(cx, argv[0]);
    if (!str)
        return JS_FALSE;
    std::string entry = System::the->getConfigEntry(JS_GetStringBytes(str));
    *rval = STRING_TO_JSVAL(JS_InternString(cx, entry.c_str()));
    return JS_TRUE;
}

static JSBool setFunctionState(JSContext *cx, JSObject *,
                               uintN argc, jsval *argv, jsval *)
{
    jsdouble val = 0.0;
    if (argc < 2)
        return JS_FALSE;
    JSString *str = JS_ValueToString(cx, argv[0]);
    if (!str)
        return JS_FALSE;
    JS_ValueToNumber(cx, argv[1], &val);
    System::the->warn("NOT IMPLEMENTED: setFunctionState:%s %d", JS_GetStringBytes(str), (int)val);
#if 0
    System::the->setBuiltInFunctionState(JS_GetStringBytes(str), (int)val);
    System::the->callBuiltInFunctionCallback(JS_GetStringBytes(str));
#endif
    return JS_TRUE;
}

static JSBool setSyncMode(JSContext *cx, JSObject *,
                          uintN argc, jsval *argv, jsval *)
{
    if (argc < 1)
        return JS_FALSE;
    JSString *str = JS_ValueToString(cx, argv[0]);
    if (!str)
        return JS_FALSE;
    System::the->setSyncMode(JS_GetStringBytes(str));
    return JS_TRUE;
}

static JSBool setFunctionValue(JSContext *cx, JSObject *,
                               uintN argc, jsval *argv, jsval *)
{
    jsdouble val = 0.0;
    if (argc < 2)
        return JS_FALSE;
    JSString *str = JS_ValueToString(cx, argv[0]);
    if (!str)
        return JS_FALSE;
    JS_ValueToNumber(cx, argv[1], &val);
    System::the->warn("NOT IMPLEMENTED: setFunctionValue:%s %f", JS_GetStringBytes(str), (float)val);
#if 0
    System::the->setBuiltInFunctionValue(JS_GetStringBytes(str), (float)val);
    System::the->callBuiltInFunctionCallback(JS_GetStringBytes(str));
#endif
    return JS_TRUE;
}

static JSBool requestMasterState(JSContext *, JSObject *,
                                 uintN, jsval *, jsval *rval)
{
    System::the->becomeMaster();
    *rval = BOOLEAN_TO_JSVAL(true);
    return JS_TRUE;
}

static JSBool isMaster(JSContext *, JSObject *,
                       uintN, jsval *, jsval *rval)
{
    *rval = BOOLEAN_TO_JSVAL(System::the->isMaster());
    return JS_TRUE;
}

//

static void
ErrorReporter(JSContext *, const char *message, JSErrorReport *report)
{
    int i, j, k, n;

    System::the->error("javascript: ");
    if (!report)
    {
        System::the->error("%s\n", message);
        return;
    }

    if (report->filename)
        System::the->error("%s, ", report->filename);
    if (report->lineno)
        System::the->error("line %u: ", report->lineno);
    System::the->error(message);
    if (!report->linebuf)
    {
        System::the->error("\n");
        return;
    }

    System::the->error(":\n%s\n", report->linebuf);
    n = (int)(report->tokenptr - report->linebuf);
    for (i = j = 0; i < n; i++)
    {
        if (report->linebuf[i] == '\t')
        {
            for (k = (j + 8) & ~7; j < k; j++)
                System::the->error(".");
            continue;
        }
        System::the->error(".");
        j++;
    }
    System::the->error("\n");
}
}
#endif // HAVE_JAVASCRIPT
