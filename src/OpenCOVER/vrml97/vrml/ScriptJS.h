/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
#ifndef _SCRIPTJS_
#define _SCRIPTJS_

#include "config.h"
#if HAVE_JAVASCRIPT
//
//  Javascript Script class
//
#include "ScriptObject.h"
#include <vrml97/js/jsapi.h>

#include "VrmlField.h"

namespace vrml
{

class VrmlNodeScript;
class VrmlScene;

class VRMLEXPORT ScriptJS : public ScriptObject
{

public:
    ScriptJS(VrmlNodeScript *, const char *source);
    ~ScriptJS();

    virtual void activate(double timeStamp,
                          const char *fname,
                          int argc,
                          const VrmlField *argv[]);

    VrmlScene *browser();
    VrmlNodeScript *scriptNode()
    {
        return d_node;
    }

    jsval vrmlFieldToJSVal(VrmlField::VrmlFieldType,
                           const VrmlField *f, bool protect);

protected:
    static JSRuntime *rt;
    static int nInstances;

    void defineAPI();
    void defineFields();

    VrmlNodeScript *d_node;

    double d_timeStamp;

    JSContext *d_cx;
    JSObject *d_globalObj;
    JSObject *d_browserObj;
    bool maybeGC;
    bool forceGC;
};
}
#endif // HAVE_JAVASCRIPT
#endif // _SCRIPTJS_
