/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  ScriptJDK.cpp
//  java script objects.
//  http://java.sun.com/docs/books/tutorial/native1.1/invoking/invo.html
//

#include "config.h"
#if HAVE_JDK
#include "ScriptJDK.h"

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
#include "VrmlSFVec2f.h"
#include "VrmlSFVec3f.h"
#include "VrmlSFVec2d.h"
#include "VrmlSFVec3d.h"

#include "VrmlMFColor.h"
#include "VrmlMFColorRGBA.h"
#include "VrmlMFDouble.h"
#include "VrmlMFFloat.h"
#include "VrmlMFInt.h"
#include "VrmlMFNode.h"
#include "VrmlMFRotation.h"
#include "VrmlMFString.h"
#include "VrmlMFTime.h"
#include "VrmlMFVec2f.h"
#include "VrmlMFVec3f.h"
#include "VrmlMFVec2d.h"
#include "VrmlMFVec3d.h"

#ifdef _WIN32
#define PATH_SEPARATOR ";"
#else /* UNIX */
#define PATH_SEPARATOR ":"
#endif

// Static members
JavaVM *ScriptJDK::d_jvm = 0;
JNIEnv *ScriptJDK::d_env = 0;

using namespace vrml;

// Construct from classname. Assumes className.class exists in classDir

ScriptJDK::ScriptJDK(VrmlNodeScript *node,
                     const char *className,
                     const char *classDir)
    : d_node(node)
    , d_class(0)
    , d_object(0)
    , d_processEventsID(0)
    , d_eventsProcessedID(0)
{
    if (!d_jvm) // Initialize static members
    {
        JDK1_1InitArgs vm_args;
        jint res;

        /* IMPORTANT: specify vm_args version # if you use JDK1.1.2 and beyond */
        vm_args.version = 0x00010001;

        // Append classDir to the end of default system class path
        JNI_GetDefaultJavaVMInitArgs(&vm_args);

        if (classDir)
        {
            char fullpath[1024];
            strncpy(fullpath, vm_args.classpath, sizeof(fullpath) - 1);
            fullpath[sizeof(fullpath) - 1] = '\0';
            int sleft = sizeof(fullpath) - strlen(fullpath) - 1;
            strncat(fullpath, PATH_SEPARATOR, sleft);
            sleft -= strlen(PATH_SEPARATOR);
            strncat(fullpath, classDir, sleft);
            fullpath[sizeof(fullpath) - 1] = '\0';
            vm_args.classpath = fullpath;
        }

        /* Create the Java VM */
        res = JNI_CreateJavaVM(&d_jvm, &d_env, &vm_args);
        if (res < 0)
            System::the->error("Can't create Java VM");
    }

    if (d_jvm && d_env) // Per-object initialization
    {
        char fqClassName[1024];
        strcpy(fqClassName, "vrml/node/Script/"); // is this correct?...
        strcat(fqClassName, className);
        d_class = d_env->FindClass(fqClassName);
        if (d_class == 0)
        {
            System::the->error("Can't find Java class %s.\n", className);
            return;
        }

        // Create an object of this class
        //      d_object = ?

        // Call initialize()
        jmethodID initID = d_env->GetStaticMethodID(d_class, "initialize", "()V");

        if (initID != 0)
        {
            d_env->CallVoidMethod(d_object, initID);
        }

        // Cache other method IDs
        d_processEventsID = d_env->GetStaticMethodID(d_class, "processEvents",
                                                     "(I[Ljava/Cloneable/Event;)V");
        d_eventsProcessedID = d_env->GetStaticMethodID(d_class, "eventsProcessed",
                                                       "(I[Ljava/Cloneable/Event;)V");
    }
}

ScriptJDK::~ScriptJDK()
{
    if (d_jvm)
    {
        d_jvm->DestroyJavaVM();
        d_jvm = 0;
    }
}

// Run a specified script

void ScriptJDK::activate(double timeStamp,
                         const char *fname,
                         int argc,
                         const VrmlField *argv[])
{
    jstring jstr;
    jobjectArray args;

    if (!d_class)
        return;

    jstr = d_env->NewStringUTF(" from VRML!");
    if (jstr == 0)
    {
        System::the->error("Out of memory\n");
        return;
    }
    args = d_env->NewObjectArray(1, d_env->FindClass("java/lang/String"), jstr);
    if (args == 0)
    {
        System::the->error("Out of memory\n");
        return;
    }
    d_env->CallStaticVoidMethod(d_class, d_processEventsID, args);
}

// Get a handle to the scene from a ScriptJDK

VrmlScene *ScriptJDK::browser() { return d_node->browser(); }
#endif // HAVE_JDK
