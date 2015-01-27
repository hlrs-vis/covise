/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS Attachable
//
// This class @@@
//
// Initial version: 2002-07-23 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#include "Attachable.h"
#include "ReadObj.h"
#include <covise/covise_do.h>
#include <assert.h>

inline char *CPY(const char *str)
{
    char *res = new char[1 + strlen(str)];
    strcpy(res, str);
    return res;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Static Variable initializers
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Constructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Attachable::Attachable(const char *filename)
{
    d_filename = CPY(filename);

    char *lastslash;
    lastslash = strrchr(d_filename, '/');
    if (!lastslash)
    {
        lastslash = d_filename;
        d_path = CPY("./");
    }
    else
    {
        char ch = *lastslash;
        *lastslash = '\0'; // temporarily terminate at / and copy
        d_path = CPY(d_filename);
        *lastslash = ch;
        lastslash++;
    }

    // Choice label is everything after last '/' without ending
    d_choiceLabel = CPY(lastslash);

    char *ending = strstr(d_choiceLabel, ".obj");
    if (!ending)
        ending = strstr(d_choiceLabel, ".OBJ");
    if (!ending)
        ending = strstr(d_choiceLabel, ".Obj");

    if (ending)
        *ending = '\0';
}

/// Copy-Constructor: copy everything
Attachable::Attachable(const Attachable &old)
{
    d_filename = CPY(old.d_filename);
    d_choiceLabel = CPY(old.d_choiceLabel);
    d_path = CPY(old.d_path);
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Destructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Attachable::~Attachable()
{
    delete[] d_filename;
    delete[] d_choiceLabel;
    delete[] d_path;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Operations
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Attribute request/set functions
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// return the name of the OBJ file
const char *Attachable::getObjFileName() const
{
    return d_filename;
}

const char *Attachable::getObjPath() const
{
    return d_path;
}

// return the choice label value
const char *Attachable::getChoiceLabel() const
{
    return d_choiceLabel;
}

/////// Create OBJ carrier object from a single OBJ file
coDistributedObject *Attachable::getObjDO(const char *objName, int index, const char *unit) const
{
    char fullObjName[1024];
    sprintf(fullObjName, "%s_%d", objName, index);

    return ReadObj::read(d_filename, fullObjName, unit);

    /*
   // create dummy point
   coDoPoints *points = new coDoPoints(fullObjName,0);
   //float *x,*y,*z;
   //points->getAddresses(&x,&y,&z);
   // *x = *y = *z = 0.0;

   // create all attributes in one step - save CRB communication
   const char *attrName[] = { "MODEL_PATH", "MODEL_FILE", "BACKFACE", "SCALE" };
   const char *attrVal[]  = { NULL,         NULL,         "OFF"     , NULL };

   if (0==strcmp("mm",unit))
   attrVal[3] = "1.0";
   else if (0==strcmp("cm",unit))
   attrVal[3] = "10.0";
   else if (0==strcmp("ft",unit))
   attrVal[3] = "25.4";
   else if (0==strcmp("m",unit))
   attrVal[3] = "1000.0";
   else
   attrVal[4] = "1.0";

   // split filename / directory
   char *modelPath = CPY(d_filename);
   char *filePart=strrchr(modelPath,'/');

   if(filePart)
   {
   *filePart='\0';
   filePart++;
   attrVal[0] = modelPath;
   }
   else
   {
   attrVal[0] = "./";
   filePart=modelPath;
   }
   attrVal[1] = filePart;

   points->addAttributes(4,attrName,attrVal);

   delete [] modelPath;

   return points;

   */
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Internally used functions
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++ Prevent auto-generated functions by assert or implement
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// Assignment operator: NOT IMPLEMENTED
Attachable &Attachable::operator=(const Attachable &)
{
    assert(0);
    return *this;
}

/// Default constructor: NOT IMPLEMENTED
Attachable::Attachable()
{
    assert(0);
}
