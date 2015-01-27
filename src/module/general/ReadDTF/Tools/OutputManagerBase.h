/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __TOOLS_OUTPUTMANAGER_BASE_H_
#define __TOOLS_OUTPUTMANAGER_BASE_H_

#include "baseobject.h"

using namespace std;

namespace Tools
{
class ClassManager;

ostream &endLine(ostream &ostr);
ostream &end(ostream &ostr);
ostream &endLineDebug(ostream &ostr);
ostream &endDebug(ostream &ostr);

extern ostream *out;

/*#define OUT \ 
      { Tools::out = Tools::ClassManager::getInstance()->getOutputMan()->getOut() ; \ 
      *Tools::out */

/** #define ENDL Tools::endLine ; \ 
      if ( Tools::out != NULL ) { delete Tools::out; Tools::out = NULL;} }*/

/*#define END \ 
      Tools::end ; \ 
      if ( Tools::out != NULL ) { delete Tools::out ; Tools::out = NULL;}}*/

/*#define DEBUGL \ 
      Tools::endLineDebug ; \ 
      if ( Tools::out != NULL ) { delete Tools::out ; Tools::out = NULL;}}*/

/*#define DEBUG \ 
      Tools::endDebug ; \ 
      if ( Tools::out != NULL ) { delete Tools::out ; Tools::out = NULL;}}*/

class OutputManagerBase : public BaseObject
{
public:
    OutputManagerBase();
    OutputManagerBase(string className, int objectID);
    virtual ~OutputManagerBase();

    virtual void setDebug(bool debugOnOff);
    virtual ostream *getOut();
    virtual void print(char *format, bool critical, ...);
};
};
#endif
