/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_COVISE_INTERACTOR_H
#define CO_COVISE_INTERACTOR_H

/*! \file
 \brief  feedback parameter changes to COVISE

 \author
 \author (C)
         Computer Centre University of Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date
 */

#include <util/DLinkList.h>
#include <util/coExport.h>

#include <cstring>
#include <cstdio>

#include <PluginUtil/coBaseCoviseInteractor.h>

namespace opencover
{
class RenderObject;
// feedback class
class coCoviseInteractor : public coBaseCoviseInteractor
{
public:
    coCoviseInteractor(const char *n, RenderObject *o, const char *attrib);
    ~coCoviseInteractor();

    // send a masseage to the module
    void sendFeedbackMessage(int len, const char *data);

private:
    void sendFeedback(const char *info, const char *key, const char *data = NULL);
};

class coInteractorList : public covise::DLinkList<coInteractor *>
{
public:
    coInteractorList();
    coInteractor *findSame(coInteractor *);
    void removeInteractor(coInteractor *);
    void removedObject(const char *objName);
};

extern coInteractorList interactorList;
}
#endif
