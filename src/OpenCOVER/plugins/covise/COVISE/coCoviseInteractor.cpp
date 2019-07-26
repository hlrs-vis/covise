/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <util/common.h>
#include <appl/RenderInterface.h>
#include <covise/covise_appproc.h>
#include "coCoviseInteractor.h"
#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cover/RenderObject.h>

#undef VERBOSE

using namespace covise;
using namespace opencover;
coCoviseInteractor::coCoviseInteractor(const char *n, RenderObject *o, const char *attrib)
    : coBaseCoviseInteractor(n, o, attrib)
{
    interactorList.append(this);
}

coCoviseInteractor::~coCoviseInteractor()
{
    interactorList.removeInteractor(this);
}

void coCoviseInteractor::sendFeedback(const char *info, const char *key, const char *data)
{
    static const char *empty = "";
    if (!data)
        data = empty;
    CoviseRender::set_feedback_info(d_feedbackInfo);
    CoviseRender::send_feedback_message(key, data);
}

void coCoviseInteractor::sendFeedbackMessage(int len, const char *data)
{

    int sLen = strlen(d_feedbackInfo);
    char *tmpData = new char[sLen + len];
    strcpy(tmpData, d_feedbackInfo + 1);
    memcpy(tmpData + sLen, data, len);

    Message message{ COVISE_MESSAGE_FEEDBACK , DataHandle{tmpData, sLen + len} };

    CoviseRender::appmod->send_ctl_msg(&message);

}

// ---------------------------------------------------------------------------

coInteractorList::coInteractorList()
{
    noDelete = 1;
}

void coInteractorList::removeInteractor(coInteractor *i)
{
    if (find(i))
        this->remove();
}

coInteractor *coInteractorList::findSame(coInteractor *inter)
{
    reset();
    while (current())
    {
        if (current()->isSameModule(inter))
        {
            return current();
        }
    }
    return 0;
}

void coInteractorList::removedObject(const char *objName)
{
    reset();
    while (current())
    {
        if (strcmp(current()->getObjName(), objName) == 0)
        {
            current()->removedObject();
        }
    }
}

coInteractorList opencover::interactorList;
