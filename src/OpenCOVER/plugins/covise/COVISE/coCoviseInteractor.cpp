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
    interactorList.push_back(this);
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
}

void coInteractorList::removeInteractor(coInteractor *i)
{
    remove(i);
}

coInteractor *coInteractorList::findSame(coInteractor *inter)
{
    for(const auto &it: *this)
    {
        if (it->isSameModule(inter))
        {
            return it;
        }
    }
    return 0;
}

void coInteractorList::removedObject(const char *objName)
{
    remove_if([objName](coInteractor* it) { return (strcmp(it->getObjName(), objName) == 0); });
}

coInteractorList opencover::interactorList;
