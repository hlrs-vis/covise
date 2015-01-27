/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description:                                                        ++
// ++             Implementation of class InvAnnoManager                  ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 19.11.2001                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <X11/keysym.h>
#include <ctype.h>

#include "InvAnnotationManager.h"
#include "ModuleInfo.h"

#ifndef max
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif

// initialize instance_
InvAnnoManager *InvAnnoManager::instance_ = NULL;

//
// Constructor
//
InvAnnoManager::InvAnnoManager()
    : isInitialized_(false)
    , isActive_(false)
    , numFlags_(0)
    , flagGroup_(NULL)
    , flagSep_(NULL)
    , viewer_(NULL)
    , kbActive_(false)
    , trueNumFlags_(0)
    , hasDataObj_(false)
    , scale_(1.0)
{
}

void
InvAnnoManager::initialize(InvCoviseViewer *v)
{
    viewer_ = v;
    isInitialized_ = true;

    flagGroup_ = new SoGroup;
}

void
InvAnnoManager::reScale(const float &s)
{

    vector<InvAnnoFlag *>::iterator it;
    for (it = flags_.begin(); it != flags_.end(); ++it)
    {
        (*it)->reScale(s);
    }
}

//
//  access method
//
InvAnnoManager *
InvAnnoManager::instance()
{
    if (!instance_)
    {
        instance_ = new InvAnnoManager;
    }
    return instance_;
}

void
InvAnnoManager::activate(const int &mode)
{
    isActive_ = true;
    mode_ = mode;
}

void
InvAnnoManager::deactivate()
{
    isActive_ = false;
}

bool
InvAnnoManager::isActive() const
{
    return isActive_;
}

//
// Method
//
void
InvAnnoManager::add()
{
    // we do nothing if *this::initialize was NOT called
    // prior to this call
    initCheck();

    InvAnnoFlag *af = new InvAnnoFlag(numFlags_);
    activeFlag_ = af;
    flags_.push_back(af);
    flagGroup_->addChild(af->getSeparator());
    numFlags_++;
    trueNumFlags_++; // that 's the real number of flags
    af->reScale(scale_);

    viewer_->createAnnotationEditor(af);
}

void
InvAnnoManager::add(const InvAnnoFlag *af)
{
    initCheck();
    if (af)
    {
        activeFlag_ = const_cast<InvAnnoFlag *>(af);
        flags_.push_back(activeFlag_);
        //	cerr << "InvAnnoManager::add( af ) flags_.size() is now: " << flags_.size() << endl;
        flagGroup_->addChild(af->getSeparator());
        numFlags_++;
        trueNumFlags_++; // that 's the real number of flags

        // show it
        if (hasDataObj_)
            ((InvActiveNode *)af)->show();
        activeFlag_->setText();
    }
}

void
InvAnnoManager::update(const char *msg)
{

    initCheck();

    std::string msgStr(msg);

    // cut message in parts each part should begin with
    // "<InvAnnoFlag"
    std::string tag("<InvAnnoFlag");
    int offset(tag.size());

    int idx(msgStr.find_first_of(tag.c_str()));

    vector<std::string> strFlags;
    std::string strObj;
    (void)offset;
    while (idx != std::string::npos)
    {
        int end(msgStr.find_first_of("\\>\\", idx));
        strObj = msgStr.substr(idx, end - idx + 4);
        strFlags.push_back(strObj);
        idx = msgStr.find_first_of(tag.c_str(), idx + 1);
    }

    // now all string represenations of incoming objs are stored
    // create flags if they do not exist
    vector<InvAnnoFlag *>::iterator it;
    vector<std::string>::iterator strIt;

    std::string serStrings;
    vector<bool> create;
    for (strIt = strFlags.begin(); strIt != strFlags.end(); ++strIt)
    {
        // check if we already have a flag with string representation (*strIt)
        bool strExist(false);
        for (it = flags_.begin(); it != flags_.end(); ++it)
        {
            if ((*it) != NULL)
            {
                if (*(*it) == *strIt)
                {
                    strExist = true;
                }
            }
        }
        create.push_back(!strExist);
    }

    // now really add new flags if necessary
    int cnt(0);
    for (strIt = strFlags.begin(); strIt != strFlags.end(); ++strIt)
    {

        if (create[cnt])
        {
            InvAnnoFlag *af = new InvAnnoFlag(*strIt);
            if (af->isAlive())
            {
                add(af);
            }
            else
            {
                delete af;
            }
        }
        cnt++;
    }

    //    we may have to delete flags
    if ((int)strFlags.size() < trueNumFlags_)
    {
        vector<vector<InvAnnoFlag *>::iterator> remove;
        for (it = flags_.begin(); it != flags_.end(); ++it)
        {
            bool toRem(true);
            // check if we already have a flag with string representation (*strIt)
            if (*it != NULL)
            {
                for (strIt = strFlags.begin(); strIt != strFlags.end(); ++strIt)
                {
                    if (*(*it) == *strIt)
                    {
                        toRem = false;
                    }
                }
            }
            else
            {
                toRem = false;
            }
            if (toRem)
                remove.push_back(it);
        }

        cnt = 0;
        vector<vector<InvAnnoFlag *>::iterator>::iterator rit;
        for (rit = remove.begin(); rit != remove.end(); ++rit)
        {
            delete **rit;
            flags_.erase(*rit);
            trueNumFlags_--;
            deactivate();
            cnt++;
        }
    }
}

SoPath *
InvAnnoManager::pickFilterCB(void *me, const SoPickedPoint *pick)
{
    InvAnnoManager *mee = static_cast<InvAnnoManager *>(me);

    SoPath *filteredPath = NULL;

    if (mee->isActive_)
    {
        mee->actPickedPoint_ = (SoPickedPoint *)pick;
    }

    SoPath *p = pick->getPath();

    filteredPath = p;

    return filteredPath;
}

void
InvAnnoManager::selectionCB(void *me, SoPath *selectedObject)
{
    InvAnnoManager *mee = static_cast<InvAnnoManager *>(me);

    mee->initCheck();

    if (!mee->isActive_)
        return;

    int isAnnoFlg = 0;

    int len = selectedObject->getLength();
    int ii;
    char *selObjNm;

    SoNode *obj = NULL;
    int objIndex;

    int mode = mee->mode_;

    // we find out if the selected obj is a InvAnnotationFlag
    // and obtain its index from the name of the (sub)-toplevel
    // separator node
    for (ii = 0; ii < len; ii++)
    {
        obj = selectedObject->getNode(ii);
        char *tmp = (char *)obj->getName().getString();
        selObjNm = new char[1 + strlen(tmp)];
        char *chNum = new char[1 + strlen(tmp)];
        strcpy(selObjNm, tmp);
        if (strncmp(selObjNm, "ANNOTATION", 10) == 0)
        {
            strcpy(chNum, &selObjNm[11]);
            int ret = sscanf(chNum, "%d", &objIndex);
            if (ret != 1)
            {
                fprintf(stderr, "InvAnnoManager::selectionCB: sscanf failed\n");
            }
            isAnnoFlg = 1;
            break;
        }
    }

    // we have got an InvAnnoFlag
    // and remove it from the scene graph
    if (isAnnoFlg == 1)
    {
        if (obj->getTypeId() == SoSeparator::getClassTypeId())
        {
            if (mode == InvAnnoManager::EDIT)
            {
                vector<InvAnnoFlag *>::iterator it, selPos;
                // search flag with instance nr = objIndex;
                for (it = mee->flags_.begin(); it != mee->flags_.end(); ++it)
                {
                    if ((*it)->getInstance() == objIndex)
                    {
                        selPos = it;
                        break;
                    }
                }
                mee->viewer_->createAnnotationEditor(*selPos);
            }

            if (mode == InvAnnoManager::REMOVE)
            {
                // deletion of an InvAnnoFlag leads to a proper removal from the
                // scene graph by calling InvActiveNode::~InvActiveNode()
                bool del(false);
                vector<InvAnnoFlag *>::iterator it, remPos;
                // search flag with instance nr = objIndex;
                for (it = mee->flags_.begin(); it != mee->flags_.end(); ++it)
                {
                    if ((*it)->getInstance() == objIndex)
                    {
                        del = true;
                        remPos = it;
                    }
                }
                // delete flag and remove it from flags_
                if (del)
                {
                    delete *remPos;
                    mee->flags_.erase(remPos);
                    mee->trueNumFlags_--;
                    mee->deactivate();
                    mee->sendParameterData();
                }
            }
        }
    }
    // we create a new flag if anything else is selected
    else
    {
        if (mode != InvAnnoManager::MAKE)
            return;

        InvAnnoManager *mee = static_cast<InvAnnoManager *>(me);

        mee->add();

        InvAnnoFlag *af = mee->getActiveFlag();

        if (!af)
            return;

        SbVec3f camPos = mee->viewer_->getCamera()->position.getValue();

        af->setPickedPoint(mee->actPickedPoint_, camPos);

        InvAnnoFlag::selectionCB(af, selectedObject);

        mee->setKbActive();

        mee->deactivate();
    }
}

void
InvAnnoManager::deSelectionCB(void *me, SoPath *p)
{
    (void)p;
    InvAnnoManager *mee = static_cast<InvAnnoManager *>(me);

    mee->setKbInactive();
}

//
// Destructor
//
InvAnnoManager::~InvAnnoManager()
{
}

bool
InvAnnoManager::kbIsActive()
{
    return kbActive_;
}

void
InvAnnoManager::setKbActive()
{
    kbActive_ = true;
}

void
InvAnnoManager::setKbInactive()
{
    kbActive_ = false;
};

int
InvAnnoManager::getNumFlags() const
{
    return trueNumFlags_;
};

SbBool
InvAnnoManager::kbEventHandler(void *me, XAnyEvent *event)
{
    InvAnnoManager *mee = static_cast<InvAnnoManager *>(me);

    SbBool handled = TRUE;

    // return if keyboard shouldnot be active
    if (!mee->kbIsActive())
        return FALSE;

    InvAnnoFlag *af = mee->getActiveFlag();

    switch (event->type)
    {

    case KeyPress:
    {

        XKeyEvent *keyEvent = (XKeyEvent *)event;

        const int bufLen(32);
        char buf[bufLen];
        KeySym keysym_return;

        int gotLen = XLookupString(keyEvent, buf, bufLen, &keysym_return, NULL);
        buf[gotLen] = '\0';

#define XK_LATIN1

        if (gotLen > 0)
        {
            // @@@
            // these settings are not necessary machine independent

            switch (keysym_return)
            {
            case XK_Return:
            {
                mee->setKbInactive();
                mee->sendParameterData();
                break;
            }
            case XK_BackSpace:
            {
                af->setBackSpace(1);
                break;
            }
            case XK_adiaeresis:
            {
                strcpy(buf, "ae");
                af->setText(buf);
                break;
            }
            case XK_Adiaeresis:
            {
                strcpy(buf, "Ae");
                af->setText(buf);
                break;
            }
            case XK_odiaeresis:
            {
                strcpy(buf, "oe");
                af->setText(buf);
                break;
            }
            case XK_Odiaeresis:
            {
                strcpy(buf, "Oe");
                af->setText(buf);
                break;
            }
            case XK_udiaeresis:
            {
                strcpy(buf, "ue");
                af->setText(buf);
                break;
            }
            case XK_Udiaeresis:
            {
                strcpy(buf, "Ue");
                af->setText(buf);
                break;
            }
            default:
                if (isprint(buf[0]))
                {
                    af->setText(buf);
                }
                break;
            }
        }
        break;
    }

    case KeyRelease:
        // we pretend to handle KeyRelease events either otherwise
        // the real callback function may be confused
        handled = TRUE;
        break;

    default:
        handled = FALSE;
    }

    return handled;
}

void
InvAnnoManager::sendParameterData()
{

    std::string msg(ModuleInfo->getCoMsgHeader());
    msg = msg + std::string("\n");
    msg = msg + std::string("AnnotationString\n");
    msg = msg + std::string("String\n1\n");

    vector<InvAnnoFlag *>::iterator it = flags_.begin();

    std::string serStrings;
    for (; it != flags_.end(); ++it)
    {
        if (*it != NULL)
        {
            serStrings = serStrings + (*it)->serialize();
        }
    }

    if (serStrings.empty())
        serStrings = std::string("empty");

    msg = msg + serStrings;
    msg = msg + std::string("\n");

    // using sendCSFeedback is not the use it was written for but
    // for the moment we don't care
    if (viewer_)
    {
        viewer_->sendCSFeedback((char *)"PARAM", const_cast<char *>(msg.c_str()));
        viewer_->sendCSFeedback((char *)"PARREP-A", const_cast<char *>(msg.c_str()));
        viewer_->sendAnnotationMsg((char *)"ANNOTATION", const_cast<char *>(msg.c_str()));
    }
}

void
InvAnnoManager::showAll()
{
    vector<InvAnnoFlag *>::iterator it;
    for (it = flags_.begin(); it != flags_.end(); ++it)
    {
        if ((*it))
            (*it)->show();
    }
}

void
InvAnnoManager::hasDataObj(const bool &b)
{
    hasDataObj_ = b;
}

// set scale
void
InvAnnoManager::setSize(const SbBox3f &bb)
{
    float dx = fabs(bb.getMin()[0] - bb.getMax()[0]);
    float dy = fabs(bb.getMin()[1] - bb.getMax()[1]);
    float dz = fabs(bb.getMin()[2] - bb.getMax()[2]);

    float hsc = max(dx, dy);
    hsc = max(hsc, dz);

    hsc *= 0.2;

    scale_ = hsc;
}

void
InvAnnoManager::initCheck()
{
    if (!isInitialized_)
    {
        cerr << "FATAL: InvAnnoManager::initCheck() object uninitialized";
        cerr << "       will exit now!!!";
        exit(EXIT_FAILURE);
    }

    return;
}
