/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description: interactive motion handle for planes                   ++
// ++              Implementation of class InvPlaneMover                  ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 03.08.2001                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "InvPlaneMover.h"
#ifndef YAC
#include "InvMain.h"
#else
#include "InvMain_yac.h"
#endif

#ifndef YAC
#include "InvCommunicator.h"
#endif

#include <Inventor/nodes/SoLineSet.h>
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoRotationXYZ.h>
#include <Inventor/nodes/SoCone.h>
#include <Inventor/nodes/SoMaterial.h>
#include <Inventor/nodes/SoLabel.h>

//
// Constructor
//
//InvPlaneMover::InvPlaneMover(InvViewer *vv):
InvPlaneMover::InvPlaneMover()
    : show_(0)
    , distOffset_(0, 0, 0)
    , feedbackInfo_(NULL)
    , planeNormal_(0, 1, 0)
    , iNorm_(0, 1, 0)
    , motionMode_(InvPlaneMover::FREE)

{
    handle_ = new SoSeparator;
    handle_->ref();

    // our move-handle is a SoJackDragger with appropriate decoration
    // i.e. a square with a vector in its center

    handleSwitch_ = new SoSwitch();
    handleSwitch_->whichChild.setValue(SO_SWITCH_NONE);
    handle_->addChild(handleSwitch_);

    jDrag_ = new SoJackDragger;
    jDrag_->addFinishCallback(InvPlaneMover::dragFinishCB, this);

    handleDrawStyle_ = new SoDrawStyle;
    handleDrawStyle_->style.setValue(SoDrawStyle::FILLED);

    transl_ = new SoTranslation;
    handleSwitch_->addChild(transl_);

    fullRot_ = new SoRotation;
    handleSwitch_->addChild(fullRot_);

    scale_ = new SoScale;
    handleSwitch_->addChild(scale_);

    int ii;
    SoSeparator *plane[6];
    for (ii = 0; ii < 6; ii++)
    {
        plane[ii] = new SoSeparator;
        plane[ii]->addChild(handleDrawStyle_);
        plane[ii]->addChild(makePlane());
    }

    SoSeparator *empty[6];
    for (ii = 0; ii < 6; ii++)
    {
        empty[ii] = new SoSeparator;
        empty[ii]->addChild(handleDrawStyle_);
    }

    SoSeparator *scale[2];
    for (ii = 0; ii < 2; ii++)
    {
        scale[ii] = new SoSeparator;
        scale[ii]->addChild(handleDrawStyle_);
        //scale[ii]->addChild(makeArrow());
    }

    SoSeparator *arrow[2];
    for (ii = 0; ii < 2; ii++)
    {
        arrow[ii] = new SoSeparator;
        arrow[ii]->addChild(handleDrawStyle_);
        arrow[ii]->addChild(makeArrow());
    }

    transl_ = new SoTranslation;
    handleSwitch_->addChild(transl_);

    fullRot_ = new SoRotation;
    handleSwitch_->addChild(fullRot_);

    scale_ = new SoScale;
    handleSwitch_->addChild(scale_);

    handleSwitch_->addChild(jDrag_);

    jDrag_->setPart("rotator.rotator", arrow[0]);
    jDrag_->setPart("rotator.rotatorActive", arrow[1]);

    jDrag_->setPart("translator.yTranslator.translator", plane[0]);
    jDrag_->setPart("translator.xTranslator.translator", plane[2]);
    jDrag_->setPart("translator.zTranslator.translator", plane[4]);
    jDrag_->setPart("translator.yTranslator.translatorActive", plane[1]);
    jDrag_->setPart("translator.xTranslator.translatorActive", plane[3]);
    jDrag_->setPart("translator.zTranslator.translatorActive", plane[5]);

    jDrag_->setPart("translator.yzTranslator.translatorActive", empty[0]);
    jDrag_->setPart("translator.xzTranslator.translatorActive", empty[1]);
    jDrag_->setPart("translator.xyTranslator.translatorActive", empty[2]);
    jDrag_->setPart("translator.yzTranslator.translator", empty[3]);
    jDrag_->setPart("translator.xzTranslator.translator", empty[4]);
    jDrag_->setPart("translator.xyTranslator.translator", empty[5]);
}

// set scale
void
InvPlaneMover::setSize(const SbBox3f &bb)
{
    float dx = fabs(bb.getMin()[0] - bb.getMax()[0]);
    float dy = fabs(bb.getMin()[1] - bb.getMax()[1]);
    float dz = fabs(bb.getMin()[2] - bb.getMax()[2]);

    float hsc = qMax(dx, dy);
    hsc = qMax(hsc, dz);

    hsc *= 0.3f;

    SbVec3f s(hsc, hsc, hsc);

    scale_->scaleFactor.setValue(s);
}

// show and activate the handle
void
InvPlaneMover::show()
{
    show_ = 1;
    handleSwitch_->whichChild.setValue(SO_SWITCH_ALL);
}

// hide and deactivate the handle
void
InvPlaneMover::hide()
{
    show_ = 0;
    handleSwitch_->whichChild.setValue(SO_SWITCH_NONE);
}

// call this callback either from your application level selection CB
// or use it as selection CB
void
InvPlaneMover::selectionCB(void *me, SoPath *sp)
{
    InvPlaneMover *mee = static_cast<InvPlaneMover *>(me);

    // get the label next to the geometry = tail
    int len = sp->getLength();
    char objNme[20];
    int showFlg = 0;
    if (len > 2)
    {
        SoNode *grp = sp->getNode(len - 2);
        if (grp->getTypeId() == SoGroup::getClassTypeId())
        {
            int gLen = ((SoGroup *)grp)->getNumChildren();
            int i;
            for (i = 0; i < gLen; ++i)
            {
                SoNode *lbl = ((SoGroup *)grp)->getChild(i);
                if (lbl->getTypeId() == SoLabel::getClassTypeId())
                {
                    char *fbs = (char *)((SoLabel *)lbl)->label.getValue().getString();
                    size_t l = strlen(fbs);
                    // make sure that feedbackInfo_ is correctly allocated
                    if (mee->feedbackInfo_)
                    {
                        delete[] mee -> feedbackInfo_;
                    }
                    char *tmpStr = new char[l + 1];
                    strcpy(tmpStr, fbs);
                    // extract the object name (for feedback-info attached to CuttingSurface)
                    if (l > 1)
                    {
                        strncpy(objNme, &fbs[1], 14);
                        if (strncmp(objNme, "CuttingSurface", 14) == 0)
                            showFlg = 1;
                        // separate feedback-attribute and ignore-attribute
                        // separator is <IGNORE>
                        char *tok = strtok(tmpStr, "<IGNORE>");
                        if (tok)
                        {
                            mee->feedbackInfo_ = new char[1 + strlen(tok)];
                            strcpy(mee->feedbackInfo_, tok);
                            tok = strtok(NULL, "<IGNORE>");
                            if (tok)
                            {
                                float dum;
                                int idum;
                                int retval;
                                retval = sscanf(tok, "%f%f%f%f%d", &(mee->planeNormal_[0]), &(mee->planeNormal_[1]), &(mee->planeNormal_[2]), &dum, &idum);
                                if (retval != 5)
                                {
                                    std::cerr << "InvPlaneMover::selectionCB: sscanf failed" << std::endl;
                                    return;
                                }
                                fprintf(stderr, "planeNormal=(%f %f %f)\n",
                                        mee->planeNormal_[0],
                                        mee->planeNormal_[1],
                                        mee->planeNormal_[2]);
                                mee->setPosition(mee->distOffset_);
                            }
                        }
                    }
                    delete[] tmpStr;
                }
            }
        }
    }
    // the handle is shown if the feedback-attribute contains "CuttingSurface"
    // ..and if it is not shown at all
    if ((!mee->show_) && showFlg)
    {
        mee->show();
    }
}

#ifdef YAC
void InvPlaneMover::sendFeedback(float *)
#else
void InvPlaneMover::sendFeedback(float *data)
#endif

{
    if (!feedbackInfo_)
    {
        cerr << "InvPlaneMover::sendFeedback(..) have NO feedback-info can't send feedback message" << endl;
        return;
    }

#ifndef YAC
    else
    {
        char buf[256];
        char msg[1024];

        //data[0] .. data[2] contain the normal
        //data[3] contains the distance

        strcpy(msg, &feedbackInfo_[1]);

        sprintf(buf, "vertex\nVector\n3\n%f\n%f\n%f\n", data[0], data[1], data[2]);

        strcat(msg, buf);

        renderer->cm->sendCSFeedback("PARAM", msg);
        renderer->cm->sendCSFeedback("PARREP-A", msg);

        msg[0] = '\0';
        strcpy(msg, &feedbackInfo_[1]);

        sprintf(buf, "scalar\nScalar\n1\n%f\n", data[3]);
        strcat(msg, buf);

        renderer->cm->sendCSFeedback("PARAM", msg);
        renderer->cm->sendCSFeedback("PARREP-A", msg);

        buf[0] = '\0';

        msg[0] = '\0';
        strcpy(msg, &feedbackInfo_[1]);
        strcat(msg, buf);

        renderer->cm->sendCSFeedback("EXEC", msg);
    }
#endif
}

// call this callback either from your application level deselection CB
// or use it as deselection CB
void InvPlaneMover::deSelectionCB(void *me, SoPath *)
{
    InvPlaneMover *mee = static_cast<InvPlaneMover *>(me);

    if (mee->show_)
    {
        mee->hide();
    }
}

// call this callback either from your application level pickFilter CB
// or use it as pickFilter CB
SoPath *InvPlaneMover::pickFilterCB(void *me, const SoPickedPoint *pick)
{

    InvPlaneMover *mee = static_cast<InvPlaneMover *>(me);

    SoPath *filteredPath = pick->getPath();

    SbVec3f point = pick->getPoint();
    //    SbVec3f normal = pick->getNormal();

    mee->setPosition(point);

    return filteredPath;
}

void InvPlaneMover::setPosition(SbVec3f &point)
{

    // make sure that the normal points in direction to the camera
    SbVec3f camPos = renderer->viewer->getCamera()->position.getValue();
    SbVec3f no;
    if (camPos.dot(planeNormal_) < 0)
    {
        no = planeNormal_;
    }
    else
    {
        no = -planeNormal_;
    }

    fprintf(stderr, "inorm=(%f %f %f), no=(%f %f %f)\n",
            iNorm_[0], iNorm_[1], iNorm_[2],
            no[0], no[1], no[2]);

    SbRotation rota(iNorm_, no);

    distOffset_ = point;
    nnn_ = no;

    // the handle lays in front of the plane by this distance
    float offset_of_handle = 0.001 * scale_->scaleFactor.getValue()[0];

    //   if (!show) {
    transl_->translation = point - offset_of_handle * no;
    fullRot_->rotation = rota;
    //   }
    SbVec3f t(0, 0, 0);
    jDrag_->translation.setValue(t);
    jDrag_->rotation.setValue(SbRotation::identity());
}

// internal callback
void InvPlaneMover::dragFinishCB(void *me, SoDragger *drag)
{

    InvPlaneMover *mee = static_cast<InvPlaneMover *>(me);

    if (mee->show_)
    {

        SbVec3f t = ((SoJackDragger *)drag)->translation.getValue();

        int i;
        for (i = 0; i < 3; ++i)
            t[i] *= mee->scale_->scaleFactor.getValue()[i];

        SbRotation r = ((SoJackDragger *)drag)->rotation.getValue();

        SbVec3f n;
        SbVec3f ax;
        float angle;
        r.getValue(ax, angle);

        SbVec3f axN;
        mee->fullRot_->rotation.getValue().multVec(ax, axN);

        r.setValue(axN, angle);

        r.multVec(mee->nnn_, n);

        // we have to rotate the translation around the x-axis
        // (because we have a y-axis dragger)
        SbVec3f tt;
        n.normalize();

        // snap normal to the closest coordinate axis
        // here done by snaping it to the axis with the biggest projection onto it.
        if (mee->motionMode_ == InvPlaneMover::SNAP)
        {
            int axis;
            float mmax;
            int dir = 1;
            SbVec3f nn;

            if (n[0] * n[0] < n[1] * n[1])
            {
                axis = 1;
                mmax = n[1];
                if (n[1] < 0)
                    dir = -1;
                else
                    dir = +1;
                //dir = (int) copysign(1,n[1]);
            }
            else
            {
                axis = 0;
                mmax = n[0];
                if (n[0] < 0)
                    dir = -1;
                else
                    dir = +1;
                //dir = (int) copysign(1,n[0]);
            }
            if (mmax * mmax < n[2] * n[2])
            {
                axis = 2;
                if (n[2] < 0)
                    dir = -1;
                else
                    dir = +1;
                //dir = (int) copysign(1,n[2]);
            }

            switch (axis)
            {
            case 0:
                nn.setValue(1, 0, 0);
                break;
            case 1:
                nn.setValue(0, 1, 0);
                break;
            case 2:
                nn.setValue(0, 0, 1);
                break;
            }
            n = dir * nn;
        }

        tt = t[1] * n;

        float d;
        d = n.dot(tt + mee->distOffset_);

        float data[4];
        data[0] = n[0];
        data[1] = n[1];
        data[2] = n[2];
        data[3] = d;

        // send feedback message to contoller
        ((InvPlaneMover *)me)->sendFeedback(data);
    }
}

// internal callback
void InvPlaneMover::dragStartCB(void *me, SoDragger *drag)
{
    (void)me;
    (void)drag;
    //cerr << "InvPlaneMover::dragStartCB(..) called" << endl;

    //InvPlaneMover *mee = static_cast<InvPlaneMover *>(me);
    // fill in whatever you like to do at the beginning of a dragging action
}

// creates a part of a scene-graph which is an arrow suitable for the decoration
// of our SoJackDragger
SoGroup *InvPlaneMover::makeArrow()
{
    static float vertexPos[2][3] = {
        { 0.0f, 0.0f, 0.f },
        { 0.0f, -0.7f, 0.0f }
    };

    SoGroup *arrow = new SoGroup;

    SoLineSet *line = new SoLineSet;

    SoCoordinate3 *coords = new SoCoordinate3;
    coords->point.setValues(0, 2, vertexPos);

    arrow->addChild(coords);

    arrow->addChild(line);

    SoTranslation *transl = new SoTranslation;
    transl->translation.setValue(0.0f, -0.85f, 0.0f);
    arrow->addChild(transl);

    SoRotationXYZ *rot = new SoRotationXYZ;
    rot->angle = (float)M_PI;
    rot->axis = SoRotationXYZ::Z;
    arrow->addChild(rot);

    SoCone *tip = new SoCone;
    tip->bottomRadius = 0.1f;
    tip->height = 0.3f;

    arrow->addChild(tip);

    return arrow;
}

// creates a part of a scene-graph which is a plane suitable for the decoration
// of our SoJackDragger
SoGroup *InvPlaneMover::makePlane()
{

    static float vertexPos[4][3] = {
        { 0, 0.5, -0.5 },
        { 0, 0.5, 0.5 },
        { 0, -0.5, 0.5 },
        { 0, -0.5, -0.5 }
    };

    static int indices[4] = { 3, 2, 1, 0 };

    SoGroup *plane = new SoGroup;
    plane->ref();

    SoMaterial *mat = new SoMaterial;

    mat->ambientColor.setValue(0.3f, 0.1f, 0.1f);
    mat->diffuseColor.setValue(0.8f, 0.7f, 0.2f);
    mat->specularColor.setValue(0.4f, 0.3f, 0.1f);
    mat->transparency = 0.3f;
    plane->addChild(mat);

    SoMaterialBinding *bndng = new SoMaterialBinding;
    bndng->value = SoMaterialBinding::DEFAULT;
    plane->addChild(bndng);

    SoCoordinate3 *coords = new SoCoordinate3;
    coords->point.setValues(0, 4, vertexPos);

    plane->addChild(coords);

    SoIndexedFaceSet *faceSet = new SoIndexedFaceSet;
    faceSet->coordIndex.setValues(0, 4, indices);

    plane->addChild(faceSet);

    return plane;
}

// set snap to axis mode
void InvPlaneMover::setSnapToAxis()
{
    motionMode_ = InvPlaneMover::SNAP;
}

// set free motion (default)
void InvPlaneMover::setFreeMotion()
{
    motionMode_ = InvPlaneMover::FREE;
}

//
// Destructor
//
InvPlaneMover::~InvPlaneMover()
{
}
