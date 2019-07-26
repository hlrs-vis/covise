/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "coImageViewer.h"
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coMovableBackgroundMenuItem.h>
#include <OpenVRUI/coSliderMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/osg/OSGVruiMatrix.h>

#include <cover/coVRPluginSupport.h>
#include <config/CoviseConfig.h>
#include <img/coImage.h>
#include <OpenVRUI/sginterface/vruiMatrix.h>
#include <net/message.h>

#include <cover/OpenCOVER.h>
#include <grmsg/coGRSendCurrentDocMsg.h>
#include <grmsg/coGRSetDocPageSizeMsg.h>

using namespace vrui;
using namespace opencover;
using namespace grmsg;
using namespace covise;

coImageViewer::coImageViewer(const char *documentName, const char *imagePath, float vsize, float aspect_ratio)
{
    if (cover->debugLevel(2))
        fprintf(stderr, "coImageViewer %s %s\n", documentName, imagePath);

    documentName_ = new char[strlen(documentName) + 1];
    strcpy(documentName_, documentName);
    vsize_ = vsize;
    currentPageIndex_ = 0;
    pageSliderInMenu = false;

    //fprintf(stderr, "getBitmap from file\n");
    // get bitmap from file
    if (strlen(imagePath) != 0)
    {
        coImage *img = new coImage(imagePath);
        if (aspect_ratio > 0.)
            aspect_ = aspect_ratio;
        else
            aspect_ = ((float)img->getWidth()) / ((float)img->getHeight());

        //img->scaleExp2();
        int ns = img->getWidth();
        int nt = img->getHeight();
        int nr = 1;
        int nc = img->getNumChannels();
        const unsigned char *bitmap = img->getBitmap();
        //fprintf(stderr,"+++++++++img after scale: nc=%d ns=%d nt=%d nr=%d\n", nc, ns, nt, nr);
        imageItemList_.push_back(new coMovableBackgroundMenuItem(imagePath, (uint *)bitmap, nc, ns, nt, nr, aspect_, vsize));

        delete img;
    }
    else
        imageItemList_.push_back(new coMovableBackgroundMenuItem(imagePath, (uint *)NULL, 0, 0, 0, 1, 0, vsize_));

    /*
      float aspect = ((float)791)/((float)1024);
      imageItemList_.push_back(new coMovableBackgroundMenuItem(imagePath, aspect));
   */

    //fprintf(stderr,"create slider\n");
    zoomSlider_ = new coSliderMenuItem("zoom", 1.0, 4.0, 1.0);
    zoomSlider_->setMenuListener(this);

    pageSlider_ = new coSliderMenuItem("page", 1, 2, 1);
    pageSlider_->setInteger(true);
    pageSlider_->setMenuListener(this);

    resetZoomButton_ = new coButtonMenuItem("reset");
    resetZoomButton_->setMenuListener(this);

    char *buttonName = new char[strlen(documentName) + 20 + 1];
    char *menuName = new char[strlen(documentName) + 1024 + 1];
    sprintf(buttonName, "Document(%s)...", documentName);
    sprintf(menuName, "Document(%s)", documentName);

    documentInMenu_ = coCoviseConfig::isOn("COVER.Plugin.DocumentViewer.DocumentInMenu", true);
    if (documentInMenu_)
    {
        pinboardButton_ = new coSubMenuItem(buttonName);
        imageMenu_ = new coRowMenu(menuName, cover->getMenu());
        //pinboardButton_->setMenu(imageMenu_);
    }
    else
    {

        pinboardButton_ = NULL;

        if (coCoviseConfig::isOn("COVER.Plugin.DocumentViewer.DocumentInScene", false))
            imageMenu_ = new coRowMenu(menuName, NULL, 0, true);
        else
            // not yet implemented
            imageMenu_ = new coRowMenu(menuName, NULL, 0, false);

        imageMenu_->setVisible(false);

        float px, py, pz;
        string line = coCoviseConfig::getEntry("COVER.Plugin.DocumentViewer.DocumentPosition");
        if (!line.empty())
        {
            //TODO
            sscanf(line.c_str(), "%f %f %f", &px, &py, &pz);
        }
        else
        {
            px = -0.5 * cover->getSceneSize();
            py = 0;
            pz = 0.3 * cover->getSceneSize();
        }

        osg::Matrix dcsTransMat, dcsMat, preRot, tmp;
        preRot.makeRotate(osg::inDegrees(90.0f), 1.0f, 0.0f, 0.0f);
        dcsTransMat.makeTranslate(px, py, pz);
        dcsMat.mult(preRot, dcsTransMat);
        OSGVruiMatrix menuMatrix;
        menuMatrix.setMatrix(dcsMat);
        imageMenu_->setTransformMatrix(&menuMatrix);
        imageMenu_->setScale(cover->getSceneSize() / 2500);

        imageMenu_->setVisible(true);
        imageMenu_->show();
    }

    delete[] buttonName;
    delete[] menuName;

    // add submenu item 'document viewer' to the pinboard
    imageMenu_->add(imageItemList_[0]);
    if (!coCoviseConfig::isOn("COVER.Plugin.CfdGui", false))
    {
        imageMenu_->add(zoomSlider_);
        imageMenu_->add(resetZoomButton_);
    }
    if (pinboardButton_)
    {
        //fprintf(stderr,"adding button to pinboard\n");
        cover->getMenu()->add(pinboardButton_);
    }

    if (documentInMenu_)
        pinboardButton_->setMenu(imageMenu_);

    if (cover->debugLevel(5))
        fprintf(stderr, "coImageViewer::coImageViewer ENDE\n");

    objName_ = "";
}

coImageViewer::~coImageViewer()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "coImageViewer::~coImageViewer(%s)\n", documentName_);

    if (imageMenu_ && imageItemList_[currentPageIndex_])
    {
        imageMenu_->remove(imageItemList_[currentPageIndex_]);
    }

    for (size_t i = 0; i < imageItemList_.size(); ++i)
        delete imageItemList_[i];

    if (imageMenu_ && zoomSlider_)
    {
        imageMenu_->remove(zoomSlider_);
        delete zoomSlider_;
        zoomSlider_ = NULL;
    }
    if (imageMenu_ && pageSlider_)
    {
        imageMenu_->remove(pageSlider_);
        delete pageSlider_;
        pageSlider_ = NULL;
    }

    if (imageMenu_)
    {
        delete imageMenu_;
        imageMenu_ = NULL;
        if (pinboardButton_)
            pinboardButton_->setMenu(NULL);
    }

    if (pinboardButton_)
    {
        delete pinboardButton_;
        pinboardButton_ = NULL;
    }

    delete[] documentName_;
}

void
coImageViewer::preFrame()
{
    if (cover->debugLevel(6))
        fprintf(stderr, "coImageViewer::preFrame()\n");
    // get scale of visible image
    float s = imageItemList_[currentPageIndex_]->getScale();
    //fprintf(stderr,"page %d scale=%f\n", currentPageIndex_, s);
    // update slider
    zoomSlider_->setValue(s);
    if (cover->debugLevel(6))
        fprintf(stderr, "coImageViewer::preFrame() ENDE\n");
}

bool
coImageViewer::addImage(const char *imagePath)
{
    if (cover->debugLevel(2))
        fprintf(stderr, "coImageViewer::addImage [%s] to list of size [%d]\n", imagePath, (int)imageItemList_.size());

    //fprintf(stderr,"coImageViewer::addImage [%s] to list of size [%d]\n", imagePath, (int)imageItemList_.size());

    // check if this image is already in the list
    for (size_t i = 0; i < imageItemList_.size(); i++)
    {
        //fprintf(stderr,"comparing item [%d] [%s]\n", i, imageItemList_[i]->getName());
        if (strcmp(imageItemList_[i]->getName(), imagePath) == 0)
        {
            //fprintf(stderr,"image %s already in list\n", imagePath);
            return false;
        }
    }

    // get bitmap from file
    coImage *img = new coImage(imagePath);
    if (aspect_ == 0)
        aspect_ = ((float)img->getWidth()) / ((float)img->getHeight());

    //img->scaleExp2();
    int ns = img->getWidth();
    int nt = img->getHeight();
    int nr = 1;
    int nc = img->getNumChannels();
    const unsigned char *bitmap = img->getBitmap();
    //fprintf(stderr,"img after scale: nc=%d ns=%d nt=%d nr=%d\n", nc, ns, nt, nr);

    imageItemList_.push_back(new coMovableBackgroundMenuItem(imagePath, (uint *)bitmap, nc, ns, nt, nr, aspect_, vsize_));
    delete img;

    // page slider only if more pages and not GUI
    if (!pageSliderInMenu && !coCoviseConfig::isOn("COVER.Plugin.CfdGui", false))
    {
        imageMenu_->add(pageSlider_);
        pageSliderInMenu = true;
    }

    ////fprintf(stderr,"setting page slider max to %d\n", imageItemList_.size());
    pageSlider_->setMax(imageItemList_.size() + 0.1); // +0.1 for rounding problems of the slider

    if (cover->debugLevel(6))
        fprintf(stderr, "coImageViewer::addImage ENDe\n");
    return true;
}

void coImageViewer::menuEvent(coMenuItem *menuItem)
{
    if (menuItem == pageSlider_)
    {
        int lastPageIndex = currentPageIndex_;

        // +0.1 for rounding problems of the slider
        float fval = pageSlider_->getValue() + 0.1;
        int ival = (int)fval;
        currentPageIndex_ = ival - 1;

        imageMenu_->remove(imageItemList_[lastPageIndex]);
        imageMenu_->insert(imageItemList_[currentPageIndex_], 0);

        // set scale for current image
        float s = imageItemList_[currentPageIndex_]->getScale();
        zoomSlider_->setValue(s);
    }
    else if (menuItem == zoomSlider_)
    {
        float s = zoomSlider_->getValue();
        imageItemList_[currentPageIndex_]->setScale(s);
    }
    else if (menuItem == resetZoomButton_)
    {
        if (cover->debugLevel(6))
            fprintf(stderr, "reset in menuEvent\n");
        imageItemList_[currentPageIndex_]->reset();
    }
}

const char *
coImageViewer::getName()
{
    return documentName_;
}

int coImageViewer::getNumPages()
{
    return imageItemList_.size();
}

float coImageViewer::getVSize(int pageNo)
{
    return imageItemList_[pageNo - 1]->getVSize();
}

float coImageViewer::getAspect(int pageNo)
{
    return imageItemList_[pageNo - 1]->getAspect();
}

void
coImageViewer::setVisible(bool visible)
{
    //fprintf(stderr,"coImageViewer::setVisible  %d\n", visible);
    if (imageMenu_)
        imageMenu_->setVisible(visible);
}

void
coImageViewer::setPageNo(int pageNo)
{
    //fprintf(stderr,"coImageViewer::setPageNo  %d\n", pageNo);

    // there should be enough pages
    if (pageNo > (int)imageItemList_.size())
        return;

    int lastPageIndex = currentPageIndex_;
    //fprintf(stderr,"coImageViewer::setPageNo  %d    currentPageIndex_ [%d] imageItemList_ [%d]\n",pageNo, currentPageIndex_, (int)imageItemList_.size() );
    currentPageIndex_ = pageNo - 1;
    imageMenu_->remove(imageItemList_[lastPageIndex]);
    imageMenu_->insert(imageItemList_[currentPageIndex_], 0);

    //send current page name an object name (for identification) for GUI
    coGRSendCurrentDocMsg currentDocMsg(documentName_, imageItemList_[currentPageIndex_]->getName(), objName_);
    Message grmsg{ Message::UI , DataHandle{(char*)(currentDocMsg.c_str()), strlen((currentDocMsg.c_str())), false} };
    cover->sendVrbMessage(&grmsg);

    // set scale for current image
    float s = imageItemList_[currentPageIndex_]->getScale();
    zoomSlider_->setValue(s);
}

void
coImageViewer::setSize(int pageNo, float hsize, float vsize)
{
    //fprintf(stderr,"coImageViewer::setSize  pageNo=%d hsize=%f vsize=%f\n", pageNo, hsize, vsize);

    // there should be enough pages
    if (pageNo > (int)imageItemList_.size())
        return;

    for (size_t i = 0; i < imageItemList_.size(); ++i)
        imageItemList_[i]->setSize(hsize, vsize);
}
void
coImageViewer::setScale(float s)
{
    //imageItemList_[currentPageIndex_]->setScale(s);
    //zoomSlider_->setValue(s);

    imageMenu_->setScale(s);
}

void
coImageViewer::setPosition(float x, float y, float z)
{
    //fprintf(stderr,"coImageViewer::setPosition  %s %f %f %f\n", documentName_, x, y, z);

    if (!documentInMenu_)
    {
        OSGVruiMatrix *t, *r, *mat;
        r = new OSGVruiMatrix();
        t = new OSGVruiMatrix();
        mat = new OSGVruiMatrix();

        t->makeTranslate(x, y, z);
        r->makeEuler(0, 90, 0);
        //t.print(0, 1, "t: ",stderr);
        mat = dynamic_cast<OSGVruiMatrix *>(r->mult(t));
        imageMenu_->setTransformMatrix(mat);
        if (documentInMenu_)
            imageMenu_->setScale(cover->getSceneSize() / 2500);
    }
}

void
coImageViewer::setObjName(const char *objName)
{
    objName_ = objName;
}
