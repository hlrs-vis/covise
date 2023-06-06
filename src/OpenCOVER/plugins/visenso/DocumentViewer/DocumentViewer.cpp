/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "DocumentViewer.h"

#include "coImageViewer.h"

#include <config/CoviseConfig.h>
#include <cover/coVRFileManager.h>
#include <cover/RenderObject.h>
#include <cover/OpenCOVER.h>
#include <cover/coTranslator.h>
#include <cover/coVRPluginList.h>
#include <net/message.h>
#include <net/message_types.h>

#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/sginterface/vruiButtons.h>

#include <grmsg/coGRObjRegisterMsg.h>
#include <grmsg/coGRAddDocMsg.h>
#include <grmsg/coGRSetDocPositionMsg.h>
#include <grmsg/coGRSetDocPageMsg.h>
#include <grmsg/coGRSetDocPageSizeMsg.h>
#include <grmsg/coGRDocVisibleMsg.h>
#include <grmsg/coGRSetDocScaleMsg.h>
#include <grmsg/coGRSendDocNumbersMsg.h>
#include <grmsg/coGRAddDocMsg.h>

using namespace vrui;
using namespace grmsg;
using namespace opencover;
using namespace covise;

//-----------------------------------------------------------------------------

/// konstruktor
DocumentViewerPlugin::DocumentViewerPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
}

bool DocumentViewerPlugin::init()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\nDocumentViewerPlugin::init\n");

    vsize_ = coCoviseConfig::getFloat("COVER.Plugin.DocumentViewer.Vsize", -1);
    aspect_ratio_ = coCoviseConfig::getFloat("COVER.Plugin.DocumentViewer.AspectRatio", 0);
    coCoviseConfig::ScopeEntries documents = coCoviseConfig::getScopeEntries("COVER.Plugin.DocumentViewer", "Document");

    for (const auto &document : documents)
    {
        add(document.first.c_str(), document.second.c_str());
    }

    coverMenuButton_ = new coSubMenuItem("Documents...");
    cover->getMenu()->add(coverMenuButton_);
    documentsMenu_ = new coRowMenu("Documents", cover->getMenu());
    coverMenuButton_->setMenu(documentsMenu_);

    toggleDocumentsButton_ = new coCheckboxMenuItem("toggle Documents", true);
    toggleDocumentsButton_->setMenuListener(this);
    documentsMenu_->add(toggleDocumentsButton_);

    if (cover->debugLevel(2))
        fprintf(stderr, "DocumentViewerPlugin::init ENDE\n");
    return true;
}

DocumentViewerPlugin::~DocumentViewerPlugin()
{
}

bool
DocumentViewerPlugin::add(const char *documentName, const char *imageName)
{
    if (cover->debugLevel(2))
        fprintf(stderr, "DocumentViewerPlugin::add %s %s\n", documentName, imageName);

    // check filename
    // cover::getName also checks covise directories
    if (!documentName && !imageName)
    {
        if (cover->debugLevel(5))
            fprintf(stderr, "documentName or imageName is NULL\n");
        return false;
    }

    const char *imagePath;
    imagePath = coVRFileManager::instance()->getName(imageName);
    if (!imagePath)
    {
        fprintf(stderr, "WARNING: file %s does not exists\n", imageName);
        return false;
    }

    std::string localizedPath = coTranslator::translatePath(imagePath);

    // search if there is already an image viewer for this document
    map<string, coImageViewer *>::iterator it = findDocument_.find(string(documentName));

    coImageViewer *imgViewer;
    // if no image viewer found
    if (it == findDocument_.end())
    {
        // create new image viewer
        imgViewer = new coImageViewer(documentName, localizedPath.c_str(), vsize_, aspect_ratio_);

        // add image viewer to map with key documentname
        findDocument_[string(documentName)] = imgViewer;
    }
    // if there is already an image viewer for this document
    else
    {
        // add only the image to document
        imgViewer = (*it).second;
        imgViewer->addImage(localizedPath.c_str());
    }

    //send current size to Gui
    int pageNo = imgViewer->getNumPages();
    float vs = imgViewer->getVSize(pageNo);
    float a = imgViewer->getAspect(pageNo);
    float hs = a * vs;
    //fprintf(stderr,"COVER send coGRSetDocPageSizeMsg MESSAGE to Gui objectName=%s pageNo=%d hs=%f vs=%f\n", initialObjectName_.c_str(), pageNo, hs, vs);
    coGRSetDocPageSizeMsg pageSizeMsg(initialObjectName_.c_str(), pageNo, hs, vs);
    cover->sendGrMessage(pageSizeMsg);

    return true;
}

void
DocumentViewerPlugin::remove(const char *documentName)
{

    map<string, coImageViewer *>::iterator it = findDocument_.find(string(documentName));

    // get image viewer
    coImageViewer *imgViewer = (*it).second;

    findDocument_.erase(it);

    delete imgViewer;
}

void DocumentViewerPlugin::registerObjAtUi(string name)
{
    if (cover->debugLevel(2))
        fprintf(stderr, "DocumentViewerPlugin::registerObjAtUi name=%s\n", name.c_str());
    initialObjectName_ = name;
    coGRObjRegisterMsg regMsg(name.c_str(), NULL);
    cover->sendGrMessage(regMsg);
}

void
DocumentViewerPlugin::preFrame()
{

    if (cover->debugLevel(5))
        fprintf(stderr, "\nDocumentViewerPlugin::preFrame\n");

    /* Beispiel!
   static int counter=0;
   if (counter==300)
      setVisible("Covise", true);
   if (counter==350)
      setPosition("Covise", 100, 0, 100);
   if (counter==400)
      setPosition("Covise", 200, 0, 100);
   if (counter==500)
      setPageNo("Covise", 2);
   if (counter==500)
      setPageNo("Covise", 3);
//   if (counter==600)
//      setVisible("Covise", false);
   if (counter==700)
      setVisible("Auge", true);
   if (counter==800)
      setPageNo("Auge", 2);

  counter++;
*/

    coPointerButton *button = cover->getPointerButton();
    if (button->wasReleased(vruiButtons::TOGGLE_DOCUMENTS))
    {
        toggleDocumentsButton_->setState(!toggleDocumentsButton_->getState());
        menuEvent(toggleDocumentsButton_);
    }

    for (map<string, coImageViewer *>::iterator it = findDocument_.begin(); it != findDocument_.end(); it++)
    {
        coImageViewer *imgViewer = (*it).second;
        imgViewer->preFrame();
    }
}

void
DocumentViewerPlugin::addObject(const RenderObject *container, osg::Group *, const RenderObject *geomobj, const RenderObject *, const RenderObject *, const RenderObject *)
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\n--- DocumentViewerPlugin::addObject\n");

    if (geomobj)
    {
        const char *attr = geomobj->getAttribute("DOCUMENT");
        if (attr)
        {
            // register object at UI
            registerObjAtUi(geomobj->getName());
            initialObjectName_ = geomobj->getName();

            // get document name
            string tmp(attr);
            char *documentName = new char[strlen(attr) + 1];
            int pos = tmp.find(' ');
            strcpy(documentName, tmp.substr(0, pos).c_str());
            tmp = tmp.substr(pos + 1);
            ////fprintf(stderr,"documentname = %s\n", documentName);

            // get numpages
            int numPages;
            pos = tmp.find(' ');
            numPages = atoi(tmp.substr(0, pos).c_str());
            tmp = tmp.substr(pos + 1);
            ////fprintf(stderr,"numPages = %d\n", numPages);

            for (int i = 0; i < numPages; i++)
            {
                // get image name
                char *imageName = new char[strlen(attr) + 1];
                pos = tmp.find('.');
                pos = tmp.find(' ', pos);
                strcpy(imageName, tmp.substr(0, pos).c_str());
                tmp = tmp.substr(pos + 1);
                // fprintf(stderr,"imageName = %s\n", imageName);

                // create image viewer and add pages
                add(documentName, imageName);
                delete[] imageName;
            }

            // get pointer to image viewer
            coImageViewer *imgViewer;
            map<string, coImageViewer *>::iterator it = findDocument_.find(string(documentName));
            if (it != findDocument_.end())
            {
                imgViewer = (*it).second;
                imgViewer->setObjName(geomobj->getName());
            }
            else
            {
                imgViewer = NULL;
            }
            //coImageViewer *imgViewer = findImageViewerFromDocumentName(documentName);

            // add it to findCobj map
            if (imgViewer)
                findCobj_[string(geomobj->getName())] = imgViewer;

            // send min and max numPages to GUI
            int minPage = findMinPage(documentName);
            int maxPage;
            // check if single file
            if (minPage == -1)
            {
                minPage = 1;
                maxPage = 1;
            }
            else
            {
                maxPage = numPages + minPage - 1;
            }

            coGRSendDocNumbersMsg docNumMsg(container->getName(), minPage, maxPage);
            cover->sendGrMessage(docNumMsg);
        }
    }
}

void DocumentViewerPlugin::removeObject(const char *objName, bool)
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\n--- coVRRemoveObject objectName=[%s]\n", objName);

    if (!objName)
    {
        return;
    }

    // get viewer for this object
    map<string, coImageViewer *>::iterator it = findCobj_.find(string(objName));
    if (it != findCobj_.end())
    {
        coImageViewer *imgViewer = (*it).second;
        const char *dname = imgViewer->getName();
        remove(dname);

        // erase cobj,imgviewer from map
        findCobj_.erase(it);
    }
}

void
DocumentViewerPlugin::setVisible(const char *documentName, bool visible)
{
    //fprintf(stderr,"DocumentViewerPlugin::setVisible %s %d\n", documentName, visible);

    map<string, coImageViewer *>::iterator it = findDocument_.find(string(documentName));

    // get image viewer
    coImageViewer *imgViewer = NULL;

    if (it != findDocument_.end())
    {
        imgViewer = (*it).second;
    }

    if (imgViewer)
    {
        imgViewer->setVisible(visible);
    }
    else
    {
        if (cover->debugLevel(5))
            fprintf(stderr, "DocumentViewerPlugin::setVisible ERROR: could not find Document %s\n", documentName);
    }
}

void
DocumentViewerPlugin::setPageNo(const char *documentName, int pageNo)
{
    //fprintf(stderr,"DocumentViewerPlugin::setPageNo document [%s] number [%d]\n", documentName, pageNo);

    map<string, coImageViewer *>::iterator it = findDocument_.find(string(documentName));

    // get image viewer
    coImageViewer *imgViewer = (*it).second;
    if (imgViewer)
    {
        imgViewer->setPageNo(pageNo);
    }
    else
    {
        if (cover->debugLevel(5))
            fprintf(stderr, "DocumentViewerPlugin::setPageNo ERROR: could not find Document %s\n", documentName);
    }
}

void
DocumentViewerPlugin::setPosition(const char *documentName, float x, float y, float z)
{
    //fprintf(stderr,"DocumentViewerPlugin::setPosition %s %f %f %f\n", documentName, x, y, z);

    map<string, coImageViewer *>::iterator it = findDocument_.find(string(documentName));

    // get image viewer
    coImageViewer *imgViewer = (*it).second;

    if (imgViewer)
    {
        imgViewer->setPosition(x, y, z);
    }
    else
    {
        if (cover->debugLevel(5))
            fprintf(stderr, "DocumentViewerPlugin::setPosition ERROR: could not find Document %s\n", documentName);
    }
}

void
DocumentViewerPlugin::setSize(const char *documentName, int pageNo, float hsize, float vsize)
{
    //fprintf(stderr,"DocumentViewerPlugin::setSize %s %d %f %f\n", documentName, pageNo, hsize, vsize);

    map<string, coImageViewer *>::iterator it = findDocument_.find(string(documentName));

    // get image viewer
    coImageViewer *imgViewer = (*it).second;

    if (imgViewer)
    {
        imgViewer->setSize(pageNo, hsize, vsize);
    }
    else
    {
        if (cover->debugLevel(5))
            fprintf(stderr, "DocumentViewerPlugin::setSize ERROR: could not find Document %s\n", documentName);
    }
}

void
DocumentViewerPlugin::setScale(const char *documentName, float s)
{
    //fprintf(stderr,"DocumentViewerPlugin::setPosition %s %f %f %f\n", documentName, x, y, z);

    map<string, coImageViewer *>::iterator it = findDocument_.find(string(documentName));

    // get image viewer
    coImageViewer *imgViewer = (*it).second;

    if (imgViewer)
    {
        imgViewer->setScale(s);
    }
    else
    {
        if (cover->debugLevel(5))
            fprintf(stderr, "DocumentViewerPlugin::setPosition ERROR: could not find Document %s\n", documentName);
    }
}

int DocumentViewerPlugin::findMinPage(const char *documentName)
{
    //fprintf(stderr,"DocumentViewerPlugin::findMinPage %s\n", documentName);

    int i = strlen(documentName), j = 0;
    int num = 0, index_suffix = 0, index_num = 0;
    char c, nb_string[32];
    bool found = false;

    // Find number in documentName (from back to front)
    while (i >= 0 && !found)
    {
        c = documentName[i];

        if (isdigit(c))
        {
            //c is last number in the filepath
            found = true;
            index_suffix = i + 1;
            index_num = index_suffix;

            //find start of number
            while (isdigit(c))
            {
                c = documentName[(index_num--) - 2];
            }

            for (j = index_num; j < index_suffix; j++)
            {
                nb_string[j - index_num] = documentName[j];
            }

            nb_string[j - index_num] = '\0';
        }
        i--;
    }
    if (!found)
    {
        return -1;
    }
    else
    {
        sscanf(nb_string, "%d", &num);
    }

    return num;
}

void DocumentViewerPlugin::guiToRenderMsg(const grmsg::coGRMsg &msg) 
{
    if (cover->debugLevel(5))
        fprintf(stderr, "\n--- Plugin DocumentViewer coVRGuiToRenderMsg msg=[%s]", msg.getString().c_str());

    if (msg.isValid())
    {
        switch (msg.getType())
        {
        case coGRMsg::ADD_DOCUMENT:
        {
            auto &addDocMsg = msg.as<coGRAddDocMsg>();
            const char *documentName = addDocMsg.getDocumentName();
            //fprintf(stderr,"\n--- Plugin DocumentViewer addDoc name=[%s]\n", documentName );
            const char *imageName = addDocMsg.getImageName();
            add(documentName, imageName);
            setVisible(documentName, true);
        }
        break;
        case coGRMsg::SET_DOCUMENT_PAGE:
        {
            auto &setDocPageMsg = msg.as<coGRSetDocPageMsg>();
            const char *documentName = setDocPageMsg.getDocumentName();
            int pageNo = setDocPageMsg.getPage();
            //fprintf(stderr,"\n--- Plugin DocumentViewer setDocPage name=[%s] page=%d\n", documentName, pageNo );
            setPageNo(documentName, pageNo);
        }
        break;
        case coGRMsg::SET_DOCUMENT_POSITION:
        {
            auto &setDocPositionMsg = msg.as<coGRSetDocPositionMsg>();
            const char *documentName = setDocPositionMsg.getDocumentName();
            float x, y, z;
            setDocPositionMsg.getPosition(x, y, z);
            //fprintf(stderr,"\n--- Plugin DocumentViewer setDocPos name=[%s] pos=%f %f %f\n", documentName, x, y, z );
            setPosition(documentName, x, y, z);
        }
        break;
        case coGRMsg::SET_DOCUMENT_SCALE:
        {
            auto &setDocScaleMsg = msg.as<coGRSetDocScaleMsg>();
            const char *documentName = setDocScaleMsg.getDocumentName();
            float s = setDocScaleMsg.getScale();
            //fprintf(stderr,"\n--- Plugin DocumentViewer setDocPos name=[%s] scale=%f\n", documentName, s );
            setScale(documentName, s);
        }
        break;
        case coGRMsg::SET_DOCUMENT_PAGESIZE:
        {
            auto &setDocPageSizeMsg = msg.as<coGRSetDocPageSizeMsg>();
            const char *documentName = setDocPageSizeMsg.getDocumentName();
            float hsize = setDocPageSizeMsg.getHSize();
            float vsize = setDocPageSizeMsg.getVSize();
            int pageNo = setDocPageSizeMsg.getPageNo();
            //fprintf(stderr,"\n--- Plugin DocumentViewer SET_DOCUMENT_PAGESIZE name=[%s] pageNo=%d hsize%f vsize=%f\n", documentName, pageNo, hsize, vsize );
            setSize(documentName, pageNo, hsize, vsize);
        }
        break;
        case coGRMsg::DOC_VISIBLE:
        {
            auto &docVisibleMsg = msg.as<coGRDocVisibleMsg>();
            const char *documentName = docVisibleMsg.getDocumentName();
            bool show = docVisibleMsg.isVisible();
            //fprintf(stderr,"\n--- Plugin DocumentViewer docVis doc=[%s] vis=%d\n", documentName, show);
            setVisible(documentName, show);
        }
        break;
        default:
        {
            if (cover->debugLevel(3))
                fprintf(stderr, "NOT-USED\n");
        }
            break;
        }
    }
}

void DocumentViewerPlugin::toggleDocuments(bool visible)
{
    map<string, coImageViewer *>::iterator it;
    for (it = findDocument_.begin(); it != findDocument_.end(); it++)
    {
        (*it).second->setVisible(visible);
    }
}

coMenuItem *DocumentViewerPlugin::getMenuButton(const std::string &buttonName)
{
    if (buttonName == "toggleDocuments")
        return toggleDocumentsButton_;
    return NULL;
}

void DocumentViewerPlugin::menuEvent(coMenuItem *menuItem)
{

    if (cover->debugLevel(3))
        fprintf(stderr, "DocumentViewerPlugin::menuItem::menuEvent for %s\n", menuItem->getName());

    if (menuItem == toggleDocumentsButton_)
    {
        toggleDocuments(toggleDocumentsButton_->getState());
    }
}

COVERPLUGIN(DocumentViewerPlugin)
