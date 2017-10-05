/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


//**************************************************************************
//
// * Description    : Inventor interactive COVISE renderer
//
// * Class(es)      :
//
// * inherited from :
//
// * Author  : Dirk Rantzau
//
// * History : 24.07.97 V 1.0
//
//**************************************************************************

#include "InvTextManager.h"

#include <Inventor/nodes/SoText2.h>
#include <Inventor/nodes/SoFont.h>
#include <Inventor/actions/SoGLRenderAction.h>

//=========================================================================
//
//=========================================================================
InvTextManager::InvTextManager()
{
    m_root = new SoSeparator();
    m_root->ref();

    m_camera = new SoOrthographicCamera;
    m_root->addChild(m_camera);

    m_callback = new SoCallback;
    m_callback->setCallback(updateCallback, this);
    m_root->addChild(m_callback);

    m_trans = new SoTranslation;
    m_root->addChild(m_trans);

    m_group = new SoGroup;
    m_root->addChild(m_group);
}

void InvTextManager::updateCallback(void *userdata, SoAction *ac)
{
    // fix to upper left corner
    const SoGLRenderAction *gl = dynamic_cast<SoGLRenderAction *>(ac);
    if (!gl)
        return;
    const InvTextManager *cm = static_cast<InvTextManager *>(userdata);
    if (!cm)
        return;
    if (!cm->m_camera)
        return;
    if (!cm->m_trans)
        return;
    const SbViewportRegion vp = gl->getViewportRegion();
    const float aspect = vp.getViewportAspectRatio();
    const SbVec2s vpSize = vp.getViewportSizePixels();
    const int xMargin = 40; // 20 Pixels at each side as our rectangle has margins on both sides ...
    const int yMargin = 60; // 30 Pixels at each side as our rectangle has margins on both sides ...
    float xMarginRel = 0.;
    float yMarginRel = 0.;
    if (vpSize[0] > 2 * xMargin)
        xMarginRel = (float)(vpSize[0] - xMargin) / (vpSize[0]);
    if (vpSize[1] > 2 * yMargin)
        yMarginRel = (float)(vpSize[1] - yMargin) / (vpSize[1]);

    if (aspect > 1.)
        cm->m_trans->translation.setValue(-aspect * xMarginRel, +0.95f * yMarginRel, 0.0);
    else
        cm->m_trans->translation.setValue(-1.0f * xMarginRel, +0.95f / aspect * yMarginRel, 0.0);
}

//=========================================================================
//
//=========================================================================
SoNode *InvTextManager::getRootNode()
{
    return m_root;
}

//=========================================================================
//
//=========================================================================
void InvTextManager::setAnnotation(const std::string &key, const std::string &annotation, float size, const char *fontname)
{
    SoText2 *text = NULL;
    SoSeparator *root = NULL;
    SoFont *font = NULL;
    AnnotationMap::iterator it = m_annotationMap.find(key);
    if (it != m_annotationMap.end())
    {
        root = it->second;
        font = static_cast<SoFont *>(root->getChild(0));
        text = static_cast<SoText2 *>(root->getChild(1));
    }
    else
    {
        root = new SoSeparator;
        m_group->addChild(root);
        m_annotationMap[key] = root;

        font = new SoFont;
        root->addChild(font);

        text = new SoText2;
        root->addChild(text);
    }

    font->name.setValue(fontname ? fontname : "Helvetica");
    font->size.setValue(size);
    text->string.setValue(annotation.c_str());
}

void InvTextManager::removeAnnotation(const std::string &key)
{
    AnnotationMap::iterator it = m_annotationMap.find(key);
    if (it != m_annotationMap.end())
    {
        SoSeparator *root = it->second;
        m_group->removeChild(root);
        m_annotationMap.erase(it);
    }
}

//=========================================================================
//
//=========================================================================
InvTextManager::~InvTextManager()
{
    m_root->unref();
}
