/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef INV_TEXT_MANAGER_H
#define INV_TEXT_MANAGER_H

#include <Inventor/nodes/SoGroup.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoOrthographicCamera.h>
#include <Inventor/nodes/SoTranslation.h>
#include <Inventor/nodes/SoCallback.h>

#include <string>
#include <map>

class InvTextManager
{

public:
    InvTextManager();
    static SoCallbackCB updateCallback;
    void setAnnotation(const std::string &key, const std::string &text, float size = 20.f, const char *font = NULL);
    void removeAnnotation(const std::string &key);
    SoNode *getRootNode();
    ~InvTextManager();

private:
    SoSeparator *m_root;
    SoOrthographicCamera *m_camera;
    SoCallback *m_callback;
    SoTranslation *m_trans;
    SoGroup *m_group;

    typedef std::map<std::string, SoSeparator *> AnnotationMap;
    AnnotationMap m_annotationMap;
};
#endif
