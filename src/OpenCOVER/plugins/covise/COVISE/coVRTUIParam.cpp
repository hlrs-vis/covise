/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// by Lars Frenzel
// 28.10.1997

#include "coVRTUIParam.h"
#include <util/common.h>
#include <appl/RenderInterface.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cover/RenderObject.h>

#include <cover/coVRTui.h>

using namespace opencover;
using namespace covise;
TUIParamParent::TUIParamParent(const char *n)
{

    name = new char[strlen(n) + 1];
    strcpy(name, n);
    element = (coTUIElement *)new coTUITab(name, coVRTui::instance()->mainFolder->getID());
    element->setEventListener(this);
    element->setPos(3, 0);
}

TUIParamParent::~TUIParamParent()
{
    delete[] name;
}

void TUIParamParent::tabletEvent(coTUIElement *)
{
}

TUIParam::TUIParam(const char *attrib, const char *sa, osg::Node *n)
{
    node = n;
    sattrib = (char *)sa;
    feedback_information = NULL;
    type = -1;
    xPos = 0;
    yPos = 0;
    element = NULL;
    moduleName = NULL;
    parameterName = NULL;
    feedback_information = new char[strlen(sattrib) + 1];
    strcpy(feedback_information, sattrib);
    char *tmp = strchr(feedback_information, '\n');
    if (tmp)
    {
        tmp = strchr(tmp + 1, '\n');
        if (tmp)
        {
            tmp = strchr(tmp + 1, '\n');
            if (tmp)
            {
                *(tmp + 1) = '\0';
            }
        }
    }
    char *buf;
    buf = new char[strlen(attrib) + 1];
    strcpy(buf, attrib);
    tmp = strtok(buf, "\n"); // mod
    moduleName = new char[strlen(tmp) + 20];
    strcpy(moduleName, tmp + 1);
    tmp = strtok(NULL, "\n"); // inst
    strcat(moduleName, " ");
    strcat(moduleName, tmp);
    tmp = strtok(NULL, "\n"); // host
    tmp = strtok(NULL, "\n"); // parameterName
    parameterName = new char[strlen(tmp) + 1];
    strcpy(parameterName, tmp);
    tmp = strtok(NULL, "\n"); // parentName
    char *parentName = new char[strlen(tmp) + 1];
    strcpy(parentName, tmp);
    tmp = strtok(NULL, "\n"); // text
    parameterText = new char[strlen(tmp) + 1];
    strcpy(parameterText, tmp);
    tmp = strtok(NULL, "\n"); // xpos
    if (sscanf(tmp, "%d", &xPos) != 1)
    {
        cerr << "TUIParam::TUIParam: sscanf1 failed" << endl;
    }
    tmp = strtok(NULL, "\n"); // ypos
    if (sscanf(tmp, "%d", &yPos) != 1)
    {
        cerr << "TUIParam::TUIParam: sscanf2 failed" << endl;
    }
    tmp = strtok(NULL, "\n"); // paramType
    char *parameterType = new char[strlen(tmp) + 1];
    strcpy(parameterType, tmp);
    if ((parent = tuiParamParentList.find(parentName)) == NULL)
    {
        parent = new TUIParamParent(parentName);
        tuiParamParentList.append(parent);
    }
    cerr << "X" << parameterType << "X" << endl;
    if (strncmp(parameterType, "floatSlider", 11) == 0)
        type = TUI_FLOAT_SLIDER;
    else if (strncmp(parameterType, "intSlider", 9) == 0)
        type = TUI_INT_SLIDER;
    else if (strncmp(parameterType, "float", 5) == 0)
        type = TUI_FLOAT;
    else if (strncmp(parameterType, "int", 3) == 0)
        type = TUI_INT;
    else if (strncmp(parameterType, "bool", 4) == 0)
        type = TUI_BOOL;
    if (type == TUI_FLOAT_SLIDER || type == TUI_INT_SLIDER)
    {
        tmp = strtok(NULL, "\n"); // min
        if (sscanf(tmp, "%f", &min) != 1)
        {
            cerr << "TUIParam::TUIParam: sscanf3 failed" << endl;
        }
        tmp = strtok(NULL, "\n"); // max
        if (sscanf(tmp, "%f", &max) != 1)
        {
            cerr << "TUIParam::TUIParam: sscanf4 failed" << endl;
        }
        tmp = strtok(NULL, "\n"); // value
        if (sscanf(tmp, "%f", &value) != 1)
        {
            cerr << "TUIParam::TUIParam: sscanf5 failed" << endl;
        }
        if (type == TUI_FLOAT_SLIDER)
        {
            coTUIFloatSlider *fs;
            fs = new coTUIFloatSlider(parameterText, parent->element->getID());
            element = (coTUIElement *)fs;
            fs->setMin(min);
            fs->setMax(max);
            fs->setValue(value);
        }
        else
        {
            coTUISlider *is;
            is = new coTUISlider(parameterText, parent->element->getID());
            element = (coTUIElement *)is;
            is->setMin((int)min);
            is->setMax((int)max);
            is->setValue((int)value);
        }
        coTUILabel *tl;
        tl = new coTUILabel(parameterText, parent->element->getID());
        tl->setPos(xPos, yPos);
        element->setPos(xPos + 1, yPos);
    }
    if (type == TUI_FLOAT || type == TUI_INT)
    {
        tmp = strtok(NULL, "\n"); // value
        if (sscanf(tmp, "%f", &value) != 1)
        {
            cerr << "TUIParam::TUIParam: sscanf5 failed" << endl;
        }
        tmp = strtok(NULL, "\n"); // step
        if (sscanf(tmp, "%f", &step) != 1)
        {
            cerr << "TUIParam::TUIParam: sscanf5 failed" << endl;
        }

        if (type == TUI_INT)
        {
            coTUIEditIntField *fs;
            fs = new coTUIEditIntField(parameterText, parent->element->getID());
            element = (coTUIElement *)fs;
            fs->setValue((int)value);
        }
        else if (type == TUI_FLOAT)
        {
            coTUIEditFloatField *fs;
            fs = new coTUIEditFloatField(parameterText, parent->element->getID());
            element = (coTUIElement *)fs;
            fs->setValue(value);
        }
        coTUILabel *tl;
        tl = new coTUILabel(parameterText, parent->element->getID());
        tl->setPos(xPos, yPos);
        element->setPos(xPos + 1, yPos);
    }
    if (type == TUI_BOOL)
    {
        tmp = strtok(NULL, "\n"); // value
        if (strcmp(tmp, "true") == 0 || strcmp(tmp, "true") == 0 || strcmp(tmp, "1") == 0)
        {
            state = true;
        }
        else
        {
            state = false;
        }
        coTUIToggleButton *fs;
        fs = new coTUIToggleButton(parameterText, parent->element->getID());
        element = (coTUIElement *)fs;
        fs->setState(state);
        coTUILabel *tl;
        tl = new coTUILabel(parameterText, parent->element->getID());
        tl->setPos(xPos, yPos);
        element->setPos(xPos + 1, yPos);
    }
    element->setEventListener(this);

    delete[] parentName;
    delete[] buf;
}

TUIParam::~TUIParam()
{
    if (feedback_information == NULL)
    {
        fprintf(stderr, "wrong Delete\n");
        return;
    }
    delete[] feedback_information;
    feedback_information = NULL;

    delete[] sattrib;
    delete[] moduleName;
    delete[] parameterName;
    if (element)
    {
        delete element;
    }
}

void TUIParam::tabletPressEvent(coTUIElement *)
{
    updateParameter(false);
}

void TUIParam::tabletReleaseEvent(coTUIElement *)
{
    updateParameter(true);
    exec();
}

void TUIParam::tabletEvent(coTUIElement *)
{
    updateParameter(false);
    if (type == TUI_FLOAT || type == TUI_INT || type == TUI_BOOL)
    {
        exec();
    }
}

int TUIParam::isTUIParam(const char *n)
{
    return (!(strcmp(sattrib, n)));
}

void TUIParam::updateParameter(bool force)
{
    char buf[600];
    if (type == TUI_FLOAT_SLIDER)
    {
        coTUIFloatSlider *fs = (coTUIFloatSlider *)element;
        value = fs->getValue();
        if ((!force) && (cover->frameTime() - lastTime) < 0.5)
            return;
    }
    else if (type == TUI_INT_SLIDER)
    {
        coTUISlider *fs = (coTUISlider *)element;
        value = (float)fs->getValue();
        if ((!force) && (cover->frameTime() - lastTime) < 0.5)
            return;
    }
    else if (type == TUI_INT)
    {
        coTUIEditIntField *fs = (coTUIEditIntField *)element;
        value = (float)fs->getValue();
    }
    else if (type == TUI_FLOAT)
    {
        coTUIEditFloatField *fs = (coTUIEditFloatField *)element;
        value = fs->getValue();
    }
    lastTime = cover->frameTime();
    if (feedback_information)
    {
        CoviseRender::set_feedback_info(feedback_information);

        if (coVRMSController::instance()->isMaster())
        {
            if (type == TUI_FLOAT_SLIDER)
            {
                sprintf(buf, "%s\nFloatSlider\n%.3f %.3f %.3f\n", parameterName, min, max, value);
            }
            else if (type == TUI_INT_SLIDER)
            {
                sprintf(buf, "%s\nIntSlider\n%d %d %d\n", parameterName, (int)min, (int)max, (int)value);
            }
            else if (type == TUI_INT)
            {
                sprintf(buf, "%s\nIntScalar\n%d\n", parameterName, (int)value);
            }
            else if (type == TUI_FLOAT)
            {
                sprintf(buf, "%s\nFloatScalar\n%f\n", parameterName, value);
            }
            else if (type == TUI_BOOL)
            {
                sprintf(buf, "%s\nBool\n%d\n", parameterName, (int)state);
            }
            CoviseRender::send_feedback_message("PARAM", buf);
        }
    }
}

void TUIParam::exec()
{
    char buf[60];

    if (feedback_information)
    {
        CoviseRender::set_feedback_info(feedback_information);

        if (coVRMSController::instance()->isMaster())
        {
            fprintf(stdout, "\a");
            fflush(stdout);
            buf[0] = '\0';
            CoviseRender::send_feedback_message("EXEC", buf);
        }
    }
}

/*
void TUIParamList::add( coDistributedObject *dobj, osg::Node *n )
{
   int i=0;
   char *attrib;
   char buf[100];
   sprintf(buf,"TUI%d",i);
   while((attrib=dobj->getAttribute(buf)))
   {
      char *sattrib=new char[strlen(attrib)+1];
      strcpy(sattrib,attrib);
      char *tmp= strchr(sattrib,'\n');
      if(tmp)
      {
         tmp= strchr(tmp+1,'\n');
         if(tmp)
         {
            tmp= strchr(tmp+1,'\n');
            if(tmp)
            {
               tmp= strchr(tmp+1,'\n');
               if(tmp)
               {
                  tmp= strchr(tmp+1,'\n');
                  if(tmp)
                  {
                     *tmp='\0';
                  }
               }
            }
         }
      }

      TUIParam *sl=find(sattrib);
      if(sl)
         sl->node=n;
      else
      {
         sl=new TUIParam(attrib,sattrib,n);
         append(sl);
      }
      i++;
      sprintf(buf,"TUI%d",i);
   }

}
*/

void TUIParamList::add(RenderObject *dobj, osg::Node *n)
{
    int i = 0;
    char buf[100];
    sprintf(buf, "TUI%d", i);
    while (const char *attrib = dobj->getAttribute(buf))
    {
        char *sattrib = new char[strlen(attrib) + 1];
        strcpy(sattrib, attrib);
        char *tmp = strchr(sattrib, '\n');
        if (tmp)
        {
            tmp = strchr(tmp + 1, '\n');
            if (tmp)
            {
                tmp = strchr(tmp + 1, '\n');
                if (tmp)
                {
                    tmp = strchr(tmp + 1, '\n');
                    if (tmp)
                    {
                        tmp = strchr(tmp + 1, '\n');
                        if (tmp)
                        {
                            *tmp = '\0';
                        }
                    }
                }
            }
        }

        TUIParam *sl = find(sattrib);
        if (sl)
        {
            sl->node = n;
            // should update TUI element here to reflect the current parameters
        }
        else
        {
            sl = new TUIParam(attrib, sattrib, n);
            append(sl);
        }
        i++;
        sprintf(buf, "TUI%d", i);
    }
}

TUIParamParent *TUIParamParentList::find(const char *name)
{
    reset();
    while (current())
    {
        if (strcmp(current()->name, name) == 0)
            return (current());

        next();
    }

    return (NULL);
}

TUIParam *TUIParamList::find(const char *attrib)
{
    reset();
    while (current())
    {
        if (current()->isTUIParam(attrib))
            return (current());

        next();
    }

    return (NULL);
}

TUIParam *TUIParamList::find(osg::Node *n)
{
    reset();
    while (current())
    {
        if (current()->node == n)
            return (current());

        next();
    }

    return (NULL);
}

void TUIParamList::removeAll(osg::Node *n)
{
    reset();
    while (current())
    {
        if (current()->node == n)
            remove();
        else
            next();
    }
}

// global stuff
TUIParamList opencover::tuiParamList;
TUIParamParentList opencover::tuiParamParentList;
