/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// by Lars Frenzel
// 28.10.1997

#include <util/common.h>
#include <util/unixcompat.h>
#include "VRVectorInteractor.h"
#include <cover/RenderObject.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <PluginUtil/coArrow.h>
#include <appl/RenderInterface.h>
#include <cover/input/VRKeys.h>

#include <osg/Geode>
#include <osg/Geometry>
#include <osg/MatrixTransform>

using namespace opencover;
using namespace covise;
// global stuff
VectorInteractorList opencover::vectorList;

VectorInteractor::VectorInteractor(const char *attrib, const char *sa, osg::Node *n)
{
    transform = NULL;
    node = n;
    sattrib = (char *)sa;
    feedback_information = NULL;
    subMenu = NULL;
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
    tmp = strtok(NULL, "\n"); // x1
    if (sscanf(tmp, "%f", &x1) != 1)
    {
        cerr << "VectorInteractor::VectorInteractor: scanf1 failed" << endl;
    }
    tmp = strtok(NULL, "\n"); // y1
    if (sscanf(tmp, "%f", &y1) != 1)
    {
        cerr << "VectorInteractor::VectorInteractor: scanf2 failed" << endl;
    }
    tmp = strtok(NULL, "\n"); // z1
    if (sscanf(tmp, "%f", &z1) != 1)
    {
        cerr << "VectorInteractor::VectorInteractor: scanf3 failed" << endl;
    }
    tmp = strtok(NULL, "\n"); // parameterName2
    parameterName2 = new char[strlen(tmp) + 1];
    strcpy(parameterName2, tmp);
    tmp = strtok(NULL, "\n"); // x2
    if (sscanf(tmp, "%f", &x2) != 1)
    {
        cerr << "VectorInteractor::VectorInteractor: scanf4 failed" << endl;
    }
    tmp = strtok(NULL, "\n"); // y2
    if (sscanf(tmp, "%f", &y2) != 1)
    {
        cerr << "VectorInteractor::VectorInteractor: scanf5 failed" << endl;
    }
    tmp = strtok(NULL, "\n"); // z2
    if (sscanf(tmp, "%f", &z2) != 1)
    {
        cerr << "VectorInteractor::VectorInteractor: scanf6 failed" << endl;
    }
    tmp = strtok(NULL, "\n"); // scaleFactor
    if (sscanf(tmp, "%f", &scaleFactor) != 1)
    {
        cerr << "VectorInteractor::VectorInteractor: scanf7 failed" << endl;
    }
    x1 *= scaleFactor;
    y1 *= scaleFactor;
    z1 *= scaleFactor;
    rotOnly = 1;
    if (strcasecmp(parameterName2, "none"))
    {
        rotOnly = 0;
    }
    transform = new osg::MatrixTransform();
    updatePosition();
    transform->addChild(getArrow());
    cover->getObjectsRoot()->addChild(transform);
#ifdef PINBOARD
    addMenue();
#endif

    delete[] buf;
}

VectorInteractor::~VectorInteractor()
{
    fprintf(stderr, "in delete VectorInteractor \n");
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
    delete[] parameterName2;
    if (transform)
    {
        fprintf(stderr, "removing arrow\n");
        transform->removeChild(transform->getChild(0));
        osg::Group *parent = transform->getParent(0);
        if (parent->removeChild(transform))
        {
            transform = NULL;
        }
    }

    if (this == vector)
        vector = NULL;
}

float VectorInteractor::getMinDist(float x, float y, float z)
{
    float dist1, dist2;
    float xd, yd, zd;
    xd = x - (x2 - x1);
    yd = y - (y2 - y1);
    zd = z - (z2 - z1);
    dist1 = xd * xd + yd * yd + zd * zd;
    xd = x - x2;
    yd = y - y2;
    zd = z - z2;
    dist2 = xd * xd + yd * yd + zd * zd;
    if (dist1 < dist2)
    {
        move = 0;
        return (dist1);
    }
    move = 1;
    return (dist2);
}

void VectorInteractor::updateValue(float x, float y, float z)
{
    if ((move) && !(rotOnly))
    {
        x2 = x;
        y2 = y;
        z2 = z;
    }
    else
    {
        x1 = x2 - x;
        y1 = y2 - y;
        z1 = z2 - z;
    }
    updatePosition();
}

void VectorInteractor::update(buttonSpecCell *)
{
    // TODO add a VECTOR entry to Menu
    updateParameter();
}

void VectorInteractor::updatePosition()
{
    osg::Matrix m;
    osg::Matrix m2;
    osg::Vec3 z(0.0, 0.0, 1.0), v(x1, y1, z1);
    float s = v.length();
    v.normalize();
    m.makeScale(s, s, s);
    m2.makeRotate(z, v);
    m.postMult(m2);
    m2.makeTranslate(x2, y2, z2);
    m.postMult(m2);
    transform->setMatrix(m);
    //fprintf(stderr,"Value: %f\n",value);
}

#ifdef PINBOARD
void
VectorInteractor::addMenue()
{
    if (menue)
        return;
    menue = 1;
    buttonSpecCell spec;

    strcpy(spec.name, "FloatVector");
    spec.actionType = BUTTON_SWITCH;
    spec.callback = &VectorInteractor::menuCallback;
    spec.calledClass = (void *)this;
    spec.state = 0.0;
    spec.dashed = false;
    spec.group = 0;
    VRPinboard::instance()->addButtonToMainMenu(&spec);

    return;
}

void VectorInteractor::updateMenu()
{
    buttonSpecCell spec;
    char buf[200];
    char buf2[200];
    char buf3[200];
    VRMenu *menu;
    sprintf(buf, "%s", moduleName);
    strcpy(buf3, buf);
    if (!(menu = VRPinboard::instance()->namedMenu(buf)))
    {
        spec.actionType = BUTTON_SUBMENU;
        strcpy(spec.subMenuName, buf);
        sprintf(buf2, "%s ...", moduleName);
        strcpy(spec.name, buf2);
        spec.callback = NULL;
        spec.calledClass = (void *)this;
        spec.state = false;
        spec.dashed = false;
        spec.group = cover->createUniqueButtonGroupId();
        VRPinboard::instance()->addButtonToMainMenu(&spec);
    }
    if (subMenu)
    {
        sprintf(buf, "%s %s", moduleName, subMenu);
        if (!(menu = VRPinboard::instance()->namedMenu(buf)))
        {
            spec.actionType = BUTTON_SUBMENU;
            strcpy(spec.subMenuName, buf);
            sprintf(buf2, "%s ...", subMenu);
            strcpy(spec.name, buf2);
            spec.callback = NULL;
            spec.calledClass = (void *)this;
            spec.state = false;
            spec.dashed = false;
            spec.group = cover->createUniqueButtonGroupId();
            VRPinboard::instance()->addButtonToNamedMenu(&spec, buf3);
        }
    }
    if (!(menu = VRPinboard::instance()->namedMenu(buf)))
    {
        fprintf(stderr, "Pinboard Error, could not create Menu %s \n", buf);
        return;
    }
    VRButton *button;
    button = menu->namedButton(parameterName);
    if (button)
    {
        //button->spec.sliderMin= min;
        //button->spec.sliderMax= max;
        //button->spec.state= value;
        //button->update();
    }
    else
    {
        strcpy(spec.name, parameterName);
        spec.actionType = BUTTON_SLIDER;
        spec.callback = &VectorInteractor::menuCallback;
        spec.calledClass = (void *)this;
        //spec.state= value;
        spec.dashed = false;
        spec.group = -1;
        //spec.sliderMin= min;
        //spec.sliderMax= max;
        menu->addButton(&spec);
    }
}
#endif

int VectorInteractor::isVectorInteractor(const char *n)
{
    return (!(strcmp(sattrib, n)));
}

void VectorInteractor::updateParameter()
{
    char buf[600];

    if (feedback_information)
    {
        CoviseRender::set_feedback_info(feedback_information);

        if (coVRMSController::instance()->isMaster())
        {
            fprintf(stdout, "\a");
            fflush(stdout);
            sprintf(buf, "%s\nFloatVector\n%.3f %.3f %.3f\n", parameterName, x1 / scaleFactor, y1 / scaleFactor, z1 / scaleFactor);
            CoviseRender::send_feedback_message("PARAM", buf);
            if (!rotOnly)
            {
                sprintf(buf, "%s\nFloatVector\n%.3f %.3f %.3f\n", parameterName2, x2, y2, z2);
                CoviseRender::send_feedback_message("PARAM", buf);
            }
            buf[0] = '\0';
            CoviseRender::send_feedback_message("EXEC", buf);
        }
    }
}

VectorInteractor *VectorInteractorList::find(float x, float y, float z)
{
    VectorInteractor *nearest;
    float near_dist, cur_dist;

    // find the nearest of the VectorInteractor-Spheres
    reset();

    near_dist = -1;
    nearest = NULL;

    while (current())
    {
        if (current()->getType() != 'Y')
        { // this is not a Menu entry
            // check distance
            cur_dist = current()->getMinDist(x, y, z);

            if (cur_dist < near_dist || near_dist == -1)
            {
                near_dist = cur_dist;
                nearest = current();
            }
        }

        next();
    }

    return (nearest);
}

/*
void VectorInteractorList::add( coDistributedObject *dobj, osg::Node *n )
{
   int i=0;
   char *attrib;
   char buf[100];
   sprintf(buf,"VECTOR%d",i);
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

      VectorInteractor *sl=find(sattrib);
      if(sl)
         sl->node=n;
      else
      {
         sl=new VectorInteractor(attrib,sattrib,n);
         append(sl);
      }
      i++;
      sprintf(buf,"VECTOR%d",i);
   }

}
*/

void VectorInteractorList::add(RenderObject *dobj, osg::Node *n)
{
    int i = 0;
    char buf[100];
    sprintf(buf, "VECTOR%d", i);
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

        VectorInteractor *sl = find(sattrib);
        if (sl)
            sl->node = n;
        else
        {
            sl = new VectorInteractor(attrib, sattrib, n);
            append(sl);
        }
        i++;
        sprintf(buf, "VECTOR%d", i);
    }
}

VectorInteractor *VectorInteractorList::find(const char *attrib)
{
    reset();
    while (current())
    {
        if (current()->isVectorInteractor(attrib))
            return (current());

        next();
    }

    return (NULL);
}

VectorInteractor *VectorInteractorList::find(osg::Node *n)
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

void VectorInteractorList::removeAll(osg::Node *n)
{
    reset();
    while (current())
    {
        if (current()->node == n)
        {
            //VectorInteractor *v=current();
            remove();
            //delete v;
        }
        else
            next();
    }
}

#ifdef PINBOARD
void VectorInteractor::menuCallback(void *slider, buttonSpecCell *spec)
{
    if (strcmp(spec->name, "FloatVector") == 0)
    {
        VRSceneGraph::instance()->manipulate(spec);
    }
    else
        ((VectorInteractor *)slider)->update(spec);
}
#endif

osg::Node *VectorInteractor::getArrow()
{
    static osg::ref_ptr<osg::Node> arrow;

    if (!arrow.get())
    {
        coArrow *a = new coArrow();
        a->setColor(osg::Vec4(1.0, 0.0, 0.0, 1.0));
        a->setName("FloatVector");
        arrow = a;
    }

    return arrow.get();
}

VectorInteractor *VectorInteractor::vector = NULL;
int VectorInteractor::menue = 0;
