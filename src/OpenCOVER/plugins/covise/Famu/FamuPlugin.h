/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef FAMU_PLUGIN_H
#define FAMU_PLUGIN_H

#include <OpenVRUI/coMenuItem.h>
#include <cover/coVRPlugin.h>

namespace vrui
{
class coRowMenu;
class coSubMenuItem;
class coCheckboxMenuItem;
class coPotiMenuItem;
class coSliderMenuItem;
class coButtonMenuItem;
class coLabelMenuItem;
}

using namespace vrui;
using namespace opencover;

class FamuPlugin : public coVRPlugin, public coMenuListener
{
private:
    bool firsttime;
    //FamuInteractor *wireFamu;
    coInteractor *inter;
    coSubMenuItem *pinboardButton;
    coRowMenu *FamuSubmenu;

    coCheckboxMenuItem *firstCheckbox;
    coCheckboxMenuItem *secondCheckbox;
    coCheckboxMenuItem *thirdCheckbox;
    coCheckboxMenuItem *fourthCheckbox;
    coCheckboxMenuItem *resetCheckbox;
    coCheckboxMenuItem *isolCheckbox;

    coSliderMenuItem *xDistSlider;
    coSliderMenuItem *yDistSlider;
    coSliderMenuItem *zDistSlider;
    coSliderMenuItem *scaleSlider;
    coSliderMenuItem *XYSlider;
    coSliderMenuItem *YZSlider;
    coSliderMenuItem *ZXSlider;
    coSliderMenuItem *xMoveIsolSlider;
    coSliderMenuItem *yMoveIsolSlider;
    coSliderMenuItem *zMoveIsolSlider;
    coSliderMenuItem *scaleIsolSlider;

    coLabelMenuItem *scaleLabel;
    coLabelMenuItem *rotateLabel;
    coLabelMenuItem *moveLabel;
    coLabelMenuItem *isolLabel;

    coButtonMenuItem *exeButton;

    void createSubmenu();
    void deleteSubmenu();

    const char *bottomLeftParaName;
    const char *bottomRightParaName;
    const char *topLeftParaName;
    const char *topRightParaName;
    const char *scaleFactorParaName;
    const char *XYParaName;
    const char *YZParaName;
    const char *ZXParaName;
    const char *resetParaName;
    const char *moveIsolParaName;
    const char *scaleIsolParaName;

    //const char *moveDistParaName;

    void menuEvent(coMenuItem *);
    void menuReleaseEvent(coMenuItem *);

public:
    //static FamuPlugin *plugin;
    static char *currentObjectName;

    FamuPlugin();
    virtual ~FamuPlugin();
    void add(coInteractor *inter);
    void newInteractor(RenderObject *cont, coInteractor *inter);
    void remove(const char *objName);
    void preFrame();
    void postFrame();
};

#endif
