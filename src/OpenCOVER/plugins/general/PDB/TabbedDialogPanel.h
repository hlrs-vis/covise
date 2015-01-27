/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TabbedDialogPanel_H
#define TabbedDialogPanel_H

#include <OpenVRUI/coPanel.h>
#include <OpenVRUI/coButton.h>
#include <OpenVRUI/coTextButtonGeometry.h>
#include <osg/MatrixTransform>
#include <osg/Matrix>
#include <osgDB/ReadFile>
#include <osg/Texture2D>
#include <osg/Geometry>
#include <osg/Drawable>
#include <OpenVRUI/osg/OSGVruiTransformNode.h>

namespace vrui
{
class coPanelGeometry;
class vruiTransformNode;
}

using namespace vrui;

class TabbedDialogPanel : public coPanel, public coButtonActor
{
public:
    TabbedDialogPanel(coPanelGeometry *geom);
    virtual ~TabbedDialogPanel();

    // hit is called whenever the button
    // with this action is intersected
    // return ACTION_CALL_ON_MISS if you want miss to be called
    // otherwise return ACTION_DONE
    virtual int hit(vruiHit *hit);

    // miss is called once after a hit, if the button is not intersected
    // anymore
    virtual void miss();

    void resize();
    //virtual void addElement(coUIElement * element);
    //void hide(coUIElement * element);
    //void show(coUIElement * element);
    //virtual void showElement(coUIElement * element);
    virtual void removeElement(coUIElement *element);

    void setPos(float x, float y, float z = 0.0f);
    virtual float getWidth()
    {
        return myWidth * scale;
    }
    virtual float getHeight()
    {
        return myHeight * scale;
    }
    virtual float getXpos() const
    {
        return myX;
    }
    virtual float getYpos() const
    {
        return myY;
    }
    virtual float getZpos() const
    {
        return myZ;
    }
    //virtual void setHeight(float h)   { sHeight = h; }
    virtual void setWidth(float w);

    virtual void setScale(float s);
    virtual vruiTransformNode *getDCS();

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

    int addTab(std::string s, std::string name, std::string &remainder, int limit);
    int addTab(std::string s, std::string name);
    int addTextureTab(std::string loc, std::string name);
    void replaceTab(std::string s, std::string newname, int index);
    void replaceTextureTab(std::string loc, std::string newname, int index);
    std::string getTabString(int index);
    void appendTabString(std::string s, int index);
    void setFontSize(float);
    float getFontSize();
    std::string getName(int index);
    void setTabSize(float w, float h);
    int getActiveTabIndex();
    void hideTab(int index);
    void showTab(int index);

    void removeAll();

protected:
    std::string parseString(std::string s, int index, int limit = 0);
    void prepDisplay(int append, int index);
    void makeDisplay(int index);
    void buttonEvent(coButton *);
    void format();

    std::vector<bool> _tabsvis;
    std::vector<int> _tabtype;
    std::vector<std::string> _text, _display, _name;
    std::vector<coToggleButton *> _buttons;
    std::vector<std::list<std::pair<std::string, float> > > _words, _wordsraw;
    std::vector<std::vector<std::list<std::pair<std::string, float> >::iterator> > _rowindex;
    std::vector<int> _lines;
    std::map<int, osg::MatrixTransform *> _nodemap;
    float _fontsize, tabWidth, tabHeight;
    int outputindex, visibletabs, texturedisplayed;

    osg::MatrixTransform *texturenode;

    coLabel *label;

    //float sHeight;
    float sWidth;
};
#endif
