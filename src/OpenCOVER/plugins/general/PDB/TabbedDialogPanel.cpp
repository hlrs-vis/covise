/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coPanel.h>
#include <OpenVRUI/coPanelGeometry.h>
#include <OpenVRUI/coLabel.h>

#include <OpenVRUI/sginterface/vruiHit.h>
#include <OpenVRUI/sginterface/vruiIntersection.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/sginterface/vruiTransformNode.h>

#include <OpenVRUI/util/vruiLog.h>

#include <list>

#include "TabbedDialogPanel.h"

using namespace std;
using namespace osg;
#define BORDERWIDTH 0.5f
/** Constructor
   @param geom Panel geometry
*/
TabbedDialogPanel::TabbedDialogPanel(coPanelGeometry *geometry)
    : coPanel(geometry)
{
    _fontsize = 20;

    sWidth = 500;
    scale = 1.0;
    tabWidth = 0;
    tabHeight = 0;
    outputindex = 0;
    visibletabs = 0;

    texturedisplayed = 0;

    label = new coLabel();

    label->setFontSize(_fontsize);
    label->resize();

    this->addElement(label);

    label->setPos(0, 0, 1);
}

/** Destructor
 */
TabbedDialogPanel::~TabbedDialogPanel()
{
    delete label;
}

/** set the scale factor of this panel and all its children
 */
void TabbedDialogPanel::setScale(float s)
{
    scale = s;
    myChildDCS->setScale(scale, scale, scale);
}

void TabbedDialogPanel::removeElement(coUIElement *el)
{
    coUIContainer::removeElement(el);
    if (el->getDCS())
    {
        myChildDCS->removeChild(el->getDCS());
    }
}

/*void TabbedDialogPanel::addElement(coUIElement * el)
{
   coUIContainer::addElement(el);
   if(el->getDCS())
   {
      myChildDCS->addChild(el->getDCS());
   }
}


void TabbedDialogPanel::hide(coUIElement * el)
{
   if(el->getDCS())
   {
      myChildDCS->removeChild(el->getDCS());
   }
}


void TabbedDialogPanel::show(coUIElement * el)
{
   if(el->getDCS())
   {
      myChildDCS->addChild(el->getDCS());
   }
}


void TabbedDialogPanel::showElement(coUIElement * el)
{
   if(el->getDCS())
   {
      myChildDCS->addChild(el->getDCS());
   }
}
 */
void TabbedDialogPanel::resize()
{
    float maxX = -100000;
    float maxY = -100000;
    //float minX = 100000;
    //float minY = 100000;
    float minX = 0.0;
    float minY = 0.0;
    float minZ = 100000;

    float xOff, yOff, zOff = 0;

    for (list<coUIElement *>::iterator i = elements.begin(); i != elements.end(); ++i)
    {
        if (maxX < (*i)->getXpos() + (*i)->getWidth())
            maxX = (*i)->getXpos() + (*i)->getWidth();

        if (maxY < (*i)->getYpos() + (*i)->getHeight())
            maxY = (*i)->getYpos() + (*i)->getHeight();

        if (minX > (*i)->getXpos())
            minX = (*i)->getXpos();

        if (minY > (*i)->getYpos())
            minY = (*i)->getYpos();

        if (minZ > (*i)->getZpos())
            minZ = (*i)->getZpos();
    }

    //if(sHeight > 0)
    //{
    //   myHeight = sHeight + (float)(2*BORDERWIDTH);
    //}
    //else
    //{
    myHeight = (maxY - minY) + (float)(2 * BORDERWIDTH);
    //}
    xOff = minX - (float)BORDERWIDTH;
    yOff = minY - (float)BORDERWIDTH;

    if (myGeometry)
        zOff = minZ - myGeometry->getDepth();

    if (sWidth > 0)
    {
        myWidth = sWidth + (float)(2 * BORDERWIDTH);
    }
    else
    {
        myWidth = (maxX - minX) + (float)(2 * BORDERWIDTH);
    }

    //myChildDCS->setTranslation(-xOff * scale, -yOff * scale, -zOff * scale);
    myChildDCS->setTranslation(-xOff * scale, -yOff * scale, 0.0);
    myChildDCS->setScale(scale, scale, scale);

    if (myGeometry)
    {

        myPosDCS->setScale(getWidth() / myGeometry->getWidth(), getHeight() / myGeometry->getHeight(), 1.0);
    }

    if (getParent())
        getParent()->childResized();
}

void TabbedDialogPanel::setPos(float x, float y, float)
{
    resize();
    myX = x;
    myY = y;
    myDCS->setTranslation(myX, myY, myZ);
}

/**hit is called whenever the panel is intersected
 @param hitPoint point of intersection in world coordinates
 @param hit Performer hit structure to queuery other information like normal
 @return ACTION_CALL_ON_MISS if you want miss to be called
otherwise return ACTION_DONE
*/
int TabbedDialogPanel::hit(vruiHit *hit)
{

    //VRUILOG("coPanel::hit info: called")

    Result preReturn = vruiRendererInterface::the()->hit(this, hit);
    if (preReturn != ACTION_UNDEF)
        return preReturn;

    return ACTION_CALL_ON_MISS;
}

/**miss is called once after a hit, if the panel is not intersected
 anymore*/
void TabbedDialogPanel::miss()
{
    vruiRendererInterface::the()->miss(this);
}

const char *TabbedDialogPanel::getClassName() const
{
    return "TabbedDialogPanel";
}

bool TabbedDialogPanel::isOfClassName(const char *classname) const
{
    // paranoia makes us mistrust the string library and check for NULL.
    if (classname && getClassName())
    {
        // check for identity
        if (!strcmp(classname, getClassName()))
        { // we are the one
            return true;
        }
        else
        { // we are not the wanted one. Branch up to parent class
            return coPanel::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}

vruiTransformNode *TabbedDialogPanel::getDCS()
{
    return myDCS;
}

void TabbedDialogPanel::setWidth(float w)
{
    for (int i = 0; i < _display.size(); i++)
    {
        if (_tabtype[i] == 1)
        {
            continue;
        }
        float rat = w / sWidth;
        //cerr << "rat: " << rat << endl;
        _lines[i] = (int)(((float)_lines[i]) * rat);
        //cerr << "newheight: " << _lines[i] << endl;
        osg::Matrix m;
        m.makeScale(w, (float)_lines[i], 1.0);
        _nodemap[i]->setMatrix(m);
    }

    sWidth = w;
    for (int i = 0; i < _display.size(); i++)
    {
        if (_tabtype[i] == 2)
        {
            continue;
        }
        prepDisplay(0, i);
        makeDisplay(i);
    }

    format();
}

void TabbedDialogPanel::setTabSize(float w, float h)
{
    if (w > 0)
    {
        tabWidth = w;
    }

    if (h > 0)
    {
        tabHeight = h;
    }

    for (int i = 0; i < _buttons.size(); i++)
    {
        this->hide(_buttons[i]);
        delete _buttons[i];
        _buttons[i] = new coToggleButton(new coTextButtonGeometry(tabWidth, tabHeight, _name[i]), this);
        this->addElement(_buttons[i]);
    }
    format();
}

void TabbedDialogPanel::hideTab(int index)
{
    if (index >= _tabsvis.size() || index < 0)
    {
        return;
    }
    if (_tabsvis[index] == true)
    {
        visibletabs--;
    }
    _tabsvis[index] = false;

    this->hide(_buttons[index]);

    if (visibletabs == 0)
    {
        outputindex = 0;
    }
    else
    {
        for (int i = 0; i < _tabsvis.size(); i++)
        {
            if (_tabsvis[i] == true)
            {
                outputindex = i;
                break;
            }
        }
    }
    format();
}

void TabbedDialogPanel::showTab(int index)
{
    if (index >= _tabsvis.size() || index < 0)
    {
        return;
    }
    if (_tabsvis[index] == false)
    {
        visibletabs++;
    }
    _tabsvis[index] = true;
    this->show(_buttons[index]);
    if (_tabsvis[outputindex] == false)
    {
        for (int i = 0; i < _tabsvis.size(); i++)
        {
            if (_tabsvis[i] == true)
            {
                outputindex = i;
                break;
            }
        }
    }
    format();
}

void TabbedDialogPanel::removeAll()
{
    outputindex = 0;
    visibletabs = 0;
    _text.clear();
    _display.clear();
    _name.clear();
    _words.clear();
    _wordsraw.clear();
    _rowindex.clear();
    _lines.clear();
    _tabsvis.clear();
    _tabtype.clear();

    for (map<int, osg::MatrixTransform *>::iterator i = _nodemap.begin(); i != _nodemap.end(); i++)
    {
        i->second->unref();
    }

    _nodemap.clear();

    for (int i = 0; i < _buttons.size(); i++)
    {
        this->hide(_buttons[i]);
        delete _buttons[i];
    }

    _buttons.clear();
    format();
}

int TabbedDialogPanel::getActiveTabIndex()
{
    if (_display.size() == 0 || visibletabs == 0)
    {
        return -1;
    }

    return outputindex;
}

int TabbedDialogPanel::addTab(string s, string name, string &remainder, int limit)
{
    int strindex = _text.size();
    _text.push_back(s);
    _name.push_back(name);
    _tabtype.push_back(1);
    if (tabWidth <= 0 || tabHeight <= 0)
    {
        tabWidth = _fontsize * 5.5;
        tabHeight = 2.0 * _fontsize * 0.8f;
    }

    _buttons.push_back(new coToggleButton(new coTextButtonGeometry(tabWidth, tabHeight, name), this));

    this->addElement(_buttons[strindex]);

    _words.push_back(std::list<std::pair<std::string, float> >());
    _wordsraw.push_back(std::list<std::pair<std::string, float> >());

    _display.push_back(std::string());

    _rowindex.push_back(std::vector<std::list<std::pair<std::string, float> >::iterator>());
    _lines.push_back(0);

    visibletabs++;
    _tabsvis.push_back(true);

    remainder = parseString(s, strindex, limit);

    prepDisplay(0, strindex);
    makeDisplay(strindex);
    //cerr << "Printing display:\n" << _display[strindex] << endl;
    format();

    return strindex;
}

int TabbedDialogPanel::addTab(std::string s, std::string name)
{

    int strindex = _text.size();
    _text.push_back(s);
    _name.push_back(name);
    _tabtype.push_back(1);
    if (tabWidth <= 0 || tabHeight <= 0)
    {
        tabWidth = _fontsize * 5.5;
        tabHeight = 2.0 * _fontsize * 0.8f;
    }

    _buttons.push_back(new coToggleButton(new coTextButtonGeometry(tabWidth, tabHeight, name), this));

    this->addElement(_buttons[strindex]);

    _words.push_back(std::list<std::pair<std::string, float> >());
    _wordsraw.push_back(std::list<std::pair<std::string, float> >());

    _display.push_back(std::string());

    _rowindex.push_back(std::vector<std::list<std::pair<std::string, float> >::iterator>());
    _lines.push_back(0);

    visibletabs++;
    _tabsvis.push_back(true);

    parseString(s, strindex);

    prepDisplay(0, strindex);
    makeDisplay(strindex);
    //cerr << "Printing display:\n" << _display[strindex] << endl;
    format();

    return strindex;
}

int TabbedDialogPanel::addTextureTab(string loc, string name)
{
    int strindex = _text.size();

    Image *image = osgDB::readImageFile(loc);
    StateSet *stateset;
    if (image == NULL)
    {
        return -1;
    }

    float twidth = image->s();
    float theight = image->t();

    //theight = theight *(sWidth / twidth);
    theight = (theight / twidth) * sWidth;

    //cerr << "twidth: " << twidth << " theight: " << theight << endl;

    osg::Geometry *geom = new osg::Geometry;

    osg::Vec3Array *vertices = new osg::Vec3Array;

    vertices->push_back(osg::Vec3(0.0, 1.0, 0.0)); //top-left corner
    vertices->push_back(osg::Vec3(0.0, 0.0, 0.0)); //bottom-left
    vertices->push_back(osg::Vec3(1.0, 0.0, 0.0)); //bottom-right
    vertices->push_back(osg::Vec3(1.0, 1.0, 0.0)); //top-right

    geom->setVertexArray(vertices);

    osg::Vec2Array *texcoords = new osg::Vec2Array;
    texcoords->push_back(osg::Vec2(0, 1));
    texcoords->push_back(osg::Vec2(0, 0));
    texcoords->push_back(osg::Vec2(1, 0));
    texcoords->push_back(osg::Vec2(1, 1));
    geom->setTexCoordArray(0, texcoords);

    osg::Vec3Array *normals = new osg::Vec3Array;
    normals->push_back(osg::Vec3(0.0f, 0.0f, 1.0f));
    geom->setNormalArray(normals);
    geom->setNormalBinding(osg::Geometry::BIND_OVERALL);

    osg::Vec4Array *colors = new osg::Vec4Array;
    colors->push_back(osg::Vec4(1.0f, 1.0, 1.0f, 1.0f));
    geom->setColorArray(colors);
    geom->setColorBinding(osg::Geometry::BIND_OVERALL);

    geom->addPrimitiveSet(new osg::DrawArrays(GL_QUADS, 0, 4));

    stateset = geom->getOrCreateStateSet();

    stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    stateset->setMode(GL_BLEND, osg::StateAttribute::ON);
    stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

    osg::Texture2D *texture;

    texture = new osg::Texture2D;
    texture->setImage(image);
    stateset->setTextureAttributeAndModes(0, texture, osg::StateAttribute::ON);

    Geode *g = new Geode();

    g->addDrawable(geom);

    Matrix mat;
    mat.makeTranslate(0, 0, 1.0);

    MatrixTransform *mt = new MatrixTransform();
    mt->setMatrix(mat);
    mt->addChild(g);
    MatrixTransform *scalemt = new MatrixTransform();
    mat.makeScale(sWidth, (float)(int)theight, 1.0);
    scalemt->setMatrix(mat);
    scalemt->addChild(mt);

    _text.push_back(loc);
    _name.push_back(name);
    _tabtype.push_back(2);
    if (tabWidth <= 0 || tabHeight <= 0)
    {
        tabWidth = _fontsize * 5.5;
        tabHeight = 2.0 * _fontsize * 0.8f;
    }

    _buttons.push_back(new coToggleButton(new coTextButtonGeometry(tabWidth, tabHeight, name), this));

    this->addElement(_buttons[strindex]);

    _words.push_back(std::list<std::pair<std::string, float> >());
    _wordsraw.push_back(std::list<std::pair<std::string, float> >());

    // The space makes all the difference
    _display.push_back(std::string(" "));

    _rowindex.push_back(std::vector<std::list<std::pair<std::string, float> >::iterator>());
    _lines.push_back((int)theight);

    visibletabs++;
    _tabsvis.push_back(true);

    _nodemap[strindex] = scalemt;
    _nodemap[strindex]->ref();
    format();
    return strindex;
}

void TabbedDialogPanel::replaceTextureTab(std::string loc, std::string newname, int index)
{
    if (index >= _display.size() || index < 0)
    {
        return;
    }

    if (_tabtype[index] != 2)
    {
        return;
    }

    Image *image = osgDB::readImageFile(loc);
    StateSet *stateset;
    if (image == NULL)
    {
        return;
    }

    if (outputindex == index)
    {
        dynamic_cast<MatrixTransform *>(dynamic_cast<OSGVruiTransformNode *>(myChildDCS)->getNodePtr())->removeChild(texturenode);
    }

    _nodemap[index]->unref();

    float twidth = image->s();
    float theight = image->t();

    //theight = theight *(sWidth / twidth);
    theight = (theight / twidth) * sWidth;

    //cerr << "twidth: " << twidth << " theight: " << theight << endl;

    osg::Geometry *geom = new osg::Geometry;

    osg::Vec3Array *vertices = new osg::Vec3Array;

    vertices->push_back(osg::Vec3(0.0, 1.0, 0.0)); //top-left corner
    vertices->push_back(osg::Vec3(0.0, 0.0, 0.0)); //bottom-left
    vertices->push_back(osg::Vec3(1.0, 0.0, 0.0)); //bottom-right
    vertices->push_back(osg::Vec3(1.0, 1.0, 0.0)); //top-right

    geom->setVertexArray(vertices);

    osg::Vec2Array *texcoords = new osg::Vec2Array;
    texcoords->push_back(osg::Vec2(0, 1));
    texcoords->push_back(osg::Vec2(0, 0));
    texcoords->push_back(osg::Vec2(1, 0));
    texcoords->push_back(osg::Vec2(1, 1));
    geom->setTexCoordArray(0, texcoords);

    osg::Vec3Array *normals = new osg::Vec3Array;
    normals->push_back(osg::Vec3(0.0f, 0.0f, 1.0f));
    geom->setNormalArray(normals);
    geom->setNormalBinding(osg::Geometry::BIND_OVERALL);

    osg::Vec4Array *colors = new osg::Vec4Array;
    colors->push_back(osg::Vec4(1.0f, 1.0, 1.0f, 1.0f));
    geom->setColorArray(colors);
    geom->setColorBinding(osg::Geometry::BIND_OVERALL);

    geom->addPrimitiveSet(new osg::DrawArrays(GL_QUADS, 0, 4));

    stateset = geom->getOrCreateStateSet();

    stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    stateset->setMode(GL_BLEND, osg::StateAttribute::ON);
    stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

    osg::Texture2D *texture;

    texture = new osg::Texture2D;
    texture->setImage(image);
    stateset->setTextureAttributeAndModes(0, texture, osg::StateAttribute::ON);

    Geode *g = new Geode();

    g->addDrawable(geom);

    Matrix mat;
    mat.makeTranslate(0, 0, 1.0);

    MatrixTransform *mt = new MatrixTransform();
    mt->setMatrix(mat);
    mt->addChild(g);
    MatrixTransform *scalemt = new MatrixTransform();
    mat.makeScale(sWidth, (float)(int)theight, 1.0);
    scalemt->setMatrix(mat);
    scalemt->addChild(mt);

    _text[index] = loc;
    _name[index] = newname;
    if (tabWidth <= 0 || tabHeight <= 0)
    {
        tabWidth = _fontsize * 5.5;
        tabHeight = 2.0 * _fontsize * 0.8f;
    }

    this->removeElement(_buttons[index]);
    delete _buttons[index];

    _buttons[index] = new coToggleButton(new coTextButtonGeometry(tabWidth, tabHeight, newname), this);

    this->addElement(_buttons[index]);

    _lines[index] = ((int)theight);

    _nodemap[index] = scalemt;
    _nodemap[index]->ref();
    format();

    if (!_tabsvis[index])
    {
        hideTab(index);
    }
}

std::string TabbedDialogPanel::getName(int index)
{
    if (index >= _display.size() || index < 0)
    {
        return string();
    }

    return _name[index];
}

void TabbedDialogPanel::replaceTab(std::string s, std::string newname, int index)
{
    if (index >= _display.size() || index < 0)
    {
        return;
    }

    _text[index] = s;
    _name[index] = newname;
    this->hide(_buttons[index]);
    delete _buttons[index];
    _buttons[index] = new coToggleButton(new coTextButtonGeometry(tabWidth, tabHeight, newname), this);
    this->addElement(_buttons[index]);
    _words[index] = std::list<std::pair<std::string, float> >();
    _wordsraw[index] = std::list<std::pair<std::string, float> >();

    _display[index] = "";
    _rowindex[index] = std::vector<std::list<std::pair<std::string, float> >::iterator>();
    _lines[index] = 0;
    _tabsvis[index] = true;

    if (_tabtype[index] == 2)
    {
        _nodemap[index]->unref();
    }

    parseString(s, index);
    prepDisplay(0, index);
    makeDisplay(index);

    format();
}

std::string TabbedDialogPanel::getTabString(int index)
{
    if (index >= _text.size() || index < 0 || _tabtype[index] == 2)
    {
        return "";
    }
    return _text[index];
}

void TabbedDialogPanel::appendTabString(std::string s, int index)
{
    if (index >= _text.size() || index < 0 || _tabtype[index] == 2)
    {
        return;
    }

    _text[index] += s;
    parseString(s, index);

    prepDisplay(1, index);
    makeDisplay(index);

    format();
}

void TabbedDialogPanel::setFontSize(float f)
{
    _fontsize = f;

    label->setFontSize(_fontsize);
    label->resize();

    for (int i = 0; i < _words.size(); i++)
    {
        if (_tabtype[i] == 2)
        {
            continue;
        }
        _words[i].clear();
        _wordsraw[i].clear();
        parseString(_text[i], i);
        prepDisplay(0, i);
        makeDisplay(i);
    }

    format();
}

float TabbedDialogPanel::getFontSize()
{
    return _fontsize;
}

string TabbedDialogPanel::parseString(std::string s, int idx, int limit)
{
    string whitespace(" \t\f\v\n\r");
    coLabel slabel;
    slabel.setFontSize(_fontsize);
    slabel.resize();
    size_t ws, index, temp;
    index = 0;
    int wordcount = 0;

    while (true)
    {
        string word;
        float size;
        int leng;
        ws = s.find_first_of(whitespace, index);
        temp = s.find_first_not_of(whitespace, index);
        if (ws == string::npos && temp == string::npos)
        {
            break;
        }

        if (ws == index)
        {
            if (ws == string::npos)
            {
                leng = s.size() - index;
            }
            else
            {
                leng = temp - index;
            }
            word = s.substr(index, leng);
            string subword;
            size_t wsindex, nlindex;
            int wsleng;
            wsindex = 0;
            while (true)
            {
                nlindex = word.find("\n", wsindex);
                if (nlindex != string::npos)
                {
                    wsleng = nlindex - wsindex;
                    if (wsleng > 0)
                    {
                        subword = word.substr(wsindex, wsleng);
                        slabel.setString(subword);
                        size = slabel.getWidth();

                        _words[idx].push_back(pair<string, float>(subword, size));
                        _wordsraw[idx].push_back(pair<string, float>(subword, size));
                        wsindex += wsleng;
                    }
                    _words[idx].push_back(pair<string, float>("\n", 0));
                    _wordsraw[idx].push_back(pair<string, float>("\n", 0));
                    wsindex++;
                    continue;
                }
                wsleng = word.size() - wsindex;
                if (wsleng > 0)
                {
                    subword = word.substr(wsindex, wsleng);
                    slabel.setString(subword);
                    size = slabel.getWidth();
                    _words[idx].push_back(pair<string, float>(subword, size));
                    _wordsraw[idx].push_back(pair<string, float>(subword, size));
                }
                break;
            }
            index += leng;
        }
        else
        {
            if (ws == string::npos)
            {
                leng = s.size() - index;
            }
            else
            {
                leng = ws - index;
            }
            word = s.substr(index, leng);

            slabel.setString(word);
            size = slabel.getWidth();

            _words[idx].push_back(pair<string, float>(word, size));
            _wordsraw[idx].push_back(pair<string, float>(word, size));
            wordcount++;
            index += leng;
        }
        if (limit != 0 && wordcount == limit)
        {
            return s.substr(index);
        }
    }
    return string();
}

void TabbedDialogPanel::prepDisplay(int append, int index)
{
    string thing1 = "", thing2 = "";

    float rowpos = 0;

    std::list<std::pair<std::string, float> >::iterator it;

    if (!append)
    {
        _words[index] = std::list<std::pair<std::string, float> >(_wordsraw[index]);
        _rowindex[index].clear();
        it = _words[index].begin();
        _rowindex[index].push_back(it);
    }
    else
    {
        if (_rowindex[index].size() > 0)
        {
            std::vector<std::list<std::pair<std::string, float> >::iterator>::iterator tempit = _rowindex[index].end();
            tempit--;
            it = *tempit;
        }
        else
        {
            it = _words[index].begin();
        }
    }

    coLabel slabel;
    slabel.setFontSize(_fontsize);
    slabel.resize();

    while (true)
    {
        if (it == _words[index].end())
        {
            break;
        }
        if (it->first != "\n")
        {
            if (it->second > sWidth)
            {
                slabel.setString(thing1);
                rowpos = slabel.getWidth();
                float splitsize = -1000;
                string split1, split2;
                int splitindex = 1;
                //cerr << "rowpos: " << rowpos << endl;
                while (splitsize < sWidth - rowpos)
                {
                    split1 = it->first.substr(0, splitindex);
                    slabel.setString(split1);
                    splitsize = slabel.getWidth();
                    //cerr << "splitsize: " << splitsize << endl;
                    splitindex++;
                }
                splitindex = splitindex - 2;
                split1 = it->first.substr(0, splitindex);
                split2 = it->first.substr(splitindex, it->first.size() - splitindex);
                if (split1 == "")
                {
                    //cerr << "moving to next line: " << thing2 << endl;
                    it = _words[index].insert(it, pair<string, float>("\n", 0.0));
                    it++;
                    thing1.clear();
                    thing2.clear();
                    continue;
                }
                else
                {
                    slabel.setString(split2);

                    it = _words[index].insert(it, pair<string, float>(split2, slabel.getWidth()));
                    it = _words[index].insert(it, pair<string, float>(split1, 0.0));
                    if (rowpos == 0.0)
                    {
                        _rowindex[index].push_back(it);
                    }
                    it++;
                    it++;
                    it = _words[index].erase(it);
                    it--;
                    thing1 += split1;
                    thing2 += split1;
                    continue;
                }
            }
            thing1 += it->first;
            slabel.setString(thing1);
            rowpos = slabel.getWidth();
            if (rowpos < sWidth)
            {
                thing2 += it->first;
                it++;
            }
            else
            {
                //cerr << "moving to next line: " << thing2 << endl;
                it = _words[index].insert(it, pair<string, float>("\n", 0.0));
                it++;
                _rowindex[index].push_back(it);
                thing1 = it->first;
                thing2 = it->first;
                it++;
                //_lines++;
                continue;
            }
        }
        else
        {
            //it = _words.insert(it, pair<string, float>("\n", 0.0));
            thing1.clear();
            thing2.clear();
            rowpos = 0;
            it++;
            _rowindex[index].push_back(it);
        }
    }
}

void TabbedDialogPanel::makeDisplay(int index)
{
    _display[index] = "";
    _lines[index] = 0;
    for (std::list<std::pair<std::string, float> >::iterator it = _words[index].begin(); it != _words[index].end(); it++)
    {
        _display[index] += it->first;
        if (it->first == "\n")
        {
            _lines[index]++;
        }
    }
    //_display[index] += " ";
}

void TabbedDialogPanel::buttonEvent(coButton *b)
{
    for (int i = 0; i < _buttons.size(); i++)
    {
        if (b == _buttons[i])
        {
            outputindex = i;
            format();
            break;
        }
    }
}

void TabbedDialogPanel::format()
{
    if (texturedisplayed)
    {
        dynamic_cast<MatrixTransform *>(dynamic_cast<OSGVruiTransformNode *>(myChildDCS)->getNodePtr())->removeChild(texturenode);
    }

    if (visibletabs > 0 && _tabtype[outputindex] == 2)
    {
        texturenode = _nodemap[outputindex];

        dynamic_cast<MatrixTransform *>(dynamic_cast<OSGVruiTransformNode *>(myChildDCS)->getNodePtr())->addChild(texturenode);
        //cerr << "theight: " <<  _lines[outputindex] << endl;
        //cerr << "x: " << texturenode->getMatrix().getScale().x() << " y: " << texturenode->getMatrix().getScale().y() << " z: " << texturenode->getMatrix().getScale().z() << endl;
        texturedisplayed = 1;
    }
    else
    {
        texturedisplayed = 0;
    }

    int tabsPerRow = (int)(sWidth / tabWidth);
    if (tabsPerRow <= 0)
    {
        tabsPerRow = 1;
    }

    int rows = visibletabs / tabsPerRow;
    if (rows * tabsPerRow < visibletabs)
    {
        rows++;
    }

    if (visibletabs > 0)
    {
        label->setString(_display[outputindex]);
    }
    else
    {
        label->setString(" ");
    }

    int count = 0;
    for (int j = 0; j < rows; j++)
    {
        if (count == _buttons.size())
        {
            break;
        }
        for (int i = 0; i < tabsPerRow; i++)
        {

            if (!_tabsvis[count])
            {
                _buttons[count]->setPos(0, 0, 0);
                i--;
                count++;
                if (count == _buttons.size())
                {
                    break;
                }
                continue;
            }

            if (_tabtype[outputindex] == 1)
            {
                _buttons[count]->setPos((i * (tabWidth)) + i, (_lines[outputindex] + 1) * _fontsize + (rows - j - 1) + ((rows - j - 1) * tabHeight), 0);
            }
            else
            {
                _buttons[count]->setPos((i * (tabWidth)) + i, _lines[outputindex] + (rows - j - 1) + ((rows - j - 1) * tabHeight), 0);
                //cerr << "tabh: " <<  _lines[outputindex] + (rows - j - 1) + ((rows - j - 1) * tabHeight) << endl;
            }

            if (count != outputindex)
            {
                _buttons[count]->setState(false, false);
            }
            else
            {
                _buttons[count]->setState(true, false);
            }

            count++;
            if (count == _buttons.size())
            {
                break;
            }
        }
    }
    this->resize();
}
