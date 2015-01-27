/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_TABLET_UI_H
#define CO_TABLET_UI_H

//#include <OpenThreads/Thread>
//#include <OpenThreads/Mutex>
#include "coTypes.h"
#include "coDLList.h"
#include "covise_connect.h"
#include "covise_socket.h"

#include <queue>
#ifndef WIN32
#include <stdint.h>
#endif

#define THREAD_NOTHING_TO_DO 0

class coTabletUI;
class coTUIElement;
class ClientConnection;
class TokenBuffer;
class Host;
class TextureThread;
class coTUITabFolder;
class coTUITab;
class coTUIFrame;

namespace osg
{
class Node;
};

/// Action listener for events triggered by any coTUIElement.
class coTUIListener
{
public:
    /** Action listener for events triggered by coTUIElement.
	@param tUIItem pointer to element item which triggered this event
	*/
    virtual ~coTUIListener()
    {
    }
    virtual void tabletEvent(coTUIElement *tUIItem);
    virtual void tabletPressEvent(coTUIElement *tUIItem);
    virtual void tabletReleaseEvent(coTUIElement *tUIItem);
};

/**
* Tablet PC Userinterface Mamager.
* This class provides a connection to a Tablet PC and handles all coTUIElements.
*/
class coTabletUI
{
private:
public:
    coTabletUI();
    virtual ~coTabletUI();
    virtual void update();
    void addElement(coTUIElement *);
    void removeElement(coTUIElement *e);
    ClientConnection *conn;
    ClientConnection *textureConn;
    ClientConnection *sgConn;
    static coTabletUI *tUI;
    void send(TokenBuffer &tb);
    int getID();
    void tryConnect();
    void close();
    coTUITabFolder *mainFolder;
    static void CALLBACK coTabletUI::timerCallback(HWND hwnd, UINT uMsg, UINT_PTR idEvent, DWORD dwTime);

protected:
    coDLPtrList<coTUIElement *> elements;
    Host *serverHost;
    Host *localHost;
    int port;
    int ID;
    float timeout;
};

/**
* Base class for Tablet PC UI Elements.
*/
class coTUIElement
{
private:
public:
    coTUIElement(const char *, int pID = 1);
    coTUIElement(const char *, int pID, int type);
    virtual ~coTUIElement();
    virtual void parseMessage(TokenBuffer &tb);
    void setVal(const char *value);
    void setVal(bool value);
    void setVal(int value);
    void setVal(float value);
    void setVal(int type, int value);
    void setVal(int type, float value);
    void setVal(int type, int value, char *nodePath);
    void setVal(int type, char *nodePath, char *simPath, char *simName);
    void setVal(int type, int value, char *nodePath, char *simPath);

    int getID();
    virtual void resend();
    virtual void setPos(int, int);
    virtual void setSize(int, int);
    virtual void setLabel(const char *l);
    virtual void setEventListener(coTUIListener *);
    virtual coTUIListener *getMenuListener();
    const char *getName()
    {
        return name;
    };
    void createSimple(int type);

protected:
    int parentID;
    char *name; ///< name of this element
    char *label; ///< label of this element
    int ID; ///< unique ID
    int xs, ys, xp, yp;
    coTUIListener *listener; ///< event listener
};

/**
* a static textField.
*/
class coTUILabel : public coTUIElement
{
private:
public:
    coTUILabel(const char *, int pID = 1);
    virtual ~coTUILabel();
    virtual void resend();

protected:
};
/**
* a push button with a Bitmap
*/
class coTUIBitmapButton : public coTUIElement
{
private:
public:
    coTUIBitmapButton(const char *, int pID = 1);
    virtual ~coTUIBitmapButton();
    virtual void resend();
    virtual void parseMessage(TokenBuffer &tb);

protected:
};
/**
* a push button.
*/
class coTUIButton : public coTUIElement
{
private:
public:
    coTUIButton(const char *, int pID = 1);
    virtual ~coTUIButton();
    virtual void resend();
    virtual void parseMessage(TokenBuffer &tb);

protected:
};

class coTUIColorTriangle : public coTUIElement
{
private:
public:
    coTUIColorTriangle(const char *, int pID = 1);
    virtual ~coTUIColorTriangle();
    virtual void resend();
    virtual void parseMessage(TokenBuffer &tb);
    virtual float getRed()
    {
        return red;
    };
    virtual float getGreen()
    {
        return green;
    };
    virtual float getBlue()
    {
        return blue;
    };
    virtual void setColor(float r, float g, float b);

protected:
    float red;
    float green;
    float blue;
};

class coTUIColorTab : public coTUIElement
{
private:
public:
    coTUIColorTab(const char *, int pID = 1);
    virtual ~coTUIColorTab();
    virtual void resend();
    virtual void parseMessage(TokenBuffer &tb);
    virtual void setColor(float r, float g, float b, float a);
    virtual float getRed()
    {
        return red;
    }
    virtual float getGreen()
    {
        return green;
    }
    virtual float getBlue()
    {
        return blue;
    }
    virtual float getAlpha()
    {
        return alpha;
    }

protected:
    float red;
    float green;
    float blue;
    float alpha;
};

class coTUISGBrowserTab : public coTUIElement
{
private:
public:
    coTUISGBrowserTab(const char *, int pID = 1);
    virtual ~coTUISGBrowserTab();
    virtual void resend();
    virtual void parseMessage(TokenBuffer &tb);
    virtual void sendType(int type, const char *nodeType, const char *name, osg::Node *parent, osg::Node *nodeptr);
    virtual void sendEnd();
    virtual void sendCurrentNode(osg::Node *node);

protected:
    osg::Node *currentNode;
};

/**
* a NavigationElement.
*/
class coTUINav : public coTUIElement
{
private:
public:
    coTUINav(const char *, int pID = 1);
    virtual ~coTUINav();
    virtual void resend();
    virtual void parseMessage(TokenBuffer &tb);
    bool down;
    int x;
    int y;

protected:
};
/**
 * a Splitter.
 */
class coTUISplitter : public coTUIElement
{
private:
public:
    enum orientations
    {
        Horizontal = 0x1,
        Vertical = 0x2
    };

    coTUISplitter(const char *, int pID = 1);
    virtual ~coTUISplitter();
    virtual void resend();
    virtual void parseMessage(TokenBuffer &tb);
    virtual void setShape(int s);
    virtual void setStyle(int t);
    virtual void setOrientation(int or );

protected:
    int shape;
    int style;
    int orientation;
};
/**
 * a Frame.
 */
class coTUIFrame : public coTUIElement
{
private:
public:
    enum styles
    {
        Plain = 0x0010,
        Raised = 0x0020,
        Sunken = 0x0030
    };
    enum shapes
    {
        NoFrame = 0x0000,
        Box = 0x0001,
        Panel = 0x0002,
        WinPanel = 0x0003,
        HLine = 0x0004,
        VLine = 0x0005,
        StyledPanel = 0x0006
    };

    coTUIFrame(const char *, int pID = 1);
    virtual ~coTUIFrame();
    virtual void resend();
    virtual void setShape(int s); /* set shape first */
    virtual void setStyle(int t);
    virtual void parseMessage(TokenBuffer &tb);

protected:
    int style;
    int shape;
};

/**
* a tab.
*/
class coTUITab : public coTUIElement
{
private:
public:
    coTUITab(const char *, int pID = 1);
    virtual ~coTUITab();
    virtual void resend();
    virtual void parseMessage(TokenBuffer &tb);

protected:
};
/**
* a tab folder.
*/
class coTUITabFolder : public coTUIElement
{
private:
public:
    coTUITabFolder(const char *, int pID = 1);
    virtual ~coTUITabFolder();
    virtual void resend();
    virtual void parseMessage(TokenBuffer &tb);

protected:
};
/**
* a toggle button.
*/
class coTUIToggleButton : public coTUIElement
{
private:
public:
    coTUIToggleButton(const char *, int pID = 1, bool state = false);
    virtual ~coTUIToggleButton();
    virtual void resend();
    virtual void setState(bool s);
    virtual bool getState();
    virtual void parseMessage(TokenBuffer &tb);

protected:
    bool state;
};
/**
* a toggleBitmapButton.
*/
class coTUIToggleBitmapButton : public coTUIElement
{
private:
public:
    coTUIToggleBitmapButton(const char *, const char *, int pID = 1, bool state = false);
    virtual ~coTUIToggleBitmapButton();
    virtual void resend();
    virtual void setState(bool s);
    virtual bool getState();
    virtual void parseMessage(TokenBuffer &tb);

protected:
    bool state;
    char *bmpUp;
    char *bmpDown;
};
/**
* a messageBox.
*/
class coTUIMessageBox : public coTUIElement
{
private:
public:
    coTUIMessageBox(const char *, int pID = 1);
    virtual ~coTUIMessageBox();
    virtual void resend();

protected:
};
/**
* a ProgressBar.
*/
class coTUIProgressBar : public coTUIElement
{
private:
public:
    coTUIProgressBar(const char *, int pID = 1);
    virtual ~coTUIProgressBar();
    virtual void resend();
    virtual void setValue(int newV);
    virtual void setMax(int maxV);

protected:
    int actValue;
    int maxValue;
};
/**
* a slider.
*/
class coTUIFloatSlider : public coTUIElement
{
private:
public:
    enum Orientation
    {
        Horizontal = 1,
        Vertical = 0
    };

    coTUIFloatSlider(const char *, int pID = 1, bool state = true);
    virtual ~coTUIFloatSlider();
    virtual void resend();
    virtual void setValue(float newV);
    virtual void setTicks(int t);
    virtual void setOrientation(bool);
    virtual void setMin(float minV);
    virtual void setMax(float maxV);
    virtual void setRange(float minV, float maxV);
    virtual float getValue()
    {
        return actValue;
    };
    virtual void parseMessage(TokenBuffer &tb);

protected:
    float actValue;
    float minValue;
    float maxValue;
    int ticks;
    bool orientation;
};
/**
* a slider.
*/
class coTUISlider : public coTUIElement
{
private:
public:
    enum Orientation
    {
        HORIZONTAL = 1,
        VERTICAL = 0
    };
    coTUISlider(const char *, int pID = 1, bool state = true);
    virtual ~coTUISlider();
    virtual void resend();
    virtual void setValue(int newV);
    virtual void setOrientation(bool o);
    virtual void setTicks(int t);
    virtual void setMin(int minV);
    virtual void setMax(int maxV);
    virtual void setRange(int minV, int maxV);
    virtual int getValue()
    {
        return actValue;
    };
    virtual void parseMessage(TokenBuffer &tb);

protected:
    int actValue;
    int minValue;
    int maxValue;
    int ticks;
    bool orientation;
};
/**
* a spinEditField.
*/
class coTUISpinEditfield : public coTUIElement
{
private:
public:
    coTUISpinEditfield(const char *, int pID = 1);
    virtual ~coTUISpinEditfield();
    virtual void resend();
    virtual void setPosition(int newV);
    virtual void setMin(int minV);
    virtual void setMax(int maxV);
    virtual void setStep(int s);
    virtual int getValue()
    {
        return actValue;
    };
    virtual void parseMessage(TokenBuffer &tb);

protected:
    int actValue;
    int minValue;
    int maxValue;
    int step;
};
/**
* a spinEditField with text.
*/
class coTUITextSpinEditField : public coTUIElement
{
private:
public:
    coTUITextSpinEditField(const char *, int pID = 1);
    virtual ~coTUITextSpinEditField();
    virtual void resend();
    virtual void setMin(int minV);
    virtual void setMax(int maxV);
    virtual void setStep(int s);
    virtual void setText(const char *text);
    virtual const char *getText()
    {
        return text;
    }
    virtual void parseMessage(TokenBuffer &tb);

protected:
    char *text;
    int minValue;
    int maxValue;
    int step;
};
/**
* a editField.
*/
class coTUIEditField : public coTUIElement
{
private:
public:
    coTUIEditField(const char *, int pID = 1);
    virtual ~coTUIEditField();
    virtual void resend();
    virtual void setText(const char *t);
    virtual void setImmediate(bool);
    virtual const char *getText();
    virtual void parseMessage(TokenBuffer &tb);
    virtual void setPasswordMode(bool b);
    virtual void setIPAddressMode(bool b);

protected:
    char *text;
    bool immediate;
};
/**
* a editIntField = EditField fuer Integer
*/
class coTUIEditIntField : public coTUIElement
{
private:
public:
    coTUIEditIntField(const char *, int pID = 1, int def = 0);
    virtual ~coTUIEditIntField();
    virtual void setImmediate(bool);
    virtual void resend();
    virtual void setValue(int val);
    virtual void setMin(int min);
    virtual void setMax(int max);
    virtual int getValue()
    {
        return value;
    }
    virtual const char *getText();
    virtual void parseMessage(TokenBuffer &tb);

protected:
    int value;
    int min;
    int max;
    bool immediate;
};
/**
* a editfloatfield = EditField fuer Kommazahlen
*/
class coTUIEditFloatField : public coTUIElement
{
private:
public:
    coTUIEditFloatField(const char *, int pID = 1, float def = 0);
    virtual ~coTUIEditFloatField();
    virtual void setImmediate(bool);
    virtual void resend();
    virtual void setValue(float val);
    virtual float getValue()
    {
        return value;
    };
    virtual void parseMessage(TokenBuffer &tb);

protected:
    float value;
    bool immediate;
};
/**
* a comboBox.
*/
class coTUIComboBox : public coTUIElement
{
private:
public:
    coTUIComboBox(const char *, int pID = 1);
    virtual ~coTUIComboBox();
    virtual void resend();
    virtual void addEntry(const char *t);
    virtual void delEntry(const char *t);
    virtual int getSelectedEntry();
    virtual void setSelectedEntry(int e);
    virtual void setSelectedText(const char *t);
    virtual const char *getSelectedText();
    virtual void parseMessage(TokenBuffer &tb);

protected:
    char *text;
    int selection;
    coDLList<char *> elements;
    coDLListIter<char *> iter;
};
/**
* a listBox.
*/
class coTUIListBox : public coTUIElement
{
private:
public:
    coTUIListBox(const char *, int pID = 1);
    virtual ~coTUIListBox();
    virtual void resend();
    virtual void addEntry(const char *t);
    virtual void delEntry(const char *t);
    virtual int getSelectedEntry();
    virtual void setSelectedEntry(int e);
    virtual void setSelectedText(const char *t);
    virtual const char *getSelectedText();
    virtual void parseMessage(TokenBuffer &tb);

protected:
    char *text;
    int selection;
    coDLList<char *> elements;
    coDLListIter<char *> iter;
};
class MapData
{
public:
    MapData(const char *name, float ox, float oy, float xSize, float ySize, float height);
    virtual ~MapData();
    char *name;
    float ox, oy, xSize, ySize, height;
};
/**
* a Map Widget
*/
class coTUIMap : public coTUIElement
{
private:
public:
    coTUIMap(const char *, int pID = 1);
    virtual ~coTUIMap();
    virtual void addMap(const char *name, float ox, float oy, float xSize, float ySize, float height);
    virtual void resend();
    virtual void parseMessage(TokenBuffer &tb);

    float angle;
    float xPos;
    float yPos;
    float height;
    int mapNum;

protected:
    coDLList<MapData *> maps;
    coDLListIter<MapData *> iter;
};

/**
 * PopUp Window with text
 */
class coTUIPopUp : public coTUIElement
{
private:
public:
    coTUIPopUp(const char *, int pID = 1);
    virtual ~coTUIPopUp();
    virtual void resend();
    virtual void setText(const char *t);
    virtual void setImmediate(bool);
    virtual void parseMessage(TokenBuffer &tb);

protected:
    char *text;
    bool immediate;
};

#endif
