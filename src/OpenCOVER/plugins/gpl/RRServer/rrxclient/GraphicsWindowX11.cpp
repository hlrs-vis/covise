/* -*-c++-*- OpenSceneGraph - Copyright (C) 1998-2006 Robert Osfield 
 *
 * This library is open source and may be redistributed and/or modified under  
 * the terms of the OpenSceneGraph Public License (OSGPL) version 0.0 or 
 * (at your option) any later version.  The full license is in LICENSE file
 * included with this distribution, and on the openscenegraph.org website.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * OpenSceneGraph Public License for more details.
*/

/* Note, elements of GraphicsWindowX11 have used Prodcer/RenderSurface_X11.cpp as both
 * a guide to use of X11/GLX and copying directly in the case of setBorder().
 * These elements are licensed under OSGPL as above, with Copyright (C) 2001-2004  Don Burns.
 */

#include <osgGA/GUIEventAdapter>

#include <map>

#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <X11/Xmd.h>
#include <X11/keysym.h>
#include <X11/cursorfont.h>

#include <X11/Xmd.h> /* For CARD16 */

#ifdef OSGVIEWER_USE_XRANDR
#include <X11/extensions/Xrandr.h>
#endif

#include <unistd.h>

#include "GraphicsWindowX11.h"

class X11KeyboardMap
{
public:
    X11KeyboardMap()
    {
#if 0
            _extendKeymap[8               ] = osgGA::GUIEventAdapter::KEY_BackSpace;
            _extendKeymap[127             ] = osgGA::GUIEventAdapter::KEY_Delete;
            _extendKeymap[27              ] = osgGA::GUIEventAdapter::KEY_Escape;
            // _extendKeymap[13              ] = osgGA::GUIEventAdapter::KEY_Enter;
#endif
        _extendedKeymap[XK_Escape] = osgGA::GUIEventAdapter::KEY_Escape;
        _extendedKeymap[XK_F1] = osgGA::GUIEventAdapter::KEY_F1;
        _extendedKeymap[XK_F2] = osgGA::GUIEventAdapter::KEY_F2;
        _extendedKeymap[XK_F3] = osgGA::GUIEventAdapter::KEY_F3;
        _extendedKeymap[XK_F4] = osgGA::GUIEventAdapter::KEY_F4;
        _extendedKeymap[XK_F5] = osgGA::GUIEventAdapter::KEY_F5;
        _extendedKeymap[XK_F6] = osgGA::GUIEventAdapter::KEY_F6;
        _extendedKeymap[XK_F7] = osgGA::GUIEventAdapter::KEY_F7;
        _extendedKeymap[XK_F8] = osgGA::GUIEventAdapter::KEY_F8;
        _extendedKeymap[XK_F9] = osgGA::GUIEventAdapter::KEY_F9;
        _extendedKeymap[XK_F10] = osgGA::GUIEventAdapter::KEY_F10;
        _extendedKeymap[XK_F11] = osgGA::GUIEventAdapter::KEY_F11;
        _extendedKeymap[XK_F12] = osgGA::GUIEventAdapter::KEY_F12;
        _extendedKeymap[XK_quoteleft] = '`';
        _extendedKeymap[XK_minus] = '-';
        _extendedKeymap[XK_equal] = '=';
        _extendedKeymap[XK_BackSpace] = osgGA::GUIEventAdapter::KEY_BackSpace;
        _extendedKeymap[XK_Tab] = osgGA::GUIEventAdapter::KEY_Tab;
        _extendedKeymap[XK_bracketleft] = '[';
        _extendedKeymap[XK_bracketright] = ']';
        _extendedKeymap[XK_backslash] = '\\';
        _extendedKeymap[XK_Caps_Lock] = osgGA::GUIEventAdapter::KEY_Caps_Lock;
        _extendedKeymap[XK_semicolon] = ';';
        _extendedKeymap[XK_apostrophe] = '\'';
        _extendedKeymap[XK_Return] = osgGA::GUIEventAdapter::KEY_Return;
        _extendedKeymap[XK_comma] = ',';
        _extendedKeymap[XK_period] = '.';
        _extendedKeymap[XK_slash] = '/';
        _extendedKeymap[XK_space] = ' ';
        _extendedKeymap[XK_Shift_L] = osgGA::GUIEventAdapter::KEY_Shift_L;
        _extendedKeymap[XK_Shift_R] = osgGA::GUIEventAdapter::KEY_Shift_R;
        _extendedKeymap[XK_Control_L] = osgGA::GUIEventAdapter::KEY_Control_L;
        _extendedKeymap[XK_Control_R] = osgGA::GUIEventAdapter::KEY_Control_R;
        _extendedKeymap[XK_Meta_L] = osgGA::GUIEventAdapter::KEY_Meta_L;
        _extendedKeymap[XK_Meta_R] = osgGA::GUIEventAdapter::KEY_Meta_R;
        _extendedKeymap[XK_Alt_L] = osgGA::GUIEventAdapter::KEY_Alt_L;
        _extendedKeymap[XK_Alt_R] = osgGA::GUIEventAdapter::KEY_Alt_R;
        _extendedKeymap[XK_Super_L] = osgGA::GUIEventAdapter::KEY_Super_L;
        _extendedKeymap[XK_Super_R] = osgGA::GUIEventAdapter::KEY_Super_R;
        _extendedKeymap[XK_Hyper_L] = osgGA::GUIEventAdapter::KEY_Hyper_L;
        _extendedKeymap[XK_Hyper_R] = osgGA::GUIEventAdapter::KEY_Hyper_R;
        _extendedKeymap[XK_Menu] = osgGA::GUIEventAdapter::KEY_Menu;
        _extendedKeymap[XK_Print] = osgGA::GUIEventAdapter::KEY_Print;
        _extendedKeymap[XK_Scroll_Lock] = osgGA::GUIEventAdapter::KEY_Scroll_Lock;
        _extendedKeymap[XK_Pause] = osgGA::GUIEventAdapter::KEY_Pause;
        _extendedKeymap[XK_Home] = osgGA::GUIEventAdapter::KEY_Home;
        _extendedKeymap[XK_Page_Up] = osgGA::GUIEventAdapter::KEY_Page_Up;
        _extendedKeymap[XK_End] = osgGA::GUIEventAdapter::KEY_End;
        _extendedKeymap[XK_Page_Down] = osgGA::GUIEventAdapter::KEY_Page_Down;
        _extendedKeymap[XK_Delete] = osgGA::GUIEventAdapter::KEY_Delete;
        _extendedKeymap[XK_Insert] = osgGA::GUIEventAdapter::KEY_Insert;
        _extendedKeymap[XK_Left] = osgGA::GUIEventAdapter::KEY_Left;
        _extendedKeymap[XK_Up] = osgGA::GUIEventAdapter::KEY_Up;
        _extendedKeymap[XK_Right] = osgGA::GUIEventAdapter::KEY_Right;
        _extendedKeymap[XK_Down] = osgGA::GUIEventAdapter::KEY_Down;
        _extendedKeymap[XK_Num_Lock] = osgGA::GUIEventAdapter::KEY_Num_Lock;
        _extendedKeymap[XK_KP_Divide] = osgGA::GUIEventAdapter::KEY_KP_Divide;
        _extendedKeymap[XK_KP_Multiply] = osgGA::GUIEventAdapter::KEY_KP_Multiply;
        _extendedKeymap[XK_KP_Subtract] = osgGA::GUIEventAdapter::KEY_KP_Subtract;
        _extendedKeymap[XK_KP_Add] = osgGA::GUIEventAdapter::KEY_KP_Add;
        _extendedKeymap[XK_KP_Home] = osgGA::GUIEventAdapter::KEY_KP_Home;
        _extendedKeymap[XK_KP_Up] = osgGA::GUIEventAdapter::KEY_KP_Up;
        _extendedKeymap[XK_KP_Page_Up] = osgGA::GUIEventAdapter::KEY_KP_Page_Up;
        _extendedKeymap[XK_KP_Left] = osgGA::GUIEventAdapter::KEY_KP_Left;
        _extendedKeymap[XK_KP_Begin] = osgGA::GUIEventAdapter::KEY_KP_Begin;
        _extendedKeymap[XK_KP_Right] = osgGA::GUIEventAdapter::KEY_KP_Right;
        _extendedKeymap[XK_KP_End] = osgGA::GUIEventAdapter::KEY_KP_End;
        _extendedKeymap[XK_KP_Down] = osgGA::GUIEventAdapter::KEY_KP_Down;
        _extendedKeymap[XK_KP_Page_Down] = osgGA::GUIEventAdapter::KEY_KP_Page_Down;
        _extendedKeymap[XK_KP_Insert] = osgGA::GUIEventAdapter::KEY_KP_Insert;
        _extendedKeymap[XK_KP_Delete] = osgGA::GUIEventAdapter::KEY_KP_Delete;
        _extendedKeymap[XK_KP_Enter] = osgGA::GUIEventAdapter::KEY_KP_Enter;

        _standardKeymap[XK_1] = '1';
        _standardKeymap[XK_2] = '2';
        _standardKeymap[XK_3] = '3';
        _standardKeymap[XK_4] = '4';
        _standardKeymap[XK_5] = '5';
        _standardKeymap[XK_6] = '6';
        _standardKeymap[XK_7] = '7';
        _standardKeymap[XK_8] = '8';
        _standardKeymap[XK_9] = '9';
        _standardKeymap[XK_0] = '0';
        _standardKeymap[XK_A] = 'A';
        _standardKeymap[XK_B] = 'B';
        _standardKeymap[XK_C] = 'C';
        _standardKeymap[XK_D] = 'D';
        _standardKeymap[XK_E] = 'E';
        _standardKeymap[XK_F] = 'F';
        _standardKeymap[XK_G] = 'G';
        _standardKeymap[XK_H] = 'H';
        _standardKeymap[XK_I] = 'I';
        _standardKeymap[XK_J] = 'J';
        _standardKeymap[XK_K] = 'K';
        _standardKeymap[XK_L] = 'L';
        _standardKeymap[XK_M] = 'M';
        _standardKeymap[XK_N] = 'N';
        _standardKeymap[XK_O] = 'O';
        _standardKeymap[XK_P] = 'P';
        _standardKeymap[XK_Q] = 'Q';
        _standardKeymap[XK_R] = 'R';
        _standardKeymap[XK_S] = 'S';
        _standardKeymap[XK_T] = 'T';
        _standardKeymap[XK_U] = 'U';
        _standardKeymap[XK_V] = 'V';
        _standardKeymap[XK_W] = 'W';
        _standardKeymap[XK_X] = 'X';
        _standardKeymap[XK_Y] = 'Y';
        _standardKeymap[XK_Z] = 'Z';
        _standardKeymap[XK_a] = 'a';
        _standardKeymap[XK_b] = 'b';
        _standardKeymap[XK_c] = 'c';
        _standardKeymap[XK_d] = 'd';
        _standardKeymap[XK_e] = 'e';
        _standardKeymap[XK_f] = 'f';
        _standardKeymap[XK_g] = 'g';
        _standardKeymap[XK_h] = 'h';
        _standardKeymap[XK_i] = 'i';
        _standardKeymap[XK_j] = 'j';
        _standardKeymap[XK_k] = 'k';
        _standardKeymap[XK_l] = 'l';
        _standardKeymap[XK_m] = 'm';
        _standardKeymap[XK_n] = 'n';
        _standardKeymap[XK_o] = 'o';
        _standardKeymap[XK_p] = 'p';
        _standardKeymap[XK_q] = 'q';
        _standardKeymap[XK_r] = 'r';
        _standardKeymap[XK_s] = 's';
        _standardKeymap[XK_t] = 't';
        _standardKeymap[XK_u] = 'u';
        _standardKeymap[XK_v] = 'v';
        _standardKeymap[XK_w] = 'w';
        _standardKeymap[XK_x] = 'x';
        _standardKeymap[XK_y] = 'y';
        _standardKeymap[XK_z] = 'z';
    }

    ~X11KeyboardMap() {}

    int remapKey(int key)
    {
        KeyMap::iterator itr = _extendedKeymap.find(key);
        if (itr != _extendedKeymap.end())
            return itr->second;

        itr = _standardKeymap.find(key);
        if (itr != _standardKeymap.end())
            return itr->second;

        return key;
    }

    bool remapExtendedKey(int &key)
    {
        KeyMap::iterator itr = _extendedKeymap.find(key);
        if (itr != _extendedKeymap.end())
        {
            key = itr->second;
            return true;
        }
        else
            return false;
    }

protected:
    typedef std::map<int, int> KeyMap;
    KeyMap _extendedKeymap;
    KeyMap _standardKeymap;
};

bool remapExtendedX11Key(int &key)
{
    static X11KeyboardMap s_x11KeyboardMap;
    return s_x11KeyboardMap.remapExtendedKey(key);
}

// Functions to handle key maps of type char[32] as contained in
// an XKeymapEvent or returned by XQueryKeymap().
static inline bool keyMapGetKey(const char *map, unsigned int key)
{
    return (map[(key & 0xff) / 8] & (1 << (key & 7))) != 0;
}

static inline void keyMapSetKey(char *map, unsigned int key)
{
    map[(key & 0xff) / 8] |= (1 << (key & 7));
}

static inline void keyMapClearKey(char *map, unsigned int key)
{
    map[(key & 0xff) / 8] &= ~(1 << (key & 7));
}

#if 0
void GraphicsWindowX11::adaptKey(XKeyEvent& keyevent, int& keySymbol)
{
    unsigned char buffer_return[32];
    int bytes_buffer = 32;
    KeySym keysym_return;

    int numChars = XLookupString(&keyevent, reinterpret_cast<char*>(buffer_return), bytes_buffer, &keysym_return, NULL);
    keySymbol = keysym_return;
    if (!remapExtendedX11Key(keySymbol) && (numChars==1))
    {
        keySymbol = buffer_return[0];
    }
}


void GraphicsWindowX11::rescanModifierMapping()
{
    XModifierKeymap *mkm = XGetModifierMapping(_eventDisplay);
    KeyCode *m = mkm->modifiermap;
    KeyCode numlock = XKeysymToKeycode(_eventDisplay, XK_Num_Lock);
    _numLockMask = 0;
    for (int i = 0; i < mkm->max_keypermod * 8; i++, m++)
    {
        if (*m == numlock)
        {
            _numLockMask = 1 << (i / mkm->max_keypermod);
            break;
        }
    }
    XFree(mkm->modifiermap);
    XFree(mkm);
}

void GraphicsWindowX11::flushKeyEvents()
{
    XEvent e;
    while (XCheckMaskEvent(_eventDisplay, KeyPressMask|KeyReleaseMask, &e))
        continue;
}

// Returns char[32] keymap with bits for every modifier key set.
void GraphicsWindowX11::getModifierMap(char* keymap) const
{
    memset(keymap, 0, 32);
    XModifierKeymap *mkm = XGetModifierMapping(_eventDisplay);
    KeyCode *m = mkm->modifiermap;
    for (int i = 0; i < mkm->max_keypermod * 8; i++, m++)
    {
        if (*m) keyMapSetKey(keymap, *m);
    }
    XFree(mkm->modifiermap);
    XFree(mkm);
}

int GraphicsWindowX11::getModifierMask() const
{
    int mask = 0;
    XModifierKeymap *mkm = XGetModifierMapping(_eventDisplay);
    for (int i = 0; i < mkm->max_keypermod * 8; i++)
    {
        unsigned int key = mkm->modifiermap[i];
        if (key && keyMapGetKey(_keyMap, key))
        {
            mask |= 1 << (i / mkm->max_keypermod);
        }
    }
    XFree(mkm->modifiermap);
    XFree(mkm);
    return mask;
}


extern "C" 
{

typedef int (*X11ErrorHandler)(Display*, XErrorEvent*);

int X11ErrorHandling(Display* display, XErrorEvent* event)
{
    OSG_NOTIFY(osg::NOTICE)<<"Got an X11ErrorHandling call display="<<display<<" event="<<event<<std::endl;

    char buffer[256];
    XGetErrorText( display, event->error_code, buffer, 256);

    OSG_NOTIFY(osg::NOTICE) << buffer << std::endl;
    OSG_NOTIFY(osg::NOTICE) << "Major opcode: " << (int)event->request_code << std::endl;
    OSG_NOTIFY(osg::NOTICE) << "Minor opcode: " << (int)event->minor_code << std::endl;
    OSG_NOTIFY(osg::NOTICE) << "Error code: " << (int)event->error_code << std::endl;
    OSG_NOTIFY(osg::NOTICE) << "Request serial: " << event->serial << std::endl;
    OSG_NOTIFY(osg::NOTICE) << "Current serial: " << NextRequest( display ) - 1 << std::endl;

    switch( event->error_code )
    {
        case BadValue:
            OSG_NOTIFY(osg::NOTICE) << "  Value: " << event->resourceid << std::endl;
            break;

        case BadAtom:
            OSG_NOTIFY(osg::NOTICE) << "  AtomID: " << event->resourceid << std::endl;
            break;

        default:
            OSG_NOTIFY(osg::NOTICE) << "  ResourceID: " << event->resourceid << std::endl;
            break;
    }
    return 0;
}

}
#endif
