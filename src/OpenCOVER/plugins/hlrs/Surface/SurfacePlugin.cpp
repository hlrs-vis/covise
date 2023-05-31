/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
**                                                            (C)2012 HLRS  **
**                                                                          **
** Description: Surface Plugin (does Surface)								**
**                                                                          **
**                                                                          **
** Author:																	**
**			Jens Dehlke														**
**			U. Woessner														**
**                                                                          **
** History:																	**
** Sep-12  v2 updated to work with Multitouch Plugin						**
** xxx-08  v1	    				       		                            **
**                                                                          **
**                                                                          **
\****************************************************************************/

#include "SurfacePlugin.h"
#include "cover/coVRPluginList.h"
#include <cover/coVRConfig.h>
#include <osg/GraphicsContext>
#include <osg/io_utils>

SurfacePlugin::SurfacePlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "SurfacePlugin::SurfacePlugin\n");
    coVRTouchTable::instance()->ttInterface = this;
    multitouchPlugin = dynamic_cast<MultitouchPlugin *>(coVRPluginList::instance()->addPlugin("Multitouch"));
}

// this is called if the plugin is removed at runtime
SurfacePlugin::~SurfacePlugin()
{
    fprintf(stderr, "SurfacePlugin::~SurfacePlugin\n");
    coVRTouchTable::instance()->ttInterface = NULL;
    coVRPluginList::instance()->unload(multitouchPlugin);
    multitouchPlugin = NULL;
}

void SurfacePlugin::manipulation(MotionEvent &me)
{
    if (cover->debugLevel(4))
    {
        cout << "SurfacePlugin::manipulation" << endl;
        cout << "AngularVelocity " << me.AngularVelocity << endl;
        cout << "CumulativeExpansion " << me.CumulativeExpansion << endl;
        cout << "CumulativeRotation " << me.CumulativeRotation << endl;
        cout << "CumulativeScale " << me.CumulativeScale << endl;
        cout << "CumulativeTranslationX " << me.CumulativeTranslationX << endl;
        cout << "CumulativeTranslationY " << me.CumulativeTranslationY << endl;
        cout << "DeltaX " << me.DeltaX << endl;
        cout << "DeltaY " << me.DeltaY << endl;
        cout << "ExpansionDelta " << me.ExpansionDelta << endl;
        cout << "ExpansionVelocity " << me.ExpansionVelocity << endl;
        cout << "ManipulationOriginX " << me.ManipulationOriginX << endl;
        cout << "ManipulationOriginY " << me.ManipulationOriginY << endl;
        cout << "RotationDelta " << me.RotationDelta << endl;
        cout << "ScaleDelta " << me.ScaleDelta << endl;
        cout << "VelocityX " << me.VelocityX << endl;
        cout << "VelocityY " << me.VelocityY << endl;
        cout << "------------------------------------" << endl;
    }
}

void SurfacePlugin::tabletEvent(coTUIElement *tUIItem)
{
    (void)tUIItem;
}

void SurfacePlugin::tabletPressEvent(coTUIElement *tUIItem)
{
    (void)tUIItem;
}

// this will be called wen a key is pressed
void SurfacePlugin::key(int type, int keySym, int mod)
{
    (void)type;
    (void)keySym;
    (void)mod;
}

//! Notify that a finger has just been made active.
void SurfacePlugin::addedContact(SurfaceContact &c)
{
    // only recognize finger as input, no blobs or tags
    if (c.finger)
    {
        _fingerContacts.push_back(c);
        TouchContact con(c.CenterX, c.CenterY, c.Id);
        multitouchPlugin->addContact(con);
    }
    else
    {
        _otherContacts.push_back(c);
        return;
    }
}

//! Notify that a finger has been updated.
void SurfacePlugin::changedContact(SurfaceContact &c)
{
    //const osg::GraphicsContext::Traits *traits = coVRConfig::instance()->windows[0].window->getTraits();
    //if(!traits)
    //	return;

    if (c.finger)
    {
        // update _fingerContact
        std::list<SurfaceContact>::iterator it;
        for (it = _fingerContacts.begin(); it != _fingerContacts.end(); it++)
        {
            if (it->Id == c.Id)
            {
                TouchContact con(c.CenterX, c.CenterY, c.Id);
                multitouchPlugin->updateContact(con);
                (*it) = c;
                return;
            }
        }
    }
    else
    {
        // update _otherContact
        std::list<SurfaceContact>::iterator it;
        for (it = _otherContacts.begin(); it != _otherContacts.end(); it++)
        {
            if (it->Id == c.Id)
            {
                (*it) = c;
                return;
            }
        }
    }
    cerr << "EXCEPTION @SurfacePlugin::changedContact: \n contact ID = " << c.Id << " could not be updated" << endl;
    cerr << "removing and re-adding contact..." << endl;
    removedContact(c);
    addedContact(c);
}

//! A finger is no longer active.
void SurfacePlugin::removedContact(SurfaceContact &c)
{
    // remove _fingerContact
    if (c.finger)
    {
        // remove _fingerContact from list
        for (std::list<SurfaceContact>::iterator it = _fingerContacts.begin(); it != _fingerContacts.end(); it++)
        {
            if (it->Id == c.Id)
            {
                TouchContact con(c.CenterX, c.CenterY, c.Id);
                multitouchPlugin->removeContact(con);
                _fingerContacts.erase(it);
                return;
            }
        }
        // contact could not be removed, trying to remove falsely identified contacts
        for (std::list<SurfaceContact>::iterator it = _otherContacts.begin(); it != _otherContacts.end(); it++)
        {
            if (it->Id == c.Id)
            {
                _otherContacts.erase(it);
                return;
            }
        }
    }
    // remove _otherContact
    else
    {
        for (std::list<SurfaceContact>::iterator it = _otherContacts.begin(); it != _otherContacts.end(); it++)
        {
            if (it->Id == c.Id)
            {
                _otherContacts.erase(it);
                return;
            }
        }
        // contact could not be removed, trying to remove falsely identified contacts
        for (std::list<SurfaceContact>::iterator it = _fingerContacts.begin(); it != _fingerContacts.end(); it++)
        {
            if (it->Id == c.Id)
            {
                TouchContact con(c.CenterX, c.CenterY, c.Id);
                multitouchPlugin->removeContact(con);
                _fingerContacts.erase(it);
                return;
            }
        }
    }
    // failed to remove contact, print error
    cerr << "EXCEPTION @SurfacePlugin::removedContact: \n contact ID = " << c.Id << " could not be removed" << endl;
}

// coVRTouchTableInterface
int SurfacePlugin::getMarker(std::string name)
{
    return atoi(name.c_str());
}

bool SurfacePlugin::isVisible(int ID)
{
    for (std::list<SurfaceContact>::iterator it = _otherContacts.begin(); it != _otherContacts.end(); it++)
    {
        if (it->Identity == ID)
        {
            return true;
        }
    }
    return false;
}
osg::Vec2 SurfacePlugin::getPosition(int ID) // Ursprung ist links unten X rechts X nach oben
{
    for (std::list<SurfaceContact>::iterator it = _otherContacts.begin(); it != _otherContacts.end(); it++)
    {
        if (it->Identity == ID)
        {
			return osg::Vec2(it->CenterX/coVRConfig::instance()->windows[0].sx,1.0 - (it->CenterY/coVRConfig::instance()->windows[0].sy));
        }
    }
    return osg::Vec2(0, 0);
}
float SurfacePlugin::getOrientation(int ID)
{
    for (std::list<SurfaceContact>::iterator it = _otherContacts.begin(); it != _otherContacts.end(); it++)
    {
        if (it->Identity == ID)
        {
            return it->Orientation;
        }
    }
    return 0;
}

COVERPLUGIN(SurfacePlugin)
