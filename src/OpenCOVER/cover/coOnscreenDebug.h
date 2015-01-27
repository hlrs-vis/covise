/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_ONSCREENDEBUG_H
#define CO_ONSCREENDEBUG_H

/*! \file
 \brief  Label for onscreen debugging output

 \author Lukas Pinkowski <lukas.pinkowski@web.de>
 \author (C) 2008
         High Performance Computing Center Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date   2008
 */

#include <sstream>
#include <util/common.h>
#include <osg/Version>
#include <osg/Camera>
#include <osg/Geode>
#include <osgText/Text>
namespace opencover
{
/*!
 * \brief This class handles onscreen debugging output
 * 
 * With this class you can print debugging messages on the screen.
 * In a cave, every screen shows output which is local to it, so
 * you can compare values for each screen. 
 *
 * Example:
 * ---
 *    #include <sstream>
 *    ...
 *    ostringstream& os = coOnscreenDebug::instance()->out();
 *    os << "Hello World!" << std::endl;
 *    coOnscreenDebug::instance()->updateString()
 *    ...
 * ---
 * 
 * The code above creates and initializes the debug label if necessary
 * and returns a stringstream reference to which you can direct your 
 * information.
 * When finished, you need to call updateString(), so the contents of the
 * stringstream are flushed to the label.
 */
class COVEREXPORT coOnscreenDebug
{
public:
    /// show the debugging output
    void show();

    /// hide the debugging output
    void hide();

    /// toggle visibility
    void toggleVisibility();

    /// redraw the debugging output
    void redraw();

    /// update (does nothing)
    void update();

    /// set the text of the label
    void setText(const char *text);

    /*!
       * retrieve the singleton instance of coOnscreenDebug
       */
    static coOnscreenDebug *instance();

    /*!
       * retrieve the ostringstream for output
       */
    ostringstream &out();

    /*!
       * flush the contents of the ostringstream to the internal text.
       * This clears the ostringstream, so anything from the last
       * updateString() is lost!
       */
    void updateString();

protected:
    coOnscreenDebug();
    virtual ~coOnscreenDebug();

    bool visible;
    osg::ref_ptr<osgText::Text> text;
    osg::ref_ptr<osg::Geode> geode;
    osg::ref_ptr<osg::Camera> camera;

    static coOnscreenDebug *singleton;

    ostringstream os;
};
}
#endif //CO_ONSCREENDEBUG_H
