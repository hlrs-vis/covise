/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file Tools/OutputManager.h
 * @brief contains definition of class Tools::OutputManager.
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 19.10.2003
 * created
 * @date 5.11.2003
 * moved static member classInfo to private scope
 */

/** @class Tools::ClassInfo_ToolsOutputManager
 * @brief used to register class Tools::OutputManager at class manager
 *
 * \b Description:
 *
 * Class defined by macro CLASSINFO(). This class is friend of
 * Tools::OutputManager and is therefore the only class allowed to create new
 * objects. Class manager is the only class which is allowed to tell
 * ClassInfo_ToolsOutputManager to create new objects of type
 * Tools::OutputManager.
 */

/** @fn CLASSINFO(ClassInfo_ToolsOutputManager, OutputManager);
 * @brief creates class ClassInfo_ToolsOutputManager.
 *
 * \b Description:
 *
 * The call of this macro creates the new class ClassInfo_ToolsOutputManager
 * which is used to create new objects of Tools::OutputManager by the class
 * manager.
 */

/** @class Tools::OutputManager
 * @brief provides generic output interface
 *
 * \b Description:
 *
 * To avoid that the application is restricted to a specific output device this
 * class provides a generic output interface.
 *
 * Outputs themself are in the same format than with cout.
 *
 * e.g.
 * @code
 * OUT << "some text" << some_value << ENDL;
 * @endcode
 *
 * This class returns an ostream object to the caller ( through getOut() ).
 * That ostream object is then used to do the actual output.
 *
 * The output is formated with several defines:
 * - OUT : calls Tools::OutputManager::getOut() and provides an ostream object.
 * - ENDL: appends endline, flushes output, and displays the contents of OUT.
 * - END: sends output without flushing it (means: no output is seen). after
 * subsequent call to ENDL the output is available.
 * - DEBUGL: these outputs are only visible when DEBUG_MODE is set. same as
 * ENDL.
 * - DEBUG: these output are only visible when DEBUG_MODE is set. same as END
 *
 * If someone dislikes this interface, then he can also use print(), but that
 * function has some drawbacks. It uses a variable argument list which leads
 * to several problems. See documentation about variable argument lists for
 * more details. Syntax is basically like that for printf.
 *
 * @see Tools::OutputManagerBase for details about the required define macros.
 */

/** @fn Tools::OutputManager::OutputManager();
 * @brief default constructor
 *
 * \b Description:
 *
 * This constructor should never be called because new objects should always
 * created by calling Tools::OutputManager::OutputManager(int objectID).
 *
 * The class manager is the only object which is allowed to call that
 * constructor.
 */

/** @fn Tools::OutputManager::OutputManager(int objectID);
 * @brief called when new objects are created
 *
 * @param objectID - unique identifier for the new object
 *
 * \b Description:
 *
 * New objects of Tools::OutputManager are created by the class manager which
 * uses this constructor to give the output manager a unique ID by which it is
 * managed.
 */

/** @fn virtual Tools::OutputManager::~OutputManager();
 * @brief destructor.
 *
 * \b Description:
 *
 * Called when an object of class Tools::OutputManager is deleted by the class
 * manager.
 */

/** @fn void Tools::OutputManager::setDebug(bool debugOnOff);
 * @brief turn debug mode on/off
 *
 * @param debugOnOff - boolean value indicating if the debug mode is to be
 * turned on or off.
 *
 * \b Description:
 *
 * This function is only relevant when print() is used for outputs.
 */

/** @fn void Tools::OutputManager::print(char* format, bool critical, ...);
 * @brief provides formatted outputs
 *
 * @param format - format string used to format the output. Format is the same
 * as in printf().
 * @param critical - indicates if the message to print is a critical message.
 * These are always printed.
 * @param ... - variable argument list
 *
 * \b Description:
 *
 * This function provides the 'old' output functionality of the output manager.
 * The call through an variable argument list has some drawbacks:
 * - always use \c int for integer type numbers.
 * - always use \c double for floating point numbers.
 * - always use \c char* for text ( no string! instead use string.c_str() )
 *
 * It is recommended to use the newer output interface provided by the macros:
 * - OUT
 * - ENDL
 * - END
 * - DEBUGL
 * - DEBUG
 *
 * @see Tools::OutputManagerBase for more information about these macros.
 */

/** @fn ostream* Tools::OutputManager::getOut();
 * @brief get ostream object used for output
 *
 * @return new ostream object. NULL if creation of ostream failed.
 *
 * \b Description:
 *
 * Returns ostream object used later to generate some output. Always use
 * the macro OUT to create (and use) the ostream object and use ENDL, END,
 * DEBUGL, DEBUG to close and delete it.
 *
 * @see Tools::OutputManagerBase for more informations about these macros.
 */

/** @var bool Tools::OutputManager::debug;
 * @brief indicates if debug mode is set
 *
 * \b Description:
 *
 * This variable is provided to support the 'old' output interface of
 * Tools::OutputManager. After call to setDebug() this variable is set and
 * subsequent, non-critical messages are printed out with function print().
 */

/** @var stringstream Tools::OutputManager::s;
 * @brief stringstream needed by created ostream objects
 *
 * \b Description:
 *
 * The created ostream object ( by getOut() ) needs a stringstream object for
 * its string buffer.
 */

/** @var static ClassInfo_ToolsOutputManager Tools::OutputManager::classInfo;
 * @brief registers class Tools::OutputManager at class manager
 *
 * \b Description:
 *
 * The class is registered when this object is initialized (in Tools/OutputManager.cpp).
 */

#ifndef __TOOLS_OUTPUTMANAGER_H_
#define __TOOLS_OUTPUTMANAGER_H_

#include "classmanager.h"

using namespace std;

namespace Tools
{
class ClassInfo_ToolsOutputManager;

class OutputManager : public OutputManagerBase
{
    /** @brief declares that ClassInfo_ToolsOutputManager is allowed to call
       * the private functions.
       *
       * \b Description:
       *
       * ClassInfo_ToolsOutputManager is allowed to call the private constructors
       * of Tools::OutputManager.
       */
    friend class ClassInfo_ToolsOutputManager;

private:
    bool debug;
    stringstream s;

    OutputManager();
    OutputManager(string className, int objectID);

    static ClassInfo_ToolsOutputManager classInfo;

public:
    virtual ~OutputManager();

    void setDebug(bool debugOnOff);

    void print(char *format, bool critical, ...);

    ostream *getOut();
};

CLASSINFO(ClassInfo_ToolsOutputManager, OutputManager);
};
#endif
