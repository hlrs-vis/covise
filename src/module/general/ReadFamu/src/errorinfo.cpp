/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file errorinfo.h
 * a container for error data.
 */

#include "errorinfo.h" // a container for error data.
#include <sstream>

ErrorInfo::ErrorInfo(OutputHandler *outputHandler,
                     std::string errMessage,
                     int sourceFileLineNo,
                     std::string sourceFilename,
                     std::string compilationDate,
                     std::string compilationTime)
    : _outputHandler(outputHandler)
    , _errMessage(errMessage)
    , _sourceFileLineNo(sourceFileLineNo)
    , _sourceFilename(sourceFilename)
    , _compilationDate(compilationDate)
    , _compilationTime(compilationTime)
{
    // place your breakpoint here
}

void ErrorInfo::outputError()
{
    _outputHandler->displayError(_errMessage.c_str());

    std::ostringstream s;
    s << "(\"" << _sourceFilename << "\", line " << (int)_sourceFileLineNo
      << ", compiled " << _compilationDate << ", " << _compilationTime << ")";
    std::string ss = s.str();
    _outputHandler->displayError(ss.c_str());
}
