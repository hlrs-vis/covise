/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file errorinfo.h
 * a container for error data.
 */

// #include "errorinfo.h" // a container for error data.

#ifndef __errorinfo_hxx__
#define __errorinfo_hxx__

#include "os.h"
#include "OutputHandler.h" // an output handler for displaying information on the screen.
#include <string>

/**
 * outputs an error message that has no output value
 */
#define ERROR0(err, outputHandler)                                                       \
    {                                                                                    \
        throw ErrorInfo((outputHandler), (err), __LINE__, __FILE__, __DATE__, __TIME__); \
    }

/**
 * outputs an error message that has 1 output value
 */
#define ERROR1(err, param, outputHandler)                                                   \
    {                                                                                       \
        std::string errMsg = err + param;                                                   \
                                                                                            \
        throw ErrorInfo((outputHandler), (errMsg), __LINE__, __FILE__, __DATE__, __TIME__); \
    }

/**
 * outputs an error message that has no output value
 */
#define ASSERT(condition, outputHandler)                \
    {                                                   \
        std::string err = "an internal error occured."; \
        if (!(condition))                               \
            ERROR0((err), (outputHandler))              \
    }

/**
 * outputs an error message that has no output value
 */
#define ASSERT0(condition, err, outputHandler) \
    {                                          \
        if (!(condition))                      \
            ERROR0((err), (outputHandler))     \
    }

/**
 * outputs an error message that has no output value
 */
#define ASSERT1(condition, err, param, outputHandler) \
    {                                                 \
        if (!(condition))                             \
            ERROR1(err, param, outputHandler)         \
    }

/**
 * deletes an array.
 * performs checks if  <CODE> arrayName != NULL </CODE> and performs <CODE> delete [] arrayName </CODE> if so.
 * @param arrayName identifier of the array
 */
#define DELETE_ARRAY(arrayName) \
    if ((arrayName) != NULL)    \
        delete[](arrayName);    \
    (arrayName) = NULL;

/**
 * deletes an object.
 * performs checks if  <CODE> arrayName != NULL </CODE> and performs <CODE> delete objectName </CODE> if so.
 * @param arrayName identifier of the array
 */
#define DELETE_OBJ(objectName) \
    if ((objectName) != NULL)  \
        delete (objectName);   \
    (objectName) = NULL;

/**
 * an error.
 */
class ErrorInfo
{
public:
    ErrorInfo(OutputHandler *outputHandler,
              std::string errMessage,
              int sourceFileLineNo,
              std::string sourceFilename,
              std::string compilationDate,
              std::string compilationTime);
    virtual ~ErrorInfo(){};

    std::string getFilename()
    {
        return _sourceFilename;
    };
    INT_F getLineNo()
    {
        return _sourceFileLineNo;
    };
    std::string getDate()
    {
        return _compilationDate;
    };
    std::string getCompilationTime()
    {
        return _compilationTime;
    };

    void outputError();

private:
    OutputHandler *_outputHandler;
    std::string _errMessage;
    INT_F _sourceFileLineNo; ///< line number where the error occured
    std::string _sourceFilename; ///< filename of the source file where the error occured
    std::string _compilationDate; ///< month/day/year of compilation
    std::string _compilationTime; ///< hour:minute:second of compilation

}; //class ErrorInfo

#endif
