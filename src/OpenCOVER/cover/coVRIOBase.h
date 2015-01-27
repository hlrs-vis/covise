/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVRIOBASE_H
#define COVRIOBASE_H

#include <string>

#include <util/coExport.h>

namespace opencover
{
class COVEREXPORT coVRIOBase
{
public:
    coVRIOBase();
    virtual ~coVRIOBase();

    float getIOProgress() const;

    /**
    * Abort current IO operation
    */
    virtual bool abortIO() = 0;

    /**
    * Get the name of the IO handler
    * @returns The handler name
    */
    virtual std::string getIOHandlerName() const = 0;

    /**
    * Check the IO capability of the handler
    * @returns true if the handler can read data
    */
    virtual bool isReader() const
    {
        return false;
    }

    /**
    * Check the IO capability of the handler
    * @returns true if the handler can write data
    */
    virtual bool isWriter() const
    {
        return false;
    }

protected:
    void setIOProgress(float progress);

private:
    float progress;
};
}
#endif // COVRIOBASE_H
