/** @file archivefilefooter.cxx
 * footer of the archive file.
 * FAMU Copyright (C) 1998-2006 Institute for Theory of Electrical Engineering
 * @author W. Hafla
 */

// #include "../../tools/sourcecode/archivefilefooter.hxx" // footer of the archive file.

#ifndef __archivefilefooter_hxx__
#define __archivefilefooter_hxx__

#include "os.h"
#include <string>
#include "serializable.hxx" // a serializable object.
#include "objectinputstream.hxx" // a stream that can be used for deserialization.
#include "errorinfo.h" // a container for error data.


/**
 * footer of the archive file.
 */
class ArchiveFileFooter : public Serializable
{
public:
    ArchiveFileFooter(void);
    ArchiveFileFooter(ObjectInputStream* archive);
    virtual ~ArchiveFileFooter() {};
    virtual void writeObject(ObjectOutputStream* archive) const;
    
    std::string getInfo(void) const;
    
private:
    std::string _logFile;
    ArchiveFileFooter(const ArchiveFileFooter& t);
    ArchiveFileFooter& operator = (const ArchiveFileFooter &t);
};


#endif
