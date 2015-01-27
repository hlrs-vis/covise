/** @file objectinputstreambinary.hxx
 * a stream that can be used for deserialization/binary files.
 * FAMU Copyright (C) 1998-2006 Institute for Theory of Electrical Engineering
 * @author W. Hafla
 */

// #include "objectinputstreambinary.hxx" // a stream that can be used for deserialization/binary files.

#ifndef __objectinputstreambinary_hxx__
#define __objectinputstreambinary_hxx__

#include <string>
#include <fstream>

#include "objectinputstreambinarybasic.hxx" // a stream that can be used for deserialization/binary files, basic functionality.
#include "archivefileheader.hxx"    // header of the archive file.
#include "archivefilefooter.hxx"    // footer of the archive file.
#include "errorinfo.h"              // a container for error data.
#include "OutputHandler.h"          // an output handler for displaying information on the screen.

/**
 * a stream that can be used for deserialization/binary files.
 */
class ObjectInputStreamBinary : public ObjectInputStreamBinaryBasic
{
public:

    ObjectInputStreamBinary(std::string filename, OutputHandler* outputHandler);
    virtual ~ObjectInputStreamBinary();
    
    ObjectInputStreamBinary(const ObjectInputStreamBinary&) : ObjectInputStreamBinaryBasic("", NULL) {ERROR0("illegal call", NULL); };
    ObjectInputStreamBinary& operator = (const ObjectInputStreamBinary &) {ERROR0("illegal call", NULL); };
    
private:    
    
    OutputHandler* _outputHandler;

    void writeFileInfo(std::string resultsFilename);
    void seekPosition(const ULONG_F& tablePos);
};

#endif
