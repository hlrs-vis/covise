/** @file archivefileheader.cxx
 * header of the archive file.
 * FAMU Copyright (C) 1998-2006 Institute for Theory of Electrical Engineering
 * @author W. Hafla
 */

// #include "../../tools/sourcecode/archivefileheader.hxx" // header of the archive file.

#ifndef __archivefileheader_hxx__
#define __archivefileheader_hxx__

#include "os.h"
#include <string>
#include "serializable.hxx" // a serializable object.
#include "objectinputstream.hxx" // a stream that can be used for deserialization.



/**
 * header of the archive file.
 */
class ArchiveFileHeader : public Serializable
{
public:
    ArchiveFileHeader(std::string version, 
                      double versionNumber, 
                      std::string content, 
                      std::string comment,
                      std::string inputFile);
    ArchiveFileHeader(ObjectInputStream* archive);
    virtual ~ArchiveFileHeader() {};
    
    virtual void writeObject(ObjectOutputStream* archive) const;
    
    std::string getSaveTime(void) const;
    std::string getVersion(void) const       { return _version;       };
    double      getVersionNumber(void) const { return _versionNumber; };
    std::string getContent(void) const       { return _content;       };
    std::string getComment(void) const       { return _comment;       };
    std::string getInputFile(void) const     { return _inputFile;     };
    
    std::string getInfo(void) const;
    
private:
    std::string _version;
    double      _versionNumber;
    time_t      _saveTime;      ///< time at which file was saved.
    std::string _content;
    std::string _comment;
    std::string _inputFile;
};


#endif
