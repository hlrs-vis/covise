/** @file archivefileinfo.cxx
 * container for information about an archive file.
 * FAMU Copyright (C) 1998-2006 Institute for Theory of Electrical Engineering
 * @author W. Hafla
 */

// #include "../../tools/sourcecode/archivefileinfo.hxx" // container for information about an archive file.

#ifndef __archivefileinfo_hxx__
#define __archivefileinfo_hxx__

#include "os.h"
#include <string>
#include "serializable.hxx" // a serializable object.
#include "objectinputstream.hxx" // a stream that can be used for deserialization.
#include "errorinfo.h" // a container for error data.

/**
 * container for information about an archive file.
 */
class ArchiveFileInfo : public Serializable
{
public:
    ArchiveFileInfo(std::string version, 
                    double versionNumber, 
                    std::string content, 
                    std::string comment,
                    std::string inputFile);
    ArchiveFileInfo(ObjectInputStream* archive);
    virtual ~ArchiveFileInfo() {};
    
    ArchiveFileInfo(const ArchiveFileInfo&src) : Serializable(src), _version(""), _versionNumber(0.), _content(""), _comment(""), _inputFile("") {ERROR0("illegal call", NULL);};
    ArchiveFileInfo& operator = (const ArchiveFileInfo &) {ERROR0("illegal call", NULL);};

    virtual void writeObject(ObjectOutputStream* archive) const;
    
    std::string getSaveTime(void) const;
    std::string getVersion(void) const { return _version; };
    double getVersionNumber(void) const { return _versionNumber; };
    std::string getContent(void) const { return _content; };
    std::string getComment(void) const { return _comment; };
    std::string getInputFile(void) const { return _inputFile; };
    
    void writeInfoToFile(std::string filenameOfResultsFile) const;
    
private:
    std::string _version;
    double      _versionNumber;
    time_t      _saveTime;      ///< time at which file was saved.
    std::string _content;
    std::string _comment;
    std::string _inputFile;
};


#endif
