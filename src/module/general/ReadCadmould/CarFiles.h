/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __CAR_FILES_H_
#define __CAR_FILES_H_
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS CarFiles
//
// Initial version: 2002-05-13 [sk]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2002 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//

#include <util/coviseCompat.h>
#include <vector>
#include <string>

/**
 * This Class handles the names of all .car files in one directory.
 */

class CarGroup;

class CarFiles
{
public:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Constructors / Destructor
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /** Constructor
       *
       */
    CarFiles(const char *path);

    /// Destructor : virtual in case we derive objects
    virtual ~CarFiles();

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Operations
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// get number of files( in whole directory)
    virtual int numFiles();

    /// get number of files( in whole directory)
    virtual int numGroups();

    /// get number of files in group
    virtual int numFiles(int gr);

    /// get file name number x
    virtual const char *get(int x);

    /// get file name number x in group
    virtual const char *get(int x, int gr);

    /// get name of group
    virtual const char *getName(int gr);

protected:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Attributes
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// pathname
    char *path_;

private:
    enum
    {
        AT_END = -2,
        AT_BEGIN = -1
    };

    /// list of result groups
    std::vector<CarGroup *> group;

    /// list of all car_files
    CarGroup *cfiles;

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Internally used functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  prevent auto-generated bit copy routines by default
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// Copy-Constructor: NOT IMPLEMENTED
    CarFiles(const CarFiles &);

    /// Assignment operator: NOT IMPLEMENTED
    CarFiles &operator=(const CarFiles &);

    /// Default constructor: NOT IMPLEMENTED
    CarFiles();
};

/**
 *   This class handles a set of filenames
 *   Only to be used by class CarFiles
 */
class CarGroup
{
public:
    void add(const char *file)
    {
        char *dat = new char[strlen(file) + 1];
        strcpy(dat, file);
        for (int i = files_.size() - 1; i >= 0; i--) // generate sorted list
        {
            if (strcmp(file, files_[i]) > 0)
            {
                if (i == files_.size() - 1)
                {
                    files_.push_back(dat);
                    return;
                }
                else
                {
                    std::vector<char *>::iterator it = files_.begin();
                    it += i+1;
                    files_.insert(it, dat);
                    return;
                }
            }
        }
        files_.insert(files_.begin(), dat);
    };

    const char *getName()
    {
        return name;
    };
    int numFiles()
    {
        return files_.size();
    };
    const char *getFile(int i)
    {
        return files_[i];
    };

    CarGroup(const char *grp_name)
    {
        strncpy(name, grp_name, 5);
        name[5] = '\0';
    };
    ~CarGroup()
    {
        for (int i = 0; i < files_.size(); i++)
        {
            delete[] files_[i];
        }
    }

private:
    char name[6]; // group name
    std::vector<char *> files_; // files in group
};
#endif
