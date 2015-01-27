/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS CarFiles
//
// Initial version: 2002-05-13 [sk]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2002 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//
#include "CarFiles.h"
#include <util/coviseCompat.h>

#ifdef __sgi
#include <sys/dir.h>
#else
#ifndef WIN32
#include <dirent.h>
#endif
#endif

#ifdef WIN32
#define DIR_SEP '\\'
#else
#define DIR_SEP '/'
#endif

#undef VERBOSE

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Static Variable initializers
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Constructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
CarFiles::CarFiles(const char *path)
{
    cfiles = new CarGroup("_all_");

    path_ = new char[strlen(path) + 1];
    strcpy(path_, path);
#ifndef WIN32
    DIR *dirp;
#ifdef __sgi
    struct direct *entry;
#else
    struct dirent *entry;
#endif
#endif

    char *model = strrchr(path_, DIR_SEP);
    if (model == NULL)
    {
        return;
    }
    std::string base_name(model);
    //base_name.del(DIR_SEP,10); ???
    //remove ending filename
    if (model != NULL)
    {
        *model = '\0';
    }

    // current group name and last handled group name
    char grp_name[6];

    CarGroup *grp;

    int i, found, match, pos;

    dirp = opendir(path_);
    if (dirp != NULL)
    {
        while ((entry = readdir(dirp)) != NULL)
        {
            char *ext = strstr(entry->d_name, ".car");
            if (ext && !strcmp(ext, ".car"))
            {
                char *fullname = new char[strlen(path_) + strlen(entry->d_name) + 2];
                sprintf(fullname, "%s%c%s", path_, DIR_SEP, entry->d_name);
                cfiles->add(fullname);

                // differentiate list into result_grps
                strncpy(grp_name, (entry->d_name) + base_name.length() - 1, 5);
                grp_name[5] = '\0';

                if (grp_name[0] == '_' && grp_name[4] == '_' && isdigit(grp_name[1]) && isdigit(grp_name[2]) && isdigit(grp_name[3]))
                {
                    // look if group exists
                    found = -1;
                    pos = AT_END; // position after which we insert new grp into list

                    for (i = 0; i < group.length(); i++)
                    {
                        match = strcmp(grp_name, group[i]->getName());
                        if (match > 0)
                        {
                            pos = i;
                        }
                        else if (match == 0)
                        {
                            found = i;
                            break;
                        }
                        else if (i == 0)
                        {
                            pos = AT_BEGIN; // insert at begin of list
                            break;
                        }
                    }

                    if (found == -1) // new group starts
                    {
                        grp = new CarGroup(grp_name);
                        if (pos == AT_BEGIN)
                        {
                            group.set(0);
                            group.insertBefore(grp);
                        }
                        else if (pos == AT_END)
                        {
                            group.append(grp);
                        }
                        else
                        {
                            group.set(pos);
                            group.insertAfter(grp);
                        }
                    }
                    else
                    {
                        grp = group[i];
                    }

                    // add file name
                    char *fullname2 = new char[strlen(fullname) + 1];
                    strcpy(fullname2, fullname);
                    if (grp != NULL)
                    {
                        grp->add(fullname2);
                    }
                }
            }
        }
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Destructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

CarFiles::~CarFiles()
{
    int i;
    delete cfiles;
    for (i = 0; i < numGroups(); i++)
    {
        delete group[i];
    }
}

int
CarFiles::numGroups()
{
    return group.length();
}

int
CarFiles::numFiles()
{
    return cfiles->numFiles();
}

int
CarFiles::numFiles(int gr)
{
    return group[gr]->numFiles();
}

const char *
CarFiles::get(int x)
{
    return cfiles->getFile(x);
}

const char *
CarFiles::get(int x, int gr)
{
    return group[gr]->getFile(x);
}

const char *
CarFiles::getName(int gr)
{
    return group[gr]->getName();
}
