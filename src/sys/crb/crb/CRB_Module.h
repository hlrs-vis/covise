/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise.h>
#include <util/DLinkList.h>

class Start
{
public:
    enum Flags
    {
        Normal,
        Debug,
        Memcheck
    };
};

class module
{
    char *name;
    char *execpath;
    char *category;

public:
    module();
    module(const char *na, const char *ex, const char *ca);
    ~module();

    void set_name(const char *str);
    void set_execpath(const char *str);
    void set_category(const char *str);
    char *get_name()
    {
        return (name);
    };
    char *get_execpath()
    {
        return (execpath);
    };
    char *get_category()
    {
        return (category);
    };
    void start(char *parameter, Start::Flags flags);

private:
};

/************************************************************************/
/* 									*/
/* 			ModuleList 					*/
/* 									*/
/************************************************************************/

class moduleList : public covise::DLinkList<module *>
{
public:
    moduleList();

    // Start a specific module
    int start(char *name, char *category, char *parameter, Start::Flags flags);

    // Find a specific module return true if found and set current position
    int find(char *name, char *category);

    // Find an alias for a given module
    void startRenderer(char *name, char *category);

    char *get_list_message(); // You have to delete the returned pointer
private:
    // search for Modules in Covise_dir/bin/subdir
    void search_dir(char *path, char *subdir);

    // append module, taking aliases into account
    void appendModule(const char *name, const char *execpath, const char *category);

    // module aliases
    struct less_than_str
    {
        bool operator()(const char *s1, const char *s2) const
        {
            return strcmp(s1, s2) < 0;
        }
    };

    typedef pair<const char *, const char *> Alias;
    typedef multimap<const char *, const char *, less_than_str> AliasMap;
    typedef ::set<const char *, less_than_str> AliasedSet;
    AliasMap aliasMap;
    AliasedSet aliasedSet;
};
