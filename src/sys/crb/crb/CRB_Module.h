/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise.h>
#include <util/DLinkList.h>
#include <string>

namespace covise{
struct CRB_EXEC;
} // namespace covise
class module
{
    std::string name;
    std::string execpath;
    std::string category;

public:
    module() = default;
    module(const char *name, const char *execpath, const char *category);

    void set_name(const char *str);
    void set_execpath(const char *str);
    void set_category(const char *str);
    const char *get_name() const;
    const char *get_execpath() const;
    const char *get_category() const;

    void start(const covise::CRB_EXEC& exec);

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
    bool start(const covise::CRB_EXEC& exec);

    // Find a specific module return true if found and set current position
    int find(const char *name,const char *category);

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
