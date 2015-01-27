/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
#ifndef _VRMLNAMESPACE_
#define _VRMLNAMESPACE_

//
// The VrmlNamespace class keeps track of defined nodes and prototypes.
//

#include "config.h"

#ifdef _WIN32
#include "VrmlNodeType.h"
#include "VrmlNode.h"
#else
#endif

#include <list>
#include <map>
#include <string>

namespace vrml
{

class VrmlNodeType;
class VrmlNode;
class VrmlNamespace;

typedef std::list<VrmlNamespace *> NamespaceList;

class VRMLEXPORT VrmlNamespace
{
public:
    VrmlNamespace(VrmlNamespace *parent = 0);
    ~VrmlNamespace();

    // PROTO definitions add node types to the namespace.
    // PROTO implementations are a separate node type namespace,
    // and require that any nested PROTOs NOT be available outside
    // the PROTO implementation. PROTOs defined outside the current
    // namespace are available.

    // addNodeType will print an error if the given type
    // is already defined (spec says behavior is undefined).
    void addNodeType(VrmlNodeType *);

    // DEFd nodes add node names to the namespace.
    // Node names are only defined in the current name space. They
    // are not available outside of the PROTO they are defined in,
    // nor are they available inside of nested PROTOs.

    void addNodeName(VrmlNode *);
    void removeNodeName(VrmlNode *);

    // Find a node type, given a type name. Returns NULL if type is not defined.
    const VrmlNodeType *findType(const char *nm);
    // Find a node type, given a type name. Returns NULL if type is not defined.
    const VrmlNodeType *findOnlyType(const char *nm);

    // Larry Feb 10_99  Find a nodeType, given a PROTO name
    const VrmlNodeType *findPROTO(const char *nm);

    // Return the first node type in scope (default EXTERNPROTO implementation)
    const VrmlNodeType *firstType();

    // Find a node by name.
    VrmlNode *findNode(const char *name);

    // Find a node by name in a specified Namespace.
    static VrmlNode *findNode(const char *name, int num);

    // get the number of this namespace
    int getNumber()
    {
        return namespaceNum;
    };

    static void addBuiltIn(VrmlNodeType *);

    // repair ROUTEs to IMPORTed nodes
    void repairRoutes();

    void addExportAs(std::string name, std::string exportName);

    std::string getExportAs(std::string name);

    // Builtin node types are stored (once) in this data structure:
    static std::list<VrmlNodeType *> builtInList;

private:
    void defineBuiltIns();

    // Defined node types (PROTOs) for this namespace
    std::list<VrmlNodeType *> d_typeList;

    // Defined node names for this namespace
    std::list<VrmlNode *> d_nameList;

    // Parent namespace
    VrmlNamespace *d_parent;

    // all Namespaces
    static NamespaceList allNamespaces;

    //  number of this Namespace
    int namespaceNum;
    static bool definedBuiltins;

    // map for EXPORT ... AS commands
    typedef std::map<std::string, std::string> ExportAsMap;
    ExportAsMap exportAsMap;
};
}
#endif // _VRMLNAMESPACE_
