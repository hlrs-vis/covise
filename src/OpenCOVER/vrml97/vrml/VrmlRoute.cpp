#include "VrmlRoute.h"
#include "VrmlNode.h"

using namespace vrml;

Route::Route(const char *fromEventOut, VrmlNode *toNode, const char *toEventIn, VrmlNode *fromNode)
    : d_prev(0)
    , d_next(0)
    , d_prevI(0)
    , d_nextI(0)
{
    d_fromEventOut = new char[strlen(fromEventOut) + 1];
    strcpy(d_fromEventOut, fromEventOut);
    //if((strlen(d_fromEventOut) > 8)&&(strcmp(d_fromEventOut+strlen(d_fromEventOut)-8,"_changed")==0))
    //    d_fromEventOut[strlen(d_fromEventOut)-8]='\0';
    d_toNode = toNode;
    toNode->addRouteI(this);
    d_fromNode = fromNode;
    d_toEventIn = new char[strlen(toEventIn) + 1];
    //if(strncmp(toEventIn,"set_",4) == 0)
    //    strcpy(d_toEventIn, toEventIn+4);
    //else
    strcpy(d_toEventIn, toEventIn);

    d_fromImportName = new char[1];
    d_fromImportName[0] = '\0';

    d_toImportName = new char[1];
    d_toImportName[0] = '\0';
}

Route::Route(const Route &r)
{
    d_fromEventOut = new char[strlen(r.d_fromEventOut) + 1];
    strcpy(d_fromEventOut, r.d_fromEventOut);
    d_toNode = r.d_toNode;
    d_fromNode = r.d_fromNode;
    d_toEventIn = new char[strlen(r.d_toEventIn) + 1];
    strcpy(d_toEventIn, r.d_toEventIn);

    d_fromImportName = new char[strlen(r.d_fromImportName) + 1];
    if (strlen(r.d_fromImportName) > 0)
        strcpy(d_fromImportName, r.d_fromImportName);
    else
        d_fromImportName[0] = '\0';

    d_toImportName = new char[strlen(r.d_toImportName) + 1];
    if (strlen(r.d_toImportName) > 0)
        strcpy(d_toImportName, r.d_toImportName);
    else
        d_toImportName[0] = '\0';
}

Route::~Route()
{
    if (d_toNode)
    {
        d_toNode->removeRoute(this);
    }
    if (d_fromNode)
    {
        d_fromNode->removeRoute(this);
    }
    delete[] d_fromEventOut;
    delete[] d_toEventIn;

    delete[] d_fromImportName;
    d_fromImportName = NULL;
    delete[] d_toImportName;
    d_toImportName = NULL;
}

void Route::addFromImportName(const char *name)
{
    delete[] d_fromImportName;
    d_fromImportName = new char[strlen(name) + 1];
    strcpy(d_fromImportName, name);
}

void Route::addToImportName(const char *name)
{
    delete[] d_toImportName;
    d_toImportName = new char[strlen(name) + 1];
    strcpy(d_toImportName, name);
}

Route *Route::newFromRoute(VrmlNode *newFromNode)
{
    return newFromNode->addRoute(d_fromEventOut, d_toNode, d_toEventIn);
}

Route *Route::newToRoute(VrmlNode *newToNode)
{
    return d_fromNode->addRoute(d_fromEventOut, newToNode, d_toEventIn);
}

VrmlNode *Route::newFromNode(void)
{
    if (strlen(d_fromImportName) != 0)
        return d_fromNode->findInside(d_fromImportName);
    return NULL;
}

VrmlNode *Route::newToNode(void)
{
    if (strlen(d_toImportName) != 0)
        return d_toNode->findInside(d_toImportName);
    return NULL;
}