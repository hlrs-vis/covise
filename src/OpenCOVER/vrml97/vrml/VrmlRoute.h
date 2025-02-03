#ifndef COVER_VRML_ROUTE_H
#define COVER_VRML_ROUTE_H


namespace vrml{
class VrmlNode;
class Route
{
public:
    Route(const char *fromEventOut, VrmlNode *toNode, const char *toEventIn, VrmlNode *fromNode);
    Route(const Route &);
    ~Route();

    char *fromEventOut()
    {
        return d_fromEventOut;
    }
    char *toEventIn()
    {
        return d_toEventIn;
    }
    VrmlNode *toNode()
    {
        return d_toNode;
    }
    VrmlNode *fromNode()
    {
        return d_fromNode;
    }

    void addFromImportName(const char *name);
    void addToImportName(const char *name);

    Route *newFromRoute(VrmlNode *newFromNode);
    Route *newToRoute(VrmlNode *newToNode);

    VrmlNode *newFromNode(void);
    VrmlNode *newToNode(void);

    Route *prev()
    {
        return d_prev;
    }
    Route *next()
    {
        return d_next;
    }
    void setPrev(Route *r)
    {
        d_prev = r;
    }
    void setNext(Route *r)
    {
        d_next = r;
    }
    Route *prevI()
    {
        return d_prevI;
    }
    Route *nextI()
    {
        return d_nextI;
    }
    void setPrevI(Route *r)
    {
        d_prevI = r;
    }
    void setNextI(Route *r)
    {
        d_nextI = r;
    }

private:
    char *d_fromEventOut;
    VrmlNode *d_toNode;
    VrmlNode *d_fromNode;
    char *d_toEventIn;

    char *d_fromImportName;
    char *d_toImportName;

    Route *d_prev, *d_next;
    Route *d_prevI, *d_nextI;
};
}

#endif // COVER_VRML_ROUTE_H