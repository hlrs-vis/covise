#ifndef __CLINK__
#define __CLINK__

#include "SharedTypes.h"

class CLink
{
    Vector3f global_link_origin;

    /************************************************************************/
    /* DH parametrs                                                         */
    /************************************************************************/
    float a;
    float alpha;

    //TODO : add mass center

public:
    CLink(float a_in , float alpha_in);

    OUT float GetCommonNormalParametr_a(){return a;}
    OUT float GetZAxisRotationParametr_aplha(){return alpha;}
};

#endif