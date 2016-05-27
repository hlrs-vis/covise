#include <osg/Image>
#include "tridelity.h"

static const GLuint R = 0x000000ff;
static const GLuint G = 0x0000ff00;
static const GLuint B = 0x00ff0000;

//-----------------------------------------------------------
// Multiview lookup (MV-Series)
//-----------------------------------------------------------
osg::Image *tridelityLookupMV()
{
    const int width = 25;
    const int height = 5;
    GLuint lookup[height][width];

    // VIEW 5
    lookup[0][0] = B;
    lookup[1][0] = G;
    lookup[2][0] = R;
    lookup[3][0] = 0;
    lookup[4][0] = 0;
    for (int i=0; i<5; ++i)
    {
        for (int j=1; j<5; ++j)
        {
            int ii = (i+2*j)%5;
            lookup[ii][j] = lookup[i][0];
        }
    }

    for (int v=1; v<5; ++v)
    {
        for (int i=0; i<5; ++i)
        {
            int ii = (i+v)%5;
            for (int j=0; j<5; ++j)
            {
                lookup[ii][j+v*5] = lookup[i][j];
            }
        }
    }

    unsigned char *d = new unsigned char[sizeof(lookup)];
    memcpy(d, &lookup[0][0], sizeof(lookup));
    osg::Image *img = new osg::Image;
    img->setImage(width, height, 1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, d, osg::Image::USE_NEW_DELETE);
    return img;
}

//-----------------------------------------------------------
// Multiview lookup (ML-Series)
//-----------------------------------------------------------
osg::Image *tridelityLookupML()
{
    const int width = 25;
    const int height = 15;
    GLuint lookup[height][width];
    memset(&lookup[0][0], 255, sizeof(lookup));

    //******** Ansicht 1 *********
    lookup[0][0] = 0;
    lookup[1][0] = 0;
    lookup[2][0] = 0;
    lookup[3][0] = B;
    lookup[4][0] = G;
    lookup[5][0] = G;
    lookup[6][0] = R;
    lookup[7][0] = 0;
    lookup[8][0] = 0;
    lookup[9][0] = 0;
    lookup[10][0] = B;
    lookup[11][0] = B;
    lookup[12][0] = G;
    lookup[13][0] = R;
    lookup[14][0] = R;

    for (int k=0; k<15; ++k)
    {
        for (int i=1; i<5; ++i)
        {
            int kk = (k+12*i)%15;
            lookup[k][i] = lookup[kk][0];
        }
    }

    for (int k=0; k<15; ++k)
    {
        int k1 = (k+9)%15;
        int k2 = (k+3)%15;
        int k3 = (k+12)%15;
        int k4 = (k+6)%15;
        for (int i = 0; i < 5; i++)
        {
            lookup[k][5 + i] = lookup[k1][i];
            lookup[k][10 + i] = lookup[k2][i];
            lookup[k][15 + i] = lookup[k3][i];
            lookup[k][20 + i] = lookup[k4][i];
        }
    }

    unsigned char *d = new unsigned char[sizeof(lookup)];
    memcpy(d, &lookup[0][0], sizeof(lookup));
    osg::Image *img = new osg::Image;
    img->setImage(width, height, 1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, d, osg::Image::USE_NEW_DELETE);
    return img;
}
