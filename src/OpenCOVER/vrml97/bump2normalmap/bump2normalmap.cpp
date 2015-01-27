/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdlib.h>
#include <stdio.h>
#include "bump2normalmap.h"

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("Usage: bump2normalmap heightmap.png bumpmap.png\n");
        return -1;
    }

    glh::array2<unsigned char> bump_img;
    if (read_png_grey(argv[1], bump_img) == -1)
        return -1;

    glh::array2<glh::vec3f> nimg;
    bumpmap_to_normalmap(bump_img, nimg, glh::vec3f(1, 1, .2f));
    int imgsize = nimg.get_width();
    char *simg = new char[imgsize * imgsize * 3];
    char *sip = simg;
    for (int j = 0; j < imgsize; j++)
    {
        for (int i = 0; i < imgsize; i++)
        {
            const glh::vec3f &n = nimg(i, j);

            /* That is the normal map it is used by most bump mapping implementations.
          *sip++ = (char)(  ((n[0] + 1.0f)/2.0f) * ((2 << 7)-1));
          *sip++ = (char)(  ((n[1] + 1.0f)/2.0f) * ((2 << 7)-1));
          *sip++ = (char)(  ((n[2] + 1.0f)/2.0f) * ((2 << 7)-1));
          */

            // That is the normal map used by our bump mapping implementation.
            *sip++ = (char)(((n[0]) * ((2 << 7) - 1)));
            *sip++ = (char)(((n[1]) * ((2 << 7) - 1)));
            *sip++ = (char)(((n[2]) * ((2 << 7) - 1)));
        }
    }

    if (write_png_normalmap(argv[2], simg, imgsize) == -1)
    {
        delete[] simg;
        return -1;
    }

    delete[] simg;
    return 0;
}
