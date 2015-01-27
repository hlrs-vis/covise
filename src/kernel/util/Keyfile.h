/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VR_KEYFILE_H
#define VR_KEYFILE_H

#include <util/coExport.h>
#include <vector>

//#include <stdio.h>

namespace Keyfile
{
std::vector<unsigned char> getKey()
{
    vector<unsigned char> retVal;
    for (int i = 0; i < 32; ++i)
    {
        retVal.push_back((unsigned char)(((i + 11) * (i + 12) + 17) % 255));
    }
    retVal[8] = 87;
    retVal[15] = 12;
    retVal[29] = 143;
    retVal[retVal[15]] = 212;
    for (int i = 0; i < 32; ++i)
    {
        retVal[i] = 255 - retVal[i];
    }
    retVal[28] = 117;

    /*
		// NEVER COMPILE FOR RELEASE WITH THE FOLLOWING CODE UNCOMMENTED
		FILE *outfile = fopen ("C:\\cc.key", "wb");  
		for (int i=0; i<32; ++i)
		{
			fputc(retVal[i], outfile);
		}
		fclose(outfile);
		*/

    return retVal;
}
}

#endif