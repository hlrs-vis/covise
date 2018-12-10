/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COMLB_H
#define COMLB_H


#include <cover/coVRPluginSupport.h>

struct MLBHeader {
	float left;
	float bottom;
	float width;
	float height;

	int t_width;
	int t_height;
	int t_depth;
};

class VRML97COVEREXPORT coMLB
{
public:
    
    coMLB(std::string fileName);
    virtual ~coMLB();

	//std::vector<double> candela;   // used to be float
	std::vector<unsigned char> data;   // used to be float
	int numValues;

	MLBHeader header;
    osg::Image *getTexture();

protected:
    std::string fileName;
	std::fstream *fh;
    bool readData();
};
#endif
