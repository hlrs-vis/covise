/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coMLB.h"
#include <iostream>
#include <fstream>
#ifndef WIN32
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#endif
 
 //#include <cnpy/cnpy.h>

coMLB::coMLB(std::string fn)
{	
	std::cout << "(coMLB) raeading...: " << fn << "\n";
    fileName = fn;
    readData();
}
coMLB::~coMLB()
{
}
bool coMLB::readData()
{
	std::fstream fh;
	fh.open(fileName.c_str(), std::fstream::in | std::fstream::binary);

	std::cout << "read header... ";
	fh.read((char*)&header, sizeof(MLBHeader));
	std::cout << "got struct MLBHeader with members: " << std::endl;
	std::cout << "left: " << header.left << std::endl;
	std::cout << "bottom: " << header.bottom << std::endl;
	std::cout << "width: " << header.width << std::endl;
	std::cout << "height: " << header.height << std::endl;
	std::cout << "t_width: " << header.t_width << std::endl;
	std::cout << "t_height: " << header.t_height << std::endl;
	std::cout << "t_depth: " << header.t_depth << std::endl;

	numValues = header.t_width * header.t_height * header.t_depth;

	std::cout << "read data... ";
	data.resize(numValues);
	fh.read((char*)&data[0], numValues); // using the address of first vector element is faster than looping

	fh.close();
	std::cout << "got vector with the following values: " << std::endl;
	std::cout << "data[0] = " << (int)(data[0]) << std::endl;
	std::cout << "data[1] = " << (int)(data[1]) << std::endl;
	std::cout << "data[-1] = " << (int)(data[numValues-1]) << std::endl;

    return true;
}


osg::Image *coMLB::getTexture()
{
    osg::Image *image = new osg::Image();
	unsigned char *data_array = &data[0];
    osg::ref_ptr<osg::Image> texImgRG = new osg::Image();
    image->setImage(header.t_width, header.t_height, header.t_depth, 1,
		GL_LUMINANCE, GL_UNSIGNED_BYTE, data_array, osg::Image::USE_NEW_DELETE, 1);
    return image;
}


int main()
{
	coMLB *mymlbFile;
	const char *filepath = "D:/Dropbox/WiSe1819/test.mlb";
	mymlbFile = new coMLB(filepath);
	std::cout << "done." << std::endl;
}
