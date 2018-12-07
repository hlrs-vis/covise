/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coMLB.h"
#include <iostream>
#include <fstream>
 
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
	std::cout << "coMLB::readData()" << std::endl;
	//MLBHeader myHeader;
	double tmpf;
	int numValues;
	//std::vector<double> data;

	int fd = open(fileName.c_str(), std::ios::binary); //std::fstream::in | std::fstream::binary);

	std::fstream fh;
	fh.open(fileName.c_str(), std::fstream::in | std::fstream::binary);
	fh.read((char*)&myHeader, sizeof(MLBHeader));

	std::cout << "read ^header:" << std::endl;
	numValues = myHeader.t_width * myHeader.t_height * myHeader.t_depth;
	// unsigned char *data = new unsigned char[numValues];  geht nicht---?

	//int N = 10;
	//unsigned char *data = new unsigned char[numValues];
	std::cout << "read data:" << std::endl;
	data.reserve(numValues);
	unsigned char tmpc;
	int a_number;
	float min = 1000000, max = 0, range;
	for (int i = 0; i < numValues; i++)
	{
		fh.read((char*)&tmpc, sizeof(tmpc));
		data.push_back(tmpc);
		{
			if (data[i]< min)
				min = data[i];
			if (data[i]> max)
				max = data[i];
		}
		if ((i<10) || (i % 1000000) == 0)
		{
			a_number = (int)(tmpc);
			std::cout << i << " (of " << numValues << "): " << a_number << std::endl;
			std::cout << "(coMLB) getTexture min / max: " << min << " / " << max << "\n";
		}
	}
	/*
	candela.reserve(numValues);
	for (int i = 0; i < numValues; i++)
	{
		fh.read((char*)&tmpf, sizeof(tmpf));
		if ((i % 100000) == 0)
		{
			std::cout << i << " (of " << numValues << "): " << tmpf << std::endl;
		}
		candela.push_back(tmpf);
	}*/
	fh.close();
	std::cout << "got struct MLBHeader with members: " << std::endl;
	std::cout << "left: " << myHeader.left << std::endl;
	std::cout << "bottom: " << myHeader.bottom << std::endl;
	std::cout << "width: " << myHeader.width << std::endl;
	std::cout << "height: " << myHeader.height << std::endl;
	std::cout << "t_width: " << myHeader.t_width << std::endl;
	std::cout << "t_height: " << myHeader.t_height << std::endl;
	std::cout << "t_depth: " << myHeader.t_depth << std::endl;
	std::cout << "data: " << (int)(data[0]) << std::endl;
	std::cout << "data: " << (int)(data[1]) << std::endl;
	std::cout << "data: " << (int)(data[numValues-1]) << std::endl;

	left = myHeader.left;
	bottom = myHeader.bottom;
	width = myHeader.width;
	height = myHeader.height;

	t_width = myHeader.t_width;
	t_height = myHeader.t_height;
	t_depth = myHeader.t_depth;
	if (fd == -1)
	{
		std::cout << "can't open file";
		return false;
	}
	//https://linux.die.net/man/2/read
	//read(fd, &myHeader, sizeof(MLBHeader));

	
	
    return true;
}


osg::Image *coMLB::getTexture()
{
	std::cout << "getTexture(): " << std::endl;
	int WIDTH = myHeader.t_width;
	int HEIGHT = myHeader.t_height;
	int DEPTH = myHeader.t_depth;
    osg::Image *image = new osg::Image();
    float min=1000000, max=0, range;
	std::cout << myHeader.t_width << ", " << myHeader.t_height << ", " << myHeader.t_depth << std::endl;
	int numValues = WIDTH * HEIGHT * DEPTH;
	std::cout << "data[" << numValues << "]" << std::endl;
	std::cout << "data: " << (int)(data[0]) << std::endl;
	std::cout << "data: " << (int)(data[1]) << std::endl;
	std::cout << "data: " << (int)(data[numValues - 1]) << std::endl;

	/*for (int i=0; i<numValues; i++)
    {
        if(candela[i]< min)
            min = candela[i];
        if(candela[i]> max)
            max = candela[i];
    }
    range = max - min;
	std::cout << "(coMLB) getTexture min / max: " << min << " / " << max << "\n";
    
    unsigned char *data = new unsigned char[numValues];
	std::cout << myHeader.t_width << " : " << myHeader.t_height << std::endl;
    for (int i=0; i<numValues; i++)
    { 
		data[i] = ((candela[i] - min) / range) * 255;  // x + WIDTH * (y + DEPTH * z)
    }*/

   /*  for (int i=0;i<numValues;i++)
    {
        data[i] = ((candela[i]-min)/range)*255;
    }*/
	unsigned char *data_array = &data[0];
    osg::ref_ptr<osg::Image> texImgRG = new osg::Image();
    image->setImage(myHeader.t_width, myHeader.t_height, myHeader.t_depth, 1,
		GL_LUMINANCE, GL_UNSIGNED_BYTE, data_array, osg::Image::USE_NEW_DELETE, 1);
    return image;
}


int main()
{
	coMLB *mymlbFile;
	const char *filepath = "D:/Dropbox/WiSe1819/test.mlb";
	mymlbFile = new coMLB(filepath);
	std::cout << mymlbFile << std::endl;
}