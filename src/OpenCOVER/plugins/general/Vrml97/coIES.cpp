/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "coIES.h"

coIES::coIES(std::string fn)
{	
    std::cout << "(coIES) raeading...: " << fn << "\n";
    fileName= fn;
    readData();
}
coIES::~coIES()
{
}
bool coIES::readData()
{
    fp=fopen(fileName.c_str(), "r");
    if(fp!=NULL)
    {
        char buf[1000];
        while(fgets(buf,1000,fp)!=NULL)
        {
            if(strncmp(buf,"TILT=",5)==0)
            {
                if(strncmp(buf+5,"NONE",4)==0)
                {
                    break; // nothing to do
                }
                else if(strncmp(buf+5,"INCLUDE",7)==0)
                {
                    if(fgets(buf,1000,fp)!=NULL)
                    {
                        sscanf(buf,"%d",&lampToLuminaireGeometry);
                    }
                    if(fgets(buf,1000,fp)!=NULL)
                    {
                        sscanf(buf,"%d",&numAnglesAndMultiplyingFactors);
                        tiltAngles.resize(numAnglesAndMultiplyingFactors);
                        tiltFactors.resize(numAnglesAndMultiplyingFactors);
                    }
                    if(fgets(buf,1000,fp)!=NULL)
                    {
                        char *tmpc = buf;
                        float tmpf;
                        int i=0;
                        do {
                            sscanf(tmpc,"%f",&tmpf);
                            tiltAngles[i]=tmpf;
                            while(*tmpc!='\0' && *tmpc!=' ')
                                tmpc++;
                            while(*tmpc!='\0' && *tmpc==' ')
                                tmpc++;
                            i++;
                        } while(i<numAnglesAndMultiplyingFactors && *tmpc!='\0');
                    }
                    if(fgets(buf,1000,fp)!=NULL)
                    {
                        char *tmpc = buf;
                        float tmpf;
                        int i=0;
                        do {
                            sscanf(tmpc,"%f",&tmpf);
                            tiltFactors[i]=tmpf;
                            while(*tmpc!='\0' && *tmpc!=' ')
                                tmpc++;
                            while(*tmpc!='\0' && *tmpc==' ')
                                tmpc++;
                            i++;
                        } while(i<numAnglesAndMultiplyingFactors && *tmpc!='\0');
                    }
                    break;
                }
                // TODO read include files
                fprintf(stderr,"WARNING: TILT files are not supported yet\n");
                break;
            }
        }
        if(fgets(buf,1000,fp)!=NULL)
        {
            int numRead = sscanf(buf,"%d %f %f %d %d %d %d %f %f %f",&numLamps,&lumensPerLamp,&multiplier,&numVerticalAngles,&numHorizontalAngles,&photometricType,&unitsType,&width,&length,&height);
            std::cout << "numLamps: " << numLamps << std::endl;
	    std::cout << "lumensPerLamp: " << lumensPerLamp << std::endl;
	    std::cout << "multiplier: " << multiplier << std::endl;
	    std::cout << "numVerticalAngles: " << numVerticalAngles << std::endl;
	    std::cout << "numHorizontalAngles: " << numHorizontalAngles << std::endl;
	    std::cout << "photometricType: " << photometricType << std::endl;
	    std::cout << "unitsType: " << unitsType << std::endl;
	    std::cout << "width: " << width << std::endl;
	    std::cout << "length: " << length << std::endl;
	    std::cout << "height: " << height << std::endl;
            if(numRead == 3)
            {
                if(fgets(buf,1000,fp)!=NULL)
                {
                    numRead = sscanf(buf,"%d %d %d %d",&numVerticalAngles,&numHorizontalAngles,&photometricType,&unitsType);
                    if(numRead != 4)
                    {
                        fprintf(stderr,"Error reading header 2 after TILT\n");
                        return false;
                    }
                    if(fgets(buf,1000,fp)!=NULL)
                    {
                        numRead = sscanf(buf,"%f %f %f",&width,&length,&height);
                        if(numRead != 3)
                        {
                            fprintf(stderr,"Error reading header 3 after TILT\n");
                            return false;
                        }
                    }
                }
            }
            else if(numRead != 10)
            {
                fprintf(stderr,"Error reading header after TILT\n");
                return false;
            }
        }
        if(fgets(buf,1000,fp)!=NULL)
        {
            int numRead = sscanf(buf,"%f %f %f",&ballastFactor,&ballastLampPhotometricFactor,&inputWatts);
            if(numRead != 3)
            {
                fprintf(stderr,"Error reading ballast header\n");
                return false;
            }
        }
        //read Angles
        readFloats(vAngles,numVerticalAngles);
        readFloats(hAngles,numHorizontalAngles);
        readFloats(candela,numHorizontalAngles*numVerticalAngles);
	std::cout << "vAngles range from " << vAngles[0] << " to " << vAngles[numVerticalAngles - 1] << "\n";
	std::cout << "hAngles range from " << hAngles[0] << " to " << hAngles[numHorizontalAngles - 1] << "\n";
    }
    else
    {
        fprintf(stderr,"Could not open %s\n",fileName.c_str());
        return false;
    }
    return true;
}

bool coIES::readFloats(std::vector<float> &arr, int numValues)
{
    arr.reserve(numValues);
    char buf[1000];
    int i=0;
    while(i<numValues && fgets(buf,1000,fp)!=NULL)
    {
        char *tmpc = buf;
        float tmpf;
        do {
            sscanf(tmpc,"%f",&tmpf);
            arr.push_back(tmpf);
            while(*tmpc!='\0' && *tmpc!='\r'&& *tmpc!='\n' && *tmpc!='\t' && *tmpc!=' ')
                tmpc++;
            while(*tmpc!='\0' && (*tmpc==' ' || *tmpc=='\r' || *tmpc=='\n' || *tmpc=='\t'))
                tmpc++;
            i++;
        } while(i<numValues && *tmpc!='\0');
    }
    if(i<numValues)
        return false;
    return true;
}

osg::Image *coIES::getTexture()
{
    osg::Image *image = new osg::Image();
    float min=1000000, max=0, range;
    
    int numValues = numHorizontalAngles * numVerticalAngles;
    for (int i=0; i<numValues; i++)
    {
        if(candela[i]< min)
            min = candela[i];
        if(candela[i]> max)
            max = candela[i];
    }
    range = max - min;
    std::cout << "(coIES) getTexture min / max: " << min << " / " << max << "\n";
        
    unsigned char *data = new unsigned char[numValues];
	std::cout << numHorizontalAngles << " : " << numVerticalAngles << std::endl;
    for (int x=0; x<numHorizontalAngles; x++)
    {
        for(int y=0 ; y<numVerticalAngles; y++)
        {
            data[x + y*numHorizontalAngles] = ((candela[y+x*numVerticalAngles]-min)/range)*255;
	    //std::cout << x + y * numHorizontalAngles << ": " << data[x + y * numHorizontalAngles] << std::endl;
        }
    }

   /*  for (int i=0;i<numValues;i++)
    {
        data[i] = ((candela[i]-min)/range)*255;
    }*/
    osg::ref_ptr<osg::Image> texImgRG = new osg::Image();
    image->setImage(numHorizontalAngles, numVerticalAngles, 1, 1,
		GL_LUMINANCE, GL_UNSIGNED_BYTE, data, osg::Image::USE_NEW_DELETE, 1);
    return image;
}

// make image_all holding all textures
// make tmp texture -> read all textures, write them into im
