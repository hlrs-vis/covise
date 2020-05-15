/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description: geometry manager class for COVISE renderer modules        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Dirk Rantzau                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  11.09.95  V1.0                                                  **
\**************************************************************************/

#include "GeometryManager.h"
//#include <Covise_Util.h>
#include "ObjectManager.h"
#include "ObjectList.h"
#include <do/coDoPixelImage.h>

#include <float.h>


//================================================================
// GeometryManager methods
//================================================================

void GeometryManager::boundingBox(float *x, float *y, float *z, int num, float *bbox)
{

    for (int i = 0; i < num; i++)
    {
        if (x[i] < bbox[0])
            bbox[0] = x[i];
        else if (x[i] > bbox[3])
            bbox[3] = x[i];

        if (y[i] < bbox[1])
            bbox[1] = y[i];
        else if (y[i] > bbox[4])
            bbox[4] = y[i];

        if (z[i] < bbox[2])
            bbox[2] = z[i];
        else if (z[i] > bbox[5])
            bbox[5] = z[i];
    }
}

void GeometryManager::boundingBox(float *x, float *y, float *z, float *r, int num, float *bbox)
{

    for (int i = 0; i < num; i++)
    {
        if (x[i] - r[i] < bbox[0])
            bbox[0] = x[i] - r[i];
        else if (x[i] + r[i] > bbox[3])
            bbox[3] = x[i] + r[i];

        if (y[i] - r[i] < bbox[1])
            bbox[1] = y[i] - r[i];
        else if (y[i] + r[i] > bbox[4])
            bbox[4] = y[i] + r[i];

        if (z[i] - r[i] < bbox[2])
            bbox[2] = z[i] - r[i];
        else if (z[i] + r[i] > bbox[5])
            bbox[5] = z[i] + r[i];
    }
}

void unpackRGBA(int *pc, int pos, float *r, float *g, float *b, float *a);

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::addGroup(const char *object, const char *rootName, int is_timestep, int minTimeStep, int maxTimeStep)
{
    (void)object;
    (void)rootName;
    (void)is_timestep;
    (void)minTimeStep;
    (void)maxTimeStep;

    CoviseRender::sendInfo("Adding a new set hierarchy");
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::addUGrid(const char *object, const char *rootName, int xsize, int ysize,
                               int zsize, float xmin, float xmax,
                               float ymin, float ymax, float zmin, float zmax,
                               int no_of_colors, int colorbinding, int colorpacking,
                               float *r, float *g, float *b, int *pc, int no_of_normals,
                               int normalbinding,
                               float *nx, float *ny, float *nz, float transparency, coMaterial *material, coDoTexture *texture)

{
    (void)object;
    (void)rootName;
    (void)xsize;
    (void)ysize;
    (void)zsize;
    (void)xmin;
    (void)xmax;
    (void)ymin;
    (void)ymax;
    (void)zmin;
    (void)zmax;
    (void)no_of_colors;
    (void)colorbinding;
    (void)colorpacking;
    (void)r;
    (void)g;
    (void)b;
    (void)pc;
    (void)no_of_normals;
    (void)normalbinding;
    (void)nx;
    (void)ny;
    (void)nz;
    (void)transparency;
    (void)material;
    (void)texture;

    CoviseRender::sendInfo("Adding a uniform grid");
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::addRGrid(const char *object, const char *rootName, int xsize, int ysize,
                               int zsize, float *x_c, float *y_c,
                               float *z_c,
                               int no_of_colors, int colorbinding, int colorpacking,
                               float *r, float *g, float *b, int *pc, int no_of_normals,
                               int normalbinding,
                               float *nx, float *ny, float *nz, float transparency, coMaterial *material, coDoTexture* texture)

{
    (void)object;
    (void)rootName;
    (void)xsize;
    (void)ysize;
    (void)zsize;
    (void)x_c;
    (void)y_c;
    (void)z_c;
    (void)no_of_colors;
    (void)colorbinding;
    (void)colorpacking;
    (void)r;
    (void)g;
    (void)b;
    (void)pc;
    (void)no_of_normals;
    (void)normalbinding;
    (void)nx;
    (void)ny;
    (void)nz;
    (void)transparency;
    (void)material;
    (void)texture;

    CoviseRender::sendInfo("Adding a rectilinear grid");
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::addSGrid(const char *object, const char *rootName, int xsize, int ysize,
                               int zsize, float *x_c, float *y_c,
                               float *z_c,
                               int no_of_colors, int colorbinding, int colorpacking,
                               float *r, float *g, float *b, int *pc, int no_of_normals,
                               int normalbinding,
                               float *nx, float *ny, float *nz, float transparency, coMaterial *material, coDoTexture* texture)
{
    (void)object;
    (void)rootName;
    (void)xsize;
    (void)ysize;
    (void)zsize;
    (void)x_c;
    (void)y_c;
    (void)z_c;
    (void)no_of_colors;
    (void)colorbinding;
    (void)colorpacking;
    (void)r;
    (void)g;
    (void)b;
    (void)pc;
    (void)no_of_normals;
    (void)normalbinding;
    (void)nx;
    (void)ny;
    (void)nz;
    (void)transparency;
    (void)material;
    (void)texture;

    CoviseRender::sendInfo("Adding a structured grid");
}


void addPixelTexture(coDoTexture* texture, NewCharBuffer& buf)
{
    int sizeu, sizev, sizew;
    float** textureCoords;
    unsigned char* imageData;
    if (texture)
    {
        coDoPixelImage* img = texture->getBuffer();
        imageData = (unsigned char*)(img->getPixels());
        sizeu = img->getWidth();
        sizev = img->getHeight();
        sizew = img->getPixelsize();
		textureCoords = texture->getCoordinates();

		char line[500];
		if (objlist->outputMode == OutputMode::VRML97)
		{
		}
		else
		{
			const char* wm = texture->getAttribute("WRAP_MODE");
			const char* repeateString = "'false'";
			if (wm && strncasecmp(wm, "repeat", 6) == 0)
			{
				repeateString = "'true'";
			}
			const char* minfm = texture->getAttribute("MIN_FILTER");
			const char* magfm = texture->getAttribute("MAG_FILTER");
			buf += "<PixelTexture ";
			buf += "image=";
			sprintf(line, "'%d,%d,%d ", sizeu, sizev, sizew);
			buf += line;
            int uv = 0;
            for (int i = 0; i < sizeu; i++)
            {
                for (int j = 0; j < sizev; j++)
                {
                    uv = ((sizev * i) + j) * sizew;
                    buf += " 0x";
                    for (int k = 0; k < sizew; k++)
                    {
                        sprintf(line, "%02x", imageData[uv+k]);
                        buf += line;
                    }
                }
            }
            buf += "'\n";
			buf += "repeatS=";
			buf += repeateString;
			buf += "repeatT=";
			buf += repeateString;
			buf += ">\n";

			buf += "<TextureProperties\n";
			if (magfm != nullptr)
			{
				buf += "magnificationFilter='\n";
				buf += magfm;
				buf += "'\n";
			}
			if (minfm != nullptr)
			{
				buf += "minificationFilter='\n";
				buf += minfm;
				buf += "'\n";
			}
			buf += "></TextureProperties>\n";

			buf += "</PixelTexture>\n";
		}
    }
}

void addTexture(coDoTexture* texture, NewCharBuffer& buf)
{
    int numTC = 0;
    float** textureCoords;
    if (texture)
    {
        textureCoords = texture->getCoordinates();

        char line[500];
        numTC = texture->getNumCoordinates();
        if (objlist->outputMode == OutputMode::VRML97)
        {
            buf += "texCoord TextureCoordinate { \npoint[\n";

            for (int i = 0; i < numTC; i++)
            {
                sprintf(line, "%1g %1g,", textureCoords[0][i], textureCoords[1][i]);
                buf += line;
                if ((i % 10) == 0)
                    buf += '\n';
            }
            buf += "]}\n";
        }
        else
        {

            buf += "<TextureCoordinate point = '";
            for (int i = 0; i < numTC; i++)
            {
                sprintf(line, "%1g %1g ", textureCoords[0][i], textureCoords[1][i]);
                buf += line;
            }
            buf += "'>\n</TextureCoordinate>\n ";
        }
    }
}
void addBindings(int colorbinding, int normalbinding,NewCharBuffer& buf)
{
    if (objlist->outputMode == OutputMode::VRML97)
    {
        if (colorbinding == CO_PER_VERTEX)
        {
            buf += "colorPerVertex TRUE\n";
        }
        if (normalbinding == CO_PER_VERTEX)
        {
            buf += "normalPerVertex TRUE\n";
        }
    }
    else
    {
        if (colorbinding == CO_PER_VERTEX)
        {
            buf += "colorPerVertex = 'true'\n";
        }
        if (normalbinding == CO_PER_VERTEX)
        {
            buf += "normalPerVertex = 'true'\n";
        }
        buf += ">\n";
    }
}
void addCoordinates(float *x_c, float *y_c, float *z_c, int numCoords, NewCharBuffer& buf)
{
    char line[500];
    if (objlist->outputMode == OutputMode::VRML97)
    {
        buf += "coord Coordinate{ \npoint[\n";
        
        for (int i = 0; i < numCoords; i++)
        {
            sprintf(line, "%1g %1g %1g,", x_c[i], y_c[i], z_c[i]);
            buf += line;
            if ((i % 10) == 0)
                buf += '\n';
        }
        buf += "]}\n";
    }
    else
    {

        buf += "<coordinate point = '";
        for (int i = 0; i < numCoords; i++)
        {
            sprintf(line, "%1g %1g %1g ", x_c[i], y_c[i], z_c[i]);
            buf += line;
        }
        buf += "'>\n</coordinate>\n ";
    }
}

void addNormals(float* nx, float* ny, float* nz, int numNormals, NewCharBuffer& buf)
{
	char line[500];
	if (numNormals > 0)
	{
		if (objlist->outputMode == OutputMode::VRML97)
		{
			buf += "\n normal Normal {\nvector[";
			for (int i = 0; i < numNormals; i++)
			{
				sprintf(line, "%1g %1g %1g,", nx[i], ny[i], nz[i]);
				buf += line;
				if ((i % 10) == 0)
					buf += '\n';
			}
			buf += "]}\n";
		}
		else
		{
			buf += "\n <Normal vector='";
			for (int i = 0; i < numNormals; i++)
			{
				sprintf(line, "%1g %1g %1g ", nx[i], ny[i], nz[i]);
				buf += line;
			}
			buf += "'\n";
			buf += "><Normal>\n";
		}
	}
}
void addColors(float* r, float* g, float* b, int *pc, int numColors, int colorpacking, NewCharBuffer& buf)
{
	char line[500];
	if (numColors > 0)
	{
		if (objlist->outputMode == OutputMode::VRML97)
		{
			buf += "\n color Color {\ncolor[";
			if (colorpacking == CO_RGBA && pc)
			{
				float r, g, b, a;

				for (int i = 0; i < numColors; i++)
				{
					unpackRGBA(pc, i, &r, &g, &b, &a);
					sprintf(line, "%1g %1g %1g,", r, g, b);
					buf += line;
					if ((i % 10) == 0)
						buf += '\n';
				}
			}
			else
			{
				for (int i = 0; i < numColors; i++)
				{
					sprintf(line, "%1g %1g %1g,", r[i], g[i], b[i]);
					buf += line;
					if ((i % 10) == 0)
						buf += '\n';
				}
			}
			buf += "]}\n\n";
		}
		else
		{
			buf += "\n <Color color='";
			if (colorpacking == CO_RGBA && pc)
			{
				float r, g, b, a;

				for (int i = 0; i < numColors; i++)
				{
					unpackRGBA(pc, i, &r, &g, &b, &a);
					sprintf(line, "%1g %1g %1g ", r, g, b);
					buf += line;
				}
			}
			else
			{
				for (int i = 0; i < numColors; i++)
				{
					sprintf(line, "%1g %1g %1g,", r[i], g[i], b[i]);
					buf += line;
					if ((i % 10) == 0)
						buf += '\n';
				}
			}
			buf += "'></Color>\n";
		}
    }
}

void GeometryManager::addMaterial(coMaterial* material, int colorbinding, int colorpacking, float* r, float* g, float* b, int* pc, coDoTexture *texture, NewCharBuffer& buf, const char* object)
{
    char line[500];
    if (objlist->outputMode == OutputMode::VRML97)
    {
        buf += "Shape{  appearance Appearance { material Material {";
    }
    else
    {
        buf += "<shape>\n<appearance>\n<material ";
        buf += "id='";
        sprintf(line, "%s'", object);
        buf += line;
    }
    if (material)
    {

        if (objlist->outputMode == OutputMode::VRML97)
        {
            buf += "diffuseColor ";
            sprintf(line, " %1g %1g %1g ", material->diffuseColor[0], material->diffuseColor[1], material->diffuseColor[2]);
            buf += line;
            buf += "emissiveColor ";
            sprintf(line, " %1g %1g %1g ", material->emissiveColor[0], material->emissiveColor[1], material->emissiveColor[2]);
            buf += line;
            buf += "ambientIntensity ";
            sprintf(line, " %1g ", (material->ambientColor[0] + material->ambientColor[1] + material->ambientColor[2]) / 3.0);
            buf += line;
            buf += "specularColor ";
            sprintf(line, " %1g %1g %1g ", material->specularColor[0], material->specularColor[1], material->specularColor[2]);
            buf += line;
            buf += "transparency ";
            sprintf(line, " %1g ", material->transparency);
            buf += line;
            buf += "shininess ";
            sprintf(line, " %1g ", material->shininess);
            buf += line;
        }
        else
        {
            buf += "diffuseColor='";
            sprintf(line, "%1g %1g %1g'", material->diffuseColor[0], material->diffuseColor[1], material->diffuseColor[2]);
            buf += line;
            buf += " emissiveColor='";
            sprintf(line, "%1g %1g %1g'", material->emissiveColor[0], material->emissiveColor[1], material->emissiveColor[2]);
            buf += line;
            buf += " ambientIntensity='";
            sprintf(line, "%1g'", (material->ambientColor[0] + material->ambientColor[1] + material->ambientColor[2]) / 3.0);
            buf += line;
            buf += " specularColor='";
            sprintf(line, "%1g %1g %1g'", material->specularColor[0], material->specularColor[1], material->specularColor[2]);
            buf += line;
            buf += " transparency='";
            sprintf(line, "%1g'", material->transparency);
            buf += line;
            buf += " shininess='";
            sprintf(line, "%1g'", material->shininess);
            buf += line;
        }
    }
    else if (colorbinding == CO_PER_VERTEX)
    {
    }
    else
    {
        if (objlist->outputMode == OutputMode::VRML97)
        {
            buf += "diffuseColor ";
            if (colorpacking == CO_RGBA && pc)
            {
                float r, g, b, a;

                unpackRGBA(pc, 0, &r, &g, &b, &a);
                sprintf(line, " %1g %1g %1g ", r, g, b);
            }
            else if (colorbinding != CO_NONE && r)
            {
                sprintf(line, " %1g %1g %1g ", r[0], g[0], b[0]);
            }
            else
            {
                sprintf(line, " 1 1 1 ");
            }
            buf += line;
        }
        else
        {
            buf += " diffuseColor='";
            if (colorpacking == CO_RGBA && pc)
            {
                float r, g, b, a;

                unpackRGBA(pc, 0, &r, &g, &b, &a);
                sprintf(line, "%1g %1g %1g'", r, g, b);
            }
            else if (colorbinding != CO_NONE && r)
            {
                sprintf(line, "%1g %1g %1g'", r[0], g[0], b[0]);
            }
            else
            {
                sprintf(line, "1 1 1'");
            }
            buf += line;
        }

    }
    if (objlist->outputMode == OutputMode::VRML97)
    {
        buf += "}";
    }
    else
    {
        buf += "></material> ";
    }
    addPixelTexture(texture, buf);

    if (objlist->outputMode == OutputMode::VRML97)
    {
        buf += "}";
    }
    else
    {
        buf += "</appearance>";
    }

}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::addPolygon(const char *object, const char *rootName, int no_of_polygons, int no_of_vertices,
                                 int no_of_coords, float *x_c, float *y_c,
                                 float *z_c,
                                 int *v_l, int *l_l,
                                 int no_of_colors, int colorbinding, int colorpacking,
                                 float *r, float *g, float *b, int *pc, int no_of_normals,
                                 int normalbinding,
                                 float *nx, float *ny, float *nz, float transparency,
                                 int vertexOrder, coMaterial *material, coDoTexture* texture)
{
    (void)vertexOrder;
    (void)transparency;
    float bbox[6] = { FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX };

    NewCharBuffer buf(10000);
    char line[500];
    int i;
    int n = 0;

    addMaterial(material, colorbinding, colorpacking,r,g,b,pc,texture, buf, object);

    if (objlist->outputMode == OutputMode::VRML97)
    {
        buf += "geometry IndexedFaceSet {\n";
        buf += "solid FALSE\nccw TRUE\nconvex TRUE\n";
    }
    else
    {
        buf += "<indexedFaceSet\n";
        buf += "solid='false'\nccw='true'\nconvex='true'\n";
    }
    if (objlist->outputMode == OutputMode::VRML97)
    {
        buf += "coordIndex [";
        n = 0;
        for (i = 0; i < no_of_vertices; i++)
        {
            if ((n < (no_of_polygons - 1)) && (l_l[n + 1] == i))
            {
                buf += "-1 ";
                n++;
            }
            sprintf(line, "%d ", v_l[i]);
            buf += line;
            if ((i % 10) == 0)
                buf += '\n';
        }
        buf += ']';
    }
    else
    {
        buf += "coordIndex='";
        n = 0;
        for (i = 0; i < no_of_vertices; i++)
        {
            if ((n < (no_of_polygons - 1)) && (l_l[n + 1] == i))
            {
                buf += "-1 ";
                n++;
            }
            sprintf(line, "%d ", v_l[i]);
            buf += line;
        }
        buf += "'\n";

    }

    addBindings(colorbinding, normalbinding, buf);

    addCoordinates(x_c,y_c,z_c,no_of_coords,buf);
    addNormals(nx, ny, nz, no_of_normals,buf);
    addColors(r, g, b, pc, no_of_colors, colorpacking,buf);
    addTexture(texture, buf);


    if (objlist->outputMode == OutputMode::VRML97)
    {
        buf += "} }\n";
    }
    else
    {
        buf += "</indexedFaceSet> </shape>\n";
    }

    boundingBox(x_c, y_c, z_c, no_of_coords, bbox);

    NewCharBuffer *newbuf = new NewCharBuffer(&buf);
    LObject *lob = new LObject();
    lob->set_boundingbox(bbox);
    lob->set_name((char *)object);
    lob->set_timestep(object);
    lob->set_rootname((char *)rootName);
    lob->set_objPtr((void *)newbuf);
    objlist->push_back(std::unique_ptr<LObject>(lob));
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::addTriangleStrip(const char *object, const char *rootName, int no_of_strips, int no_of_vertices,
                                       int no_of_coords, float *x_c, float *y_c,
                                       float *z_c,
                                       int *v_l, int *s_l,
                                       int no_of_colors, int colorbinding, int colorpacking,
                                       float *r, float *g, float *b, int *pc, int no_of_normals,
                                       int normalbinding,
                                       float *nx, float *ny, float *nz, float transparency,
                                       int vertexOrder, coMaterial *material, coDoTexture* texture)
{

    (void)transparency;
    (void)vertexOrder;
    float bbox[6] = { FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX };

    NewCharBuffer buf(10000);
    char line[500];
    int i;
    int n = 0;

    addMaterial(material, colorbinding, colorpacking, r, g, b, pc, texture, buf, object);

    if (objlist->outputMode == OutputMode::VRML97)
    {
        buf += "geometry IndexedFaceSet {\n";
        buf += "solid FALSE\nccw TRUE\nconvex TRUE\n";
    }
    else
    {
        buf += "<indexedFaceSet\n";
        buf += "solid='false'\nccw='true'\nconvex='true'\n";
    }
    if (objlist->outputMode == OutputMode::VRML97)
    {
        buf += "coordIndex [";

        n = 0;
        for (i = 0; i < no_of_strips; i++)
        {

            if (i == (no_of_strips - 1))
            {
                for (n = 2; n < (no_of_vertices - s_l[i]); n++)
                {
                    if (n % 2)
                        sprintf(line, "%d %d %d -1 ", v_l[s_l[i] + n - 1], v_l[s_l[i] + n - 2], v_l[s_l[i] + n]);
                    else
                        sprintf(line, "%d %d %d -1 ", v_l[s_l[i] + n - 2], v_l[s_l[i] + n - 1], v_l[s_l[i] + n]);
                    buf += line;
                }
            }
            else
            {
                for (n = 2; n < (s_l[i + 1] - s_l[i]); n++)
                {
                    if (n % 2)
                        sprintf(line, "%d %d %d -1 ", v_l[s_l[i] + n - 1], v_l[s_l[i] + n - 2], v_l[s_l[i] + n]);
                    else
                        sprintf(line, "%d %d %d -1 ", v_l[s_l[i] + n - 2], v_l[s_l[i] + n - 1], v_l[s_l[i] + n]);
                    buf += line;
                }
            }
            if ((i % 5) == 0)
                buf += '\n';
        }
        buf += ']';
    }
    else
    {
        buf += "coordIndex='";
        n = 0;
        for (i = 0; i < no_of_strips; i++)
        {

            if (i == (no_of_strips - 1))
            {
                for (n = 2; n < (no_of_vertices - s_l[i]); n++)
                {
                    if (n % 2)
                        sprintf(line, "%d %d %d -1 ", v_l[s_l[i] + n - 1], v_l[s_l[i] + n - 2], v_l[s_l[i] + n]);
                    else
                        sprintf(line, "%d %d %d -1 ", v_l[s_l[i] + n - 2], v_l[s_l[i] + n - 1], v_l[s_l[i] + n]);
                    buf += line;
                }
            }
            else
            {
                for (n = 2; n < (s_l[i + 1] - s_l[i]); n++)
                {
                    if (n % 2)
                        sprintf(line, "%d %d %d -1 ", v_l[s_l[i] + n - 1], v_l[s_l[i] + n - 2], v_l[s_l[i] + n]);
                    else
                        sprintf(line, "%d %d %d -1 ", v_l[s_l[i] + n - 2], v_l[s_l[i] + n - 1], v_l[s_l[i] + n]);
                    buf += line;
                }
            }
        }
        buf += "'\n";

    }

    addBindings(colorbinding, normalbinding, buf);

    addCoordinates(x_c, y_c, z_c, no_of_coords, buf);
    addNormals(nx, ny, nz, no_of_normals, buf);
    addColors(r, g, b, pc, no_of_colors, colorpacking, buf);
    addTexture(texture, buf);


    if (objlist->outputMode == OutputMode::VRML97)
    {
        buf += "} }\n";
    }
    else
    {
        buf += "</indexedFaceSet> </shape>\n";
    }


    boundingBox(x_c, y_c, z_c, no_of_coords, bbox);

    NewCharBuffer *newbuf = new NewCharBuffer(&buf);
    LObject *lob = new LObject(object, rootName, newbuf);
    lob->set_boundingbox(bbox);
    lob->set_timestep(object);

    objlist->push_back(std::unique_ptr<LObject>(lob));
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::addLine(const char *object, const char *rootName, int no_of_lines, int no_of_vertices,
                              int no_of_coords, float *x_c, float *y_c,
                              float *z_c,
                              int *v_l, int *l_l,
                              int no_of_colors, int colorbinding, int colorpacking,
                              float *r, float *g, float *b, int *pc, int no_of_normals,
                              int normalbinding,
                              float *nx, float *ny, float *nz, int isTrace, coMaterial *material, coDoTexture* texture)

{
    (void)no_of_normals;
    (void)normalbinding;
    (void)nx;
    (void)ny;
    (void)nz;
    (void)isTrace;

    float bbox[6] = { FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX };

    NewCharBuffer buf(10000);
    char line[500];
    int i;
    int n = 0;


    addMaterial(material, colorbinding, colorpacking, r, g, b, pc, texture, buf, object);

    if (objlist->outputMode == OutputMode::VRML97)
    {
        buf += "geometry IndexedLineSet {\n";
        buf += "solid FALSE\nccw TRUE\nconvex TRUE\n";
    }
    else
    {
        buf += "<indexedLineSet\n";
        buf += "solid='false'\nccw='true'\nconvex='true'\n";
    }
    if (objlist->outputMode == OutputMode::VRML97)
    {
        buf += "coordIndex [";
        n = 0;
        for (i = 0; i < no_of_vertices; i++)
        {
            if ((n < (no_of_lines - 1)) && (l_l[n + 1] == i))
            {
                buf += "-1 ";
                n++;
            }
            sprintf(line, "%d ", v_l[i]);
            buf += line;
            if ((i % 10) == 0)
                buf += '\n';
        }
        buf += ']';
    }
    else
    {
        buf += "coordIndex='";
        n = 0;
        for (i = 0; i < no_of_vertices; i++)
        {
            if ((n < (no_of_lines - 1)) && (l_l[n + 1] == i))
            {
                buf += "-1 ";
                n++;
            }
            sprintf(line, "%d ", v_l[i]);
            buf += line;
        }
        buf += "'\n";

    }

    addBindings(colorbinding, normalbinding, buf);

    addCoordinates(x_c, y_c, z_c, no_of_coords, buf);
    addNormals(nx, ny, nz, no_of_normals, buf);
    addColors(r, g, b, pc, no_of_colors, colorpacking, buf);
    addTexture(texture, buf);


    if (objlist->outputMode == OutputMode::VRML97)
    {
        buf += "} }\n";
    }
    else
    {
        buf += "</indexedLineSet> </shape>\n";
    }



    NewCharBuffer *newbuf = new NewCharBuffer(&buf);
    boundingBox(x_c, y_c, z_c, no_of_coords, bbox);
    LObject *lob = new LObject(object, rootName, newbuf);
    lob->set_boundingbox(bbox);
    lob->set_timestep(object);
    objlist->push_back(std::unique_ptr<LObject>(lob));
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::addPoint(const char *object, const char *rootName, int no_of_points,
                               float *x_c, float *y_c, float *z_c,
                               int colorbinding, int colorpacking,
                               float *r, float *g, float *b, int *pc, coMaterial *material, coDoTexture* texture)

{
    NewCharBuffer buf(10000);

    float bbox[6] = { FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX };


    addMaterial(material, colorbinding, colorpacking, r, g, b, pc, texture, buf, object);

    if (objlist->outputMode == OutputMode::VRML97)
    {
        buf += "geometry PointSet {\n";
        buf += "solid FALSE\nccw TRUE\nconvex TRUE\n";
    }
    else
    {
        buf += "<pointSet\n";
        buf += "solid='false'\nccw='true'\nconvex='true'\n";
    }
    addBindings(colorbinding, CO_NONE, buf);

    addCoordinates(x_c, y_c, z_c, no_of_points, buf);
    if(colorbinding == CO_PER_VERTEX)
    {
		addColors(r, g, b, pc, no_of_points, colorpacking, buf);
    }
    addTexture(texture, buf);


    if (objlist->outputMode == OutputMode::VRML97)
    {
        buf += "} }\n";
    }
    else
    {
        buf += "</pointSet> </shape>\n";
    }


    boundingBox(x_c, y_c, z_c, no_of_points, bbox);
    NewCharBuffer *newbuf = new NewCharBuffer(&buf);
    LObject *lob = new LObject(object, rootName, newbuf);
    lob->set_boundingbox(bbox);
    lob->set_timestep(object);

    objlist->push_back(std::unique_ptr<LObject>(lob));
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::addSphere(const char *object, const char *rootName, int no_of_points,
                                float *x_c, float *y_c, float *z_c, float *r_c,
                                int colorbinding, int colorpacking,
                                float *r, float *g, float *b, int *pc, coMaterial *material, coDoTexture* texture)

{
    NewCharBuffer buf(10000);
    char line[500];
    float bbox[6] = { FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX };

    if (objlist->outputMode == OutputMode::VRML97)
    {
        buf += "Group { children [\n";
    }
    else
    {
        buf += "<transform>\n";
    }
    for (int i = 0; i < no_of_points; ++i)
    {
        if (objlist->outputMode == OutputMode::VRML97)
        {
            buf += "Transform { translation ";
            sprintf(line, "%g %g %g", x_c[i], y_c[i], z_c[i]);
            buf += line;
            buf += " children [";
        }
        else
        {
            buf += "<transform translation='\n";
            sprintf(line, "%g %g %g '>", x_c[i], y_c[i], z_c[i]);
            buf += line;
        }



        addMaterial(material, colorbinding, colorpacking, r, g, b, pc, texture, buf, object);

        if (objlist->outputMode == OutputMode::VRML97)
        {
            buf += " geometry Sphere { radius ";
            sprintf(line, "%g", r_c[i]);
            buf += line;
            buf += " }";
            buf += "} ] }\n"; // Shape + Transform
        }
        else
        {
            buf += "<sphere radius='";
            sprintf(line, "%g'>", r_c[i]);
            buf += line;
            buf += "</shape> </transform>";
        }


    }
    if (objlist->outputMode == OutputMode::VRML97)
    {
        buf += "] }\n";
    }
    else
    {
        buf += "</transform>";
    }

    boundingBox(x_c, y_c, z_c, r_c, no_of_points, bbox);
    NewCharBuffer *newbuf = new NewCharBuffer(&buf);
    LObject *lob = new LObject(object, rootName, newbuf);
    lob->set_boundingbox(bbox);
    lob->set_timestep(object);

    objlist->push_back(std::unique_ptr<LObject>(lob));
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::replaceUGrid(const char *, int, int,
                                   int, float, float,
                                   float, float, float, float,
                                   int, int, int,
                                   float *, float *, float *, int *, int,
                                   int,
                                   float *, float *, float *, float)

{

    CoviseRender::sendInfo("Replacing uniform grid");
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::replaceRGrid(const char *, int, int,
                                   int, float *, float *,
                                   float *,
                                   int, int, int,
                                   float *, float *, float *, int *, int,
                                   int,
                                   float *, float *, float *, float)

{

    CoviseRender::sendInfo("Replacing rectilinear grid");
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------

void GeometryManager::replaceSGrid(const char *, int, int,
                                   int, float *, float *,
                                   float *,
                                   int, int, int,
                                   float *, float *, float *, int *, int,
                                   int,
                                   float *, float *, float *, float)
{

    CoviseRender::sendInfo("Replacing structured grid");
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::replacePolygon(const char *, int, int,
                                     int, float *, float *,
                                     float *,
                                     int *, int *,
                                     int, int, int,
                                     float *, float *, float *, int *, int,
                                     int,
                                     float *, float *, float *, float,
                                     int)
{

    CoviseRender::sendInfo("Replacing polygons");
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::replaceTriangleStrip(const char *, int, int,
                                           int, float *, float *,
                                           float *,
                                           int *, int *,
                                           int, int, int,
                                           float *, float *, float *, int *, int,
                                           int,
                                           float *, float *, float *, float,
                                           int)
{

    CoviseRender::sendInfo("Replacing triangle strips");
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::replaceLine(const char *, int, int,
                                  int, float *, float *,
                                  float *,
                                  int *, int *,
                                  int, int, int,
                                  float *, float *, float *, int *, int,
                                  int,
                                  float *, float *, float *)

{

    CoviseRender::sendInfo("Replacing lines");
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::replacePoint(const char *, int,
                                   float *, float *, float *,
                                   int, int,
                                   float *, float *, float *, int *)

{

    CoviseRender::sendInfo("Replacing points");
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::replaceSphere(const char *, int,
                                    float *, float *, float *, float *,
                                    int, int,
                                    float *, float *, float *, int *)

{

    CoviseRender::sendInfo("Replacing spheres");
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::remove_geometry(const char *name)
{
    objlist->removeone(name);
}

void unpackRGBA(int *pc, int pos, float *r, float *g, float *b, float *a)
{

    unsigned char *chptr;

    chptr = (unsigned char *)&pc[pos];
// RGBA switched 12.03.96 due to color bug in Inventor Renderer
// D. Rantzau

#ifdef BYTESWAP
    *a = ((float)(*chptr)) / 255.0f;
    chptr++;
    *b = ((float)(*chptr)) / 255.0f;
    chptr++;
    *g = ((float)(*chptr)) / 255.0f;
    chptr++;
    *r = ((float)(*chptr)) / 255.0f;
#else
    *r = ((float)(*chptr)) / 255.0;
    chptr++;
    *g = ((float)(*chptr)) / 255.0;
    chptr++;
    *b = ((float)(*chptr)) / 255.0;
    chptr++;
    *a = ((float)(*chptr)) / 255.0;
#endif
}
