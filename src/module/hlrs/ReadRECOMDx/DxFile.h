/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __DXFILE_DEFINED
#define __DXFILE_DEFINED

#include <util/coviseCompat.h>
/** an object of this class can
    reads in dx-Files and assembles them into
    coordinates, connections, and data for COVISE
 */
class DxFile
{
    ifstream input_;
    bool selfContained_;
    int dataStart_;
    bool valid_;

public:
    /// constructor
    DxFile(const char *filename, bool selfContained);
    /**
       * @param filename the name of the file to be read in
       * @param selfContained determines whether the
         data are part of a dx-File or they are in a separate file
      */
    ~DxFile();

    /** Data in dx files begin after an "end" tag
       * which is alone in a single line
       * this method seeks the position of "end" in the file where
       * data start.
       */
    bool seekEnd();
    /** Set the position in the file
       * where to begin reading
       * @param pos starting position in the file
       */
    void setPos(int pos);

    bool isValid()
    {
        return valid_;
    }
    /** read the coordinates of an unstructured grid
       * @param x_coord array providing space for the x coordinates
       * @param xScale scaling factor for resizing the grid in x direction
       * @param y_coord array providing space for the y coordinates
       * @param yScale scaling factor for resizing the grid in y direction
       * @param z_coord array providing space for the z coordinates
       * @param zScale scaling factor for resizing the grid in z direction
       * @param offset position in the file where to start reading
       * @param items number of coordinates to be read in
       * @param byteOrder Byteorder of the data to be read in
                    ( not of the machine the code is running on )
      */

    void readCoords(float *x_coord,
                    float xScale,
                    float *y_coord,
                    float yScale,
                    float *z_coord,
                    float zScale,
                    int offset,
                    int items,
                    int byteOrder);
    /** read the connections for the unstructured grid
       * @param connections integer array providing space for the connections
       * @param offset position in the file where to start reading
       * @param shape shape of the grid i.e. the number of vertices
                a grid element has; e.g. 8 for cubes
       * @param byteOrder Byteorder of the data to be read in
                    ( not of the machine the code is running on )
       */
    void readConnections(int *connections,
                         int offset,
                         int shape,
                         int items,
                         int byteOrder);
    /** @param data floating point arrays providing space for data to
          be read in. The number of the arrays equals to shape
       * @param offset position in the file where to start reading
       * @param shape number of values per vertex
       * @param items number of vertices to get data for
       * @param byteOrder Byteorder of the data to be read in
       */
    void readData(float **data,
                  int offset,
                  int shape,
                  int items,
                  int byteOrder, float &min, float &max);
};
#endif
