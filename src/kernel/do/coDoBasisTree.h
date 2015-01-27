/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_DO_BASIS_TREE_H
#define CO_DO_BASIS_TREE_H

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS BasisTree
//
//  Basis class for OctTrees and QuadTrees
//
//  Initial version: 2001-12-10 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2001 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#include "coDistributedObject.h"
#include <vector>
#include <util/coVector.h>

namespace covise
{

class DOEXPORT coDoBasisTree : public coDistributedObject
{
public:
    enum // number of shared-memory objects
    {
        SHM_OBJ = 8,
        MAX_NO_LEVELS = 10, // maximum tree depth
        MIN_SMALL_ENOUGH = 32, // minimum goal for macro-cell division
        CRIT_LEVEL = 3, // after this level tree division is interrupted
        // if a son leaf inherits all cells from the father
        NORMAL_SIZE = 800 // population per tree if the grid were uniform
    };
    /** Constructor
       * @param n objedct name
       * @param label1 either OCTRE or OCTREP
       * @param label2 another label
       * @param nelem_l number of cells
       * @param nconn_l number of vertices
       * @param ncoord_l number of points
       * @param el cell array
       * @param conn vertices array
       * @param x_c X-coordinate array
       * @param y_c Y-coordinate array
       * @param z_c Z-coordinate array
       * @param normal_size determines how many octrees are created
       * @param max_no_levels determines the maximum number of octree levels
       * @param min_small_enough criteria for stopping octree division
       * @param limit_fX limits number of octrees
       * @param limit_fY limits number of octrees
       * @param limit_fZ limits number of octrees
       */
    coDoBasisTree(const coObjInfo &info,
                  const char *label1, const char *label2,
                  int nelem_l, int nconn_l, int ncoord_l,
                  int *el, int *conn,
                  float *x_c, float *y_c, float *z_c,
                  int normal_size,
                  int max_no_levels,
                  int min_small_enough,
                  int crit_level,
                  int limit_fX,
                  int limit_fY,
                  int limit_fZ);
    /// Another constructor
    coDoBasisTree(const coObjInfo &info);
    /// Still another constructor
    coDoBasisTree(const coObjInfo &info, const char *label);

    /** Constructor for deserialization
       * @param info object name
       * @param label1 either OCTREE or OCTREP
       * @param label2 another label
       * @param cellListSize number of cellList items
       * @param macroCellListSize number of macroCellList items
       * @param cellBBoxesSize number of cellBBoxes items
       * @param gridBBoxSize number of gridBBox items
       */
    coDoBasisTree(const coObjInfo &info,
                  const char *label1, const char *label2,
                  int cellListSize,
                  int macroCellListSize,
                  int cellBBoxesSize,
                  int gridBBoxSize);

    /// Destructor
    virtual ~coDoBasisTree();

    /**  Visualise fills std::vectors
       *   with the information required to produce a coDoLines
       *   object (line and connectivity list, and coordinate arrays)
       *   with which you may create for instance a coDoLines
       *   object for visualising the octtree
       */
    void Visualise(std::vector<int> &ll, std::vector<int> &cl,
                   std::vector<float> &xi, std::vector<float> &yi, std::vector<float> &zi);
    /** search: Finding the cell (cells) in which a point lies
       * @param point array with the 3 coordinates of the point
       * @return an array is returned whose first element gives the number of candidate cells, the cell labels follow
       */
    const int *search(const float *point) const;
    /// Dump of information of the macrocells and cells defining the octtree
    const int *extended_search(const coVector &point1, const coVector &point2, std::vector<int> &OctreePolygonList) const;
    ///extends normal search function, by searching for all cells between two points
    bool cutLineCuboid(const coVector &point1, const coVector &point2, const float *cuboidBbox, std::vector<coVector> &CutVec) const;
    /// cuts a line with a cuboid and gives back a vector with the coordinates
    bool cutThroughOct(int baum, const float bbox[6], std::vector<int> &ll, std::vector<int> &cl, std::vector<float> &xi, std::vector<float> &yi, std::vector<float> &zi, const coVector &point1, const coVector &point2, std::vector<int> &OctreePolygonList) const;
    /// recursive cut through a octree macro cell
    const int *getBbox(std::vector<float> &boundingBox) const;
    ///get the BBox of a given Octtree
    const int *area_search(std::vector<float> &Bbox, std::vector<int> &GridElemList) const;
    ///search for all cells within a 3D area
    void getMacroCellElements(int baum, float bbox[6], std::vector<int> &ElementList, const float *reference_Bbox) const;
    ///get all Elements which are in a specific macro cell
    bool lineInBbox(const coVector &point1, const coVector &point2, const float bbox[6]) const;
    ///check if line is in a bbox

    friend ostream &operator<<(ostream &outfile, coDoBasisTree &tree);
    /** getChunks: get a list of chunks in macroCellList for
       * a test object that defines: bool operator()(const float[3]);
       */
    class functionObject
    {
    public:
        virtual bool operator()(const float[6]) const = 0;

        virtual ~functionObject() = 0;
    };
    void getChunks(vector<const int *> &chunks, const functionObject *test);

    int getNumCellLists();
    int getNumMacroCellLists();
    int getNumCellBBoxes();
    int getNumGridBBoxes();

    void getAddresses(int **cellList, int **macroCellList, float **cellBBox, float **gridBBox, int **fX, int **fY, int **fZ, int **max_no_levels);

protected:
    int getObjInfo(int, coDoInfo **) const;
    int rebuildFromShm();

    void BBoxForElement(int i);

    // once the tree is made up to some level, we share the cell
    // population and recursively split the cells
    void ShareCellsBetweenLeaves();
    void IniBBox(float *,
                 const int *) const;
    static int IsInMacroCell(const int *okey,
                             const int *macro_keys);
    void fillBBoxSon(float *bbox_son,
                     const float *bbox,
                     int son) const;
    void SplitOctTree(const float *bbox,
                      std::vector<int> &population_,
                      int level,
                      int offset);
    int CellsAreTooBig(const float *bbox,
                       std::vector<int> &population);
    // Recreate Shared Memory objects here
    void RecreateShm(const char *label1, const char *label2);
    // determines how many times we apply SplitMacroCells
    void DivideUpToLevel();
    // used by search...
    const int *lookUp(int position,
                      int *okey,
                      int mask) const;
    // used in constructor...
    void MakeOctTree(int *el, int *conn, float *x_c, float *y_c, float *z_c);
    // position of a tree in the forest
    int Position(int *key) const;

    // Shared memory stuff
    coIntShmArray cellList;
    coIntShmArray macroCellList;
    coFloatShmArray cellBBoxes;
    coFloatShmArray gridBBox;

    coIntShm fXShm;
    coIntShm fYShm;
    coIntShm fZShm;
    coIntShm max_no_levels_Shm;

    //+++++++++++++++++++++++++++++++++++++++++++++++++++++
    // real tree construction
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++
    std::vector<int> macCellList_;
    std::vector<int> cellList_;
    std::vector<int> *populations_;
    std::vector<float> cellBBoxes_;
    // size and pointers to grid lists... do not delete them
    int nelem;
    int nconn;
    int ncoord;
    int *el_;
    int *conn_;
    float *x_c_;
    float *y_c_;
    float *z_c_;
    // used only when making the tree!!!
    float *cell_bbox_;
    mutable float grid_bbox_[6];

    mutable int fX_;
    mutable int fY_;
    mutable int fZ_;

    // use when making the octtree for the first time
    const int normal_size_;
    mutable int max_no_levels_;
    const int min_small_enough_;
    const int crit_level_;
    const int limit_fX_;
    const int limit_fY_;
    const int limit_fZ_;

private:
    void addChunk(vector<const int *> &chunks,
                  int position, const float bbox[6], const functionObject *);
    float cellFactor_;
    int small_enough_;
    void treePrint(ostream &outfile, int level,
                   int *key, int offset);
    void VisualiseOneOctTree(int baum, const float bbox[6], std::vector<int> &ll, std::vector<int> &cl, std::vector<float> &xi, std::vector<float> &yi, std::vector<float> &zi);

    void RecreateShmDL(covise_data_list *);
};
}
#endif
