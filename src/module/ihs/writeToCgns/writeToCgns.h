#ifndef _writeToCgns_H
#define _writeToCgns_H

#include <string.h>
#include <api/coModule.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoIntArr.h>
#include <do/coDoPolygons.h>
#include <do/coDoSet.h>
#include <cgnslib.h>

using namespace covise;
/*!
 * \brief converts computational grid and boundary conditions of AxialRunner tool to cgns format.
 * \author A. Tismer
 * \date 14.08.2012
 * \version 1.0
 */
class writeToCgns : public coModule
{

   private:
      virtual int compute(const char *port);
      ///unstructured grid in covise format
      coInputPort *p_inputPort_grid;
      /*! \brief coDoSet containing coDoIntArrays with boundary element faces. 
       *  \details Each coDoIntArray is ordered as:
       * \f[
       * coDoIntArray[0] = \left[
       *                     \begin{matrix}
       *                       n_{NF} \\
       *                     \end{matrix}
       *                   \right]
       * \f]
       * \f[
       * coDoIntArray[1] = \left[
       *                     \begin{matrix}
       *                       element_1Node_1 \\
       *                       element_1Node_2 \\
       *                       element_1Node_3 \\
       *                       element_1Node_4 \\
       *                       \vdots \\
       *                       element_mnode_n
       *                     \end{matrix}  
       *                   \right]
       * .
       * \f]
       * In the first array \f$n_{NF}\f$ is the number of nodes per element 
       * face and the second array contains the nodes of each face. Formula
       * shows an example of a 4-node face of a 8-node element.
       * \remark covise places the two arraies successively in memory. 
      */
      coInputPort *p_inputPort_boundaryElementFaces;
      ///where to store the generated cgns file
      coFileBrowserParam *cgns_filebrowser;
   public:
      ///constructor
      writeToCgns(int argc, char *argv[]);
};
#endif
