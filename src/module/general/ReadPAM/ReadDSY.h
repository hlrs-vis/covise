/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_DSY_H_
#define _READ_DSY_H_

#include <string>
#include <api/coModule.h>
using namespace covise;
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoSet.h>
#include <do/coDoData.h>
#include "auxiliary.h"

#define TITLE_MAT 48

#ifdef CO_hp1020
#define dsyclo_ dsyclo
#define dsyvar_ dsyvar
#define dsysta_ dsysta
#define dsybri_ dsybri
#define dsytim_ dsytim
#define dsybea_ dsybea
#define dsylbr_ dsylbr
#define dsylbe_ dsylbe
#define dsyshe_ dsyshe
#define dsytoo_ dsytoo
#define dsylsh_ dsylsh
#define dsylto_ dsylto
#define dsylno_ dsylno
#define dsynod_ dsynod
#define dsywno_ dsywno
#define dsyopn_ dsyopn
#define dsytit_ dsytit
#define dsyhal_ dsyhal
#define dsyhva_ dsyhva
#define dsywbr_ dsywbr
#define dsywsh_ dsywsh
#define dsywto_ dsywto
#define dsywbe_ dsywbe
#define dsyhgl_ dsyhgl
#define dsynam_ dsynam
#define dsyhma_ dsyhma
#define dsyhse_ dsyhse
#define dsyhct_ dsyhct
#define dsyhrw_ dsyhrw
#define dsyhba_ dsyhba
#define dsyhch_ dsyhch
#define dsyhwa_ dsyhwa
#define dsyair_ dsyair
#define dsyths_ dsyths
#define dsyalt_ dsyalt
#define dsysph_ dsysph
#define dsywsp_ dsywsp
#endif

extern "C" {
int dsyclo_();
int dsyvar_(int *, int *, int *, int *, int *, int *);
int dsysta_(int *);
int dsybri_(int *);
int dsytim_(int *, float *);
int dsybea_(int *);
int dsylbr_(int *);
int dsylbe_(int *);
int dsyshe_(int *);
int dsytoo_(int *);
int dsylsh_(int *);
int dsylto_(int *);
int dsylno_(int *);
int dsynod_(int *, int *, int *, int *, int *);
int dsywno_(int *, int *, int *, int *, int *, float *, float *, int *);
int dsyopn_(const char *, int *, int);
int dsytit_(int *, int *, char *, int);
int dsyhal_(int *, int *, int *, int *);
int dsyhva_(int *, int *, int *, int *, int *, int *, int *, int *, int *, int *, int *);
int dsywbr_(int *, int *, int *, int *, int *, float *, float *, int *);
int dsywsh_(int *, int *, int *, int *, int *, float *, float *, int *);
int dsywto_(int *, int *, int *, int *, int *, float *, float *, int *);
int dsywbe_(int *, int *, int *, int *, int *, float *, float *, int *);
int dsyhgl_(int *, int *, int *, int *, float *, int *);
int dsynam_(int *, int *, char *, int *, int *);
int dsyhma_(int *, int *, int *, int *, int *, int *, int *, float *, int *);
int dsyhse_(int *, int *, int *, int *, int *, int *, int *, float *, int *);
int dsyhct_(int *, int *, int *, int *, int *, int *, int *, float *, int *);
int dsyhrw_(int *, int *, int *, int *, int *, int *, int *, float *, int *);
int dsyhba_(int *, int *, int *, int *, int *, int *, int *, float *, int *);
int dsyhch_(int *, int *, int *, int *, int *, int *, int *, float *, int *);
int dsyhwa_(int *, int *, int *, int *, int *, int *, int *, float *, int *);
int dsyair_(int *, int *, int *);
#ifdef _INCLUDE_SPH_
int dsyths_(int *, int *);
int dsyalt_(int *, int *, int *);
int dsysph_(int *);
int dsywsp_(int *, int *, int *, int *, int *, float *, float *, int *);
#endif
}

class ReadDSY
{
    // limit number of variables read for a given
    // cell type or for a given global variable type
    int CELL_REDUCE;
    int GLOBAL_REDUCE;
    // For special globals associated to materials,
    // transmission sections, constact interfaces,
    // rigis walls, airbags, airbag chambers and airbag walls,
    // you may set limits with the following constants.
    // They limit the number of entities for which variables
    // are created.
    int MAT_GLOBAL_REDUCE;
    int TS_GLOBAL_REDUCE;
    int CI_GLOBAL_REDUCE;
    int RW_GLOBAL_REDUCE;
    int AB_GLOBAL_REDUCE;
    int ABCH_GLOBAL_REDUCE;
    int ABW_GLOBAL_REDUCE;

    std::string path_;
    std::string path_glo_;
    whichContents contents_;
    whichContents cell_contents_;
    whichContents global_contents_;
    coDoSet *theMaterials_;
    coDoSet *theELabels_;
    coDoSet *theReferences_;
    int ndim_;
    int numnp_;
    int no_entities_[noTypes];

    int no_1D_entities_;
    int *oneD_conn_;
    int *oneD_label_;

    // Globals and related "strange" things, see dsyhva_
    int iglob_;
    int nmat_;
    int imat_;
    int nsect_;
    int isect_;
    int nctct_;
    int ictct_;
    int nrgdw_;
    int irgdw_;
    int nrbg_;
    int irbg_;

    int rtn_; // error for many library calls

    int nstate_;
    int min_;
    int max_;
    int jump_;
    int noTimeReq_;
    float *zeit_;

    int fromIntToDSY(int internal)
    {
        return (1 + min_ + internal * jump_);
    }

    int isdisp_;
    int isvel_;
    int isacc_;
    int isadd_;

    int *entity_conn_[noTypes];
    int entity_nodes_[noTypes];
    // create in grid, reuse for data ..............
    int *entity_label[noTypes];
    int *node_entity_label_[noTypes];
    Map1D node_map_;

    Map1D node_entity_map_[noTypes];

    int lrec;
    int unit;
    int shells4;
    int shells3;
    int tools4;
    int tools3;

    void count_shells(const int *, int *shells4, int *shells3, coStringObj::ElemType type);
    int bookkeeping(int); // work out number of entities

    int theSame_; // flag set by scanContents if path is the same as path_
    // this flag is read by scanCellContents in order to
    // decide whether the old cell_contents_ is returned

    coDoSet *gridAtTime(std::string name, std::string mat_name, std::string ela_name,
#ifdef _LOCAL_REFERENCES_
                        std::string ref_name,
#endif
                        int time, coDistributedObject **, coDistributedObject **
#ifdef _LOCAL_REFERENCES_
                        ,
                        coDistributedObject **
#endif
                        );
    coDoSet *cellObjAtTime(std::string name, int time, int req_ind);
    coDoSet *tensorObjAtTime(const TensDescriptions &tdesc, int time, int req_ind);
    coDoSet *scalarNodal(const char *name, const char *species,
                         int this_state, float *node_vars,
                         int offset, int num_vars);
    coDoSet *vectorNodal(const char *name, const char *species, int this_state,
                         float *node_vars, int offset1, int offset2,
                         int offset3, int num_vars);

    int cellVarNotInFile(coStringObj::ElemType cell_type, INDEX ind);
    int globalVarNotInFile(INDEX ind);
    void addSpecialGlobals(coStringObj::ElemType typa);

    coDistributedObject *realObj(std::string &name, int this_state,
                                 coStringObj::ElemType e_type, int first, int last,
                                 int type, INDEX ind);
    coDistributedObject *realTensorObj(std::string &name, int this_state,
                                       coStringObj::ElemType e_type, int first, int last,
                                       coDoTensor::TensorType ttype, int *req_label_ind);

    void mydsyhal(int *type, int *nall, int *iall, int *rtn);

    void initEntities(void);
    int Translations(coStringObj::ElemType, char *);
    void FindVectors(coStringObj::ElemType, int, char *);
    void Lump(const char *, const char *, const char *, const char *, int, char *);
    int FindTitle(const char *tit, int no_titles, const char *titles);
    void stripSpecies(char *species, std::string &title);

    void fillReferences(int cell_type, float *ref_addr, coDistributedObject *grid,
                        const float *node_coor);

    float scale_;
    int useDisplacements_;
    void setScale(float);
    void Connectivity(int);
    void Labels(int, int *);
    void ReadCellVar(int cell_type, int *nstate, int *numvar,
                     int *indice, int *ifirst, int *ilast, float *zeit, float *varval);

public:
    int getNoStates()
    {
        int ret;
        dsyopn_(path_.c_str(), &rtn_, path_.length());
        dsysta_(&ret);
        dsyclo_();
        return ret;
    }
    void setTimeRequest(int min, int max, int jump)
    {
        min_ = min - 1;
        max_ = max - 1;
        jump_ = jump;
        noTimeReq_ = 1 + (max_ - min_) / jump_;
    }
    whichContents getCellContents()
    {
        return cell_contents_;
    }
    whichContents getGlobalContents()
    {
        return global_contents_;
    }
    const char *getTitle(int i, int typ, char ret[26])
    {
        int c;
        ret[24] = ret[25] = '\0';
        if (typ == 0)
            strncpy(ret, contents_[i].ObjName_.c_str(), 24);
        else if (typ == 1)
            strncpy(ret, cell_contents_[i].ObjName_.c_str(), 24);
        else if (typ == 2)
            return global_contents_[i].ObjName_.c_str();
        //         strncpy(ret, global_contents_[i].ObjName_,24);
        for (c = 24; c > 0; --c)
        {
            if (ret[c - 1] != '\0' && ret[c - 1] != ' ')
            {
                ret[c] = '\n';
                ret[c + 1] = '\0';
                return ret;
            }
        }
        ret[0] = '\n';
        return ret;
    }
    coDoSet *materials()
    {
        return theMaterials_;
    }
    coDoSet *eleLabels()
    {
        return theELabels_;
    }
    coDoSet *references()
    {
        return theReferences_;
    }
    float getScale()
    {
        return scale_;
    }

    ReadDSY(); //:unit(1),theSame_(0) {
    /*
      #if defined(__sgi)
           lrec=1024;
      #endif
      #if defined(__hpux) || defined(_AIX)
           lrec=4096;
      #endif
            isvel_= isacc_ = isadd_ = isdisp_ = 0;
            int i;
            for(i=0;i<noTypes;++i) node_entity_label_[i] = 0;
            oneD_conn_ = oneD_label_ = 0;
      }
      */
    whichContents scanContents(const char *path);
    whichContents scanCellContents(void);
    whichContents scanGlobalContents(const char *);
    coDoSet *grid(const char *objName, const char *matName, const char *elaName,
#ifdef _LOCAL_REFERENCES_
                  const char *refName,
#endif
                  float scale = 1.0);
    coDoSet **nodalObj(int no_objects, std::string *objName, int *req_label);
    coDoSet **cellObj(int no_objects, std::string *objName, int *req_label);
    coDoSet **tensorObj(const TensDescriptions &);
    coDistributedObject **globalObj(int no_objects, std::string *objName, int *req_label);
    ~ReadDSY()
    {
        delete[] oneD_conn_;
        delete[] oneD_label_;
    }
    void clean();
};
#endif
