MODULE DTF_var
  IMPLICIT NONE
  INTEGER, PARAMETER :: string_length=80, units_length=32

!!$  These parameters correspond to the DATA TYPE enumeration
!!$  in DTF.
  
  INTEGER :: DTF_INT_DATA= 1, DTF_DOUBLE_DATA=2,&
       & DTF_SINGLE_DATA=3, DTF_STRING_DATA=4

!!$  Same for topotypes
  
  INTEGER :: DTF_GNRL_TOPO=0, DTF_NODE_TOPO=1, DTF_EDGE_TOPO=2,&
       & DTF_FACE_TOPO=3, DTF_CELL_TOPO=4

  INTEGER :: DTF_ERROR=(-1),DTF_OK=(0)
  INTEGER :: DTF_READ_ALL_ELEMENTS=(-1)
  INTEGER :: DTF_TRUE=(1),DTF_FALSE=(0)
  INTEGER :: DTF_INVALID=(-1)

!!$  Information that should be included from dtf.h, but
!!$  is not.
  
  INTEGER, PARAMETER :: DTF_NCELLTYPES=11
  INTEGER, DIMENSION(DTF_NCELLTYPES) :: DTF_CELLTYPE_TO_NNODES= &
                                      &   (/3, 4, 4, 5, 6, 8, 6, 8,&
                                      & 10, 20, -1 /)
  
  INTEGER, PARAMETER :: DTF_NFACETYPES=7
  INTEGER, PARAMETER :: DTF_NFACEKINDS=DTF_NFACETYPES*3
  INTEGER, DIMENSION(DTF_NFACETYPES) :: DTF_FACETYPE_TO_NNODES= &
                         & (/ 2, 3, 4, 3, 6, 8, -1 /)
  
END MODULE DTF_var
