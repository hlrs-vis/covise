*KEYWORD
*TITLE
LS-DYNA v.960 user input
$
*DATABASE_FORMAT
         0
$
$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$                             SECTION DEFINITIONS                              $
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$
*SECTION_SOLID
         2         1
$
$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$                             MATERIAL DEFINITIONS                             $
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$
*MAT_RIGID
         1 0.780E-02 0.210E+06  0.330000       0.0       0.0       0.0
      1.00      4.00      7.00    

*HOURGLASS
         1         5     0.130         0        50 0.600E-01      0.00      0.00
$
$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$                              PARTS DEFINITIONS                               $
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$
$
*PART
Part          1 for Mat         1 and Elem Type         1
         1         1         1         0         0         0         0
$
*PART
Part          2 for Mat         2 and Elem Type         1
         2         1         2         0         0         0         0
$
*PART
Part          3 for Mat         3 and Elem Type         2
         3         2         3         0         1         0         0
$
$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$                                RIGID BOUNDRIES                               $
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$
*LOAD_RIGID_BODY
         1         3         1     1.000         0         0         0         0
$
$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$                              CONTROL OPTIONS                                 $
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$
*CONTROL_ENERGY
         2         2         2         2
*CONTROL_SHELL
      20.0         1        -1         1         2         2         1
*CONTROL_CONTACT
   0.01000   0.00000         2         1         1         1         1
         0         0        10         0      4.00    
*CONTROL_TIMESTEP
    0.0000    0.5000         0      0.00      0.00    
$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$                                 TIME HISTORY                                 $
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$
$*DATABASE_MATSUM  
$     0.002
*DATABASE_RWFORC
    0.1000
*DATABASE_GLSTAT
    0.1000
*DATABASE_DEFORC
    0.1000
*DATABASE_MATSUM
    0.1000
*DATABASE_NCFORC
    0.1000
*DATABASE_RCFORC
    0.1000
*DATABASE_DEFGEO
    0.1000
*DATABASE_SPCFORC
    0.1000
*DATABASE_SWFORC
    0.1000
*DATABASE_BNDOUT
    0.1000
*DATABASE_RBDOUT
    0.1000
*DATABASE_GCEOUT
    0.1000
*DATABASE_SLEOUT
    0.1000
*DATABASE_JNTFORC
    0.1000
$*********************************
*DATABASE_MOVIE
    0.1000
*DATABASE_EXTENT_MOVIE
         3        30
*DATABASE_ELOUT
    0.1000
*DATABASE_NODOUT
    0.1000
$*DATABASE_MOVIE 
$     0.002
$*DATABASE_EXTENT_MOVIE 
$         3        30
$*DATABASE_ELOUT   
$     0.002
$*DATABASE_NODOUT  
$     0.002
$*********************************
*INCLUDE
history.k
$*********************************
*DATABASE_BINARY_D3PLOT
    0.5000
*DATABASE_BINARY_D3THDT
    0.1000
*DATABASE_BINARY_D3DUMP
     63739
$*DATABASE_BINARY_D3PLOT
$     0.002
$*DATABASE_BINARY_D3THDT
$     0.002
$*DATABASE_BINARY_D3DUMP
$      9626
$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$                              DATABASE OPTIONS                                $
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$
*DATABASE_EXTENT_BINARY
         0         0         5         1         0         0         0         0
         0         0         4         0         0         0
$*********************************
*INTERFACE_SPRINGBACK_DYNA3D
         1
*SET_PART_LIST
         1
         2
$*********************************
*INCLUDE
topology.k
*INCLUDE
boundary.k
*INCLUDE
anpressD.k
*INCLUDE
explOptions.k
$*********************************
*END
