*KEYWORD
*TITLE
LS-DYNA v.970 user input (Implicit)
$
*DATABASE_FORMAT
         0
$
$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$                             CONTROL DEFINITIONS                              $
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$
*CONTROL_IMPLICIT_GENERAL
$   IMFLAG       DT0    IMFORM      NSBS       IGS
         1     0.001         0         0         0
*CONTROL_IMPLICIT_SOLUTION
$   NSOLVR    ILIMIT    MAXREF     DCTOL     ECTOL     RCTOL     LSTOL
         2         0        50         0         0
$    DNORM    DIVERG     ISTIF   nlprint
         0         2         0         1
*CONTROL_IMPLICIT_SOLVER
$   LSOLVR    LPRINT     NEGEV
         5         0         1
*CONTROL_IMPLICIT_STABILIZATION
$      IAS     SCALE    TSTART      TEND
         1     0.001         0         0
*CONTROL_TERMINATION
$   ENDTIM
      0.01
$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$                              PARTS DEFINITIONS                               $
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$
*PART
Part          2 for Mat         2 and Elem Type         1
         2         1         2         0         0         0         0
$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$                              CONTROL OPTIONS                                 $
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$
*CONTROL_SHELL
      20.0         1        -1         1         2         2         1
$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$                                 TIME HISTORY                                 $
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$
*DATABASE_DEFGEO  
      0.01    
$*********************************
*DATABASE_MOVIE 
      0.01    
*DATABASE_EXTENT_MOVIE 
         3        30
$*DATABASE_ELOUT   
$      0.01    
$*DATABASE_NODOUT  
$      0.01    
$*********************************
*INCLUDE
history.k
$*********************************
*DATABASE_BINARY_D3PLOT
      0.01
$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$                              DATABASE OPTIONS                                $
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$
*DATABASE_EXTENT_BINARY
         0         0         5         1         0         0         0         0
         0         0         4         0         0         0
$*********************************
*INCLUDE
dynain
*INCLUDE
implOptions.k
*INCLUDE
bPaper.k
$*********************************
*END
