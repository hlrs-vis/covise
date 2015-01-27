#include "cfx5ext.h"
dllexport(sinangular)
      SUBROUTINE SINANGULAR (
     &  NLOC, NRET, NARG, RET, ARGS, CRESLT, CZ,DZ,IZ,LZ,RZ )
CC
CD User routine: defines an inlet velocity profile using the 1/7 power law
CC
CC --------------------
CC        Input
CC --------------------
CC
CC  NLOC   - size of current locale
CC  NRET   - number of components in result
CC  NARG   - number of arguments in call
CC  ARGS() - (NLOC,NARG) argument values
CC
CC --------------------
CC      Modified
CC --------------------
CC
CC  Stacks possibly.
CC
CC --------------------
CC        Output
CC --------------------
CC
CC  RET()  - (NLOC,NRET) return values
CC  CRESLT - 'GOOD' for success
CC
CC --------------------
CC       Details
CC --------------------
CC
CC  Sets the velocity profile at an inlet.
CC
CC  This routine evaluates the CEL function:-
CC
CC     <Function Name>(Wmax,r,Rmax)
CC
CC  The profile is: velocity = Wmax*(1-(r/Rmax))**(1/7)
CC
CC  The Solver will fail if (1-(r-/Rmax)) is negative.
CC  Slightly negative values can be obtained at the walls due to geometry
CC  tolerance errors (i.e. r = 0.02000001 when Rmax = 0.02).
CC  To prevent this, negative values of (1-(r-/Rmax)) are set equal to 0.0.
CC
CC======================================================================
C
C ------------------------------
C     Preprocessor includes
C ------------------------------
C
#include "MMS.h"
C ------------------------------
C        Global Parameters
C ------------------------------
C
C
C ------------------------------
C        Argument list
C ------------------------------
C
      INTEGER NLOC,NARG,NRET
C
      CHARACTER CRESLT*(*)
C
      REAL ARGS(NLOC,NARG), RET(NLOC,NRET)
C
      INTEGER IZ(*)
      CHARACTER CZ(*)*(1)
      DOUBLE PRECISION DZ(*)
      LOGICAL LZ(*)
      REAL RZ(*)
C
C ------------------------------
C        External routines
C ------------------------------
C
C
C ------------------------------
C        Local Parameters
C ------------------------------
C
C
C ------------------------------
C        Local Variables
C ------------------------------
C
      INTEGER ILOC, ISIZE, JADRES
      
      CHARACTER*4 CDTYPE
      REAL angular, velfactor, sinang
      REAL pi
	  CHARACTER* 200  CURDIR
C ------------------------------
C        Stack pointers
C ------------------------------
C
C=======================================================================
C
C ---------------------------
C    Executable Statements
C ---------------------------
C
C
C Initialise RET() to 1.
C
      CALL SET_A_R( RET,1.0,NLOC*NRET )
C
C---- Get Value from MMS, written in a Junction Box
C     NLOC..number of inletnodes
c	  
c      CALL LOCDAT(factor,CDTYPE,'STOP',ISIZE,JADRES,CRESLT)
c      write(*,*)'+++++++++ USER FUNCTION sinangular+++++++++'  
      CALL PUTDIR(CURDIR,LENCD)
c      write(*,*)'CURDIR: ',CURDIR
      CALL PEEKR('angular',1,angular,'SKIP',CRESLT,RZ)  
      IF(CRESLT .EQ. 'NONE') THEN
	  angular = 0
c	  write (*,*)'Error PEEKR, could not get angular.'
      ENDIF		  
      pi = 3.141592654
c	  write(*,*)'pi: ',pi	    
      sinang=sin(pi*angular/180)	  	  	  
c      write(*,*)'angular:    ',angular
c      write(*,*)'sin:        ',sinang	    
c      write(*,*)'NLOC: ',NLOC
c      write(*,*)'NRET: ',NRET	  
      DO ILOC = 1,NLOC
c     RET()  - (NLOC,NRET) return values        
	RET(ILOC,1) = sinang
      END DO
      
C
C Set success flag.
      CRESLT = 'GOOD'
C
C=======================================================================
      END



