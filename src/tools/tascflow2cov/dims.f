      subroutine dims
     +	( i, j, k, grdnum, grdname, IERR )
c these routine reads the (sub)grid coordinates

      INTEGER i,j,k
      INTEGER IERR
      CHARACTER*20 grdname(20)


C     Get dimensions of possible subgrid
      CALL TRGDIM (i, j, k, grdname, grdnum, IERR)
      IF (IERR .GT. 0) GOTO 999

999   CONTINUE

      END

