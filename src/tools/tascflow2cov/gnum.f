      subroutine gnum
     +	( grdname, grdnum, IERR )
c these routine find the index of a subgrid block given its name

      INTEGER IERR
      CHARACTER  *(*) grdname


C     Get dimensions of possible subgrid
      CALL TRGNUM (grdname, grdnum, IERR)
      IF (IERR .GT. 0) GOTO 999

999   CONTINUE

      END

