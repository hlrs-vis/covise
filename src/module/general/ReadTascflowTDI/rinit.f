      subroutine rinit
     +	( nr, ni, nc, IERR )
c these routine reads the (sub)grid coordinates

      INTEGER nr,ni,nc
      INTEGER IERR



C     Initialize the Database
      CALL TGINIT (nr, ni, nc, IERR)
      IF (IERR .GT. 0) GOTO 999

999   CONTINUE

      END

