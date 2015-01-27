      subroutine ngbl
     +	( ngrids, IERR )

      INTEGER  ngrids
      INTEGER IERR

C  look how many blocks are present
      CALL TRNGBL(ngrids, IERR)
      IF (IERR .GT. 0) GOTO 999
      

999   CONTINUE


      END

