      subroutine subreg
     +	( regia, nsreg, IERR )
c these routine reads the (sub)region number

      INTEGER IERR
      INTEGER nsreg, I
      CHARACTER *(*) regia

      CALL TRREGN ( regia, nsreg, IERR)
      IF (IERR .GT. 0) GOTO 999
      

999   CONTINUE

      END

