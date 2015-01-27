      subroutine rclose
     +	(IERR)

      INTEGER IERR

c     close the database

      CALL TGCLOS(IERR)
      IF (IERR .GT. 0) GOTO 999
      

999   CONTINUE


      END
