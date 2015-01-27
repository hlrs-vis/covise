      subroutine region
     +	( regnam, i, j, k, u, v, w, ng, isreg, IERR )
c these routine reads the (sub)grid coordinates

      INTEGER IERR
      INTEGER i, j, k, u, v, w
      INTEGER ng
      CHARACTER *(*) regnam
c      CHARACTER buffer*20

c      buffer = regnam(:length)

C     Get coordinates of the given region
      CALL TRREGS (regnam, i, u, j, v, k, w, ng, isreg, IERR)
      IF (IERR .GT. 0) GOTO 999

999   CONTINUE

      END

