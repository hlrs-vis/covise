      subroutine file
     +	( ftype, fname, fmt, IERR)
c these routine reads the (sub)grid coordinates

      INTEGER IERR
      CHARACTER*4 ftype
      CHARACTER *(*) fname
      CHARACTER*2 fmt
      


C     Override the Default Filename and Format
      CALL TGFILE (ftype, fname, fmt, IERR)
      IF (IERR .GT. 0) GOTO 999

999   CONTINUE

      END

