C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      program main         
      implicit none
      integer isp,lmax,ierr

c     Fuer Propeller
c     parameter(lmax=76000000)

c     Fuer Valid fein
c     parameter(lmax=46000000)

c     parameter(lmax=8300000)
c     parameter(lmax=5000000)

c     256MB
c     parameter(lmax=67108864)

c     300MB
      parameter(lmax=78643200)

      common  isp(lmax)
c     **********************************************************

c     **********************************************************
      CALL MAINN(isp,lmax)
c     **********************************************************


      CALL MPI_FINALIZE(ierr)
      stop
      end

