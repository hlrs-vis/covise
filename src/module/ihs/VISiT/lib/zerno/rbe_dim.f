C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE RBE_DIM(rbe_name,coord_zeig,
     *                   ndisp_zeil,ndisp_spalt,
     *                   covise_run,
     *                   numwall, numbal,
     *                   numdir, coldir,
     *                   dirindex, dirval,
     *                   numpress)
          
      implicit none
 
      include 'mpif.h'

      include 'common.zer'

      integer i,lu,npoper,luerr,iread,ityp,kn,
     *        coord_zeig,ndisp_zeil,ndisp_spalt,nnn

      real    wert

      logical     format_read

      dimension coord_zeig(knmax_num)

      character*80  rbe_name,reihe,comment
     
      parameter (lu=50)


      integer   covise_run
      integer   numwall, numbal, numdir, coldir, dirindex, numpress
      real      dirval
      dimension dirindex(coldir*numdir)
      dimension dirval(numdir)
      
c     *****************************************************************


c     *****************************************************************
c     DIMENSIONEN DER RANDBEDINGUNGEN:

      if (covise_run.eq.0) then
          open(lu,file=rbe_name,status='old',err=777)
          format_read=.true.
          CALL HEAD_READ(lu,rbe_name,format_read,reihe)

          nrbpoi=iread(reihe)
          nwand=iread(reihe)
          npres=iread(reihe)
          nsyme=iread(reihe)
          nzykl=iread(reihe)
          npoper=iread(reihe)
          nconv=iread(reihe)
          ntemp=iread(reihe)
      else
          nrbpoi=numdir
          nwand=numwall
          npres=numpress
          nsyme=0
          nzykl=0
          npoper=0
          nconv=numbal
          ntemp=0
      endif

      if (nzykl.ne.0.or.npoper.ne.0) then
         call erro_init(myid,parallel,luerr)
         write(luerr,*)'Fehler in Routine RBE_DIM'
         write(luerr,*)'Periodische FENFLOSS-Randbedingungen  '
         write(luerr,*)'koennen in FLOW nicht verarbeitet werden.'
         write(luerr,*)'Periodische Randbedingungen werden in FLOW '
         write(luerr,*)'ueber Bilanzflaechen und entsprechende     '
         write(luerr,*)'Angaben im Steuerfile spezifiziert.        '
         call erro_ende(myid,parallel,luerr)
      endif


c     Bestimmung der Anzahl Knoten mit Displ-Randbedingungen:
      do 101 i=1,knmax_num
        coord_zeig(i)=0
 101  continue

      ndisp_zeil=0
      ndisp_spalt=0
      do 100 i=1,nrbpoi
          if (covise_run.eq.0) then
              read(lu,*) kn,ityp,wert
          else
             kn   = dirindex(2*i-1)
             ityp = dirindex(2*i)
             wert = dirval(i)
          endif

          ndisp_spalt=MAX(ndisp_spalt,ityp)
          if (kn.gt.knmax_num) then
             call erro_init(myid,parallel,luerr)
             write(luerr,*)'Fehler in Routine RBE_DIM !     '
             write(luerr,*)'Im Geometrie-File existieren nur '
             write(luerr,*)'Knoten-Nummern bis ',knmax_num   
             write(luerr,*)'In den Displ-Randbedingungen gibt es aber '
             write(luerr,*)'groessere Knotennummern !! '
             write(luerr,*)'Knotennummer im Displ-Rb.:',kn  
             write(luerr,*)'Adresse                  :',i
             call erro_ende(myid,parallel,luerr)
          endif
          if (coord_zeig(kn).eq.0) then
             ndisp_zeil=ndisp_zeil+1
             coord_zeig(kn)=coord_zeig(kn)+1
          endif
 100  continue

      if (parallel) then
         nnn=ndisp_spalt
         CALL MPI_ALLREDUCE(nnn,ndisp_spalt,1,MPI_INTEGER,MPI_MAX,
     *                      MPI_COMM_WORLD,ierr)
      endif

      if (covise_run.eq.0) then
          close(lu)
      endif
      goto 888
c     *****************************************************************

 777  continue      
      comment='Fehler beim Oeffnen von File (rbe_dim):'
      call erro_init(myid,parallel,luerr)
      call char_druck(comment,rbe_name,luerr)
      call erro_ende(myid,parallel,luerr)
 888  continue      

      return
      end

