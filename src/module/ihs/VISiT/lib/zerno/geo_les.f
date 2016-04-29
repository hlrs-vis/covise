C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE GEO_LES(coord,coord_num,coord_mod,
     *                   lnods,lnods_num,lnods_mod,
     *                   coord_zeig,lnods_zeig,
     *                   geo_name)

      implicit none

      include 'mpif.h'      
      include 'common.zer'      

      integer lnods,lnods_num,coord_num,coord_mod,lnods_mod,
     *        coord_zeig,lnods_zeig

      integer i,j,k,ielem,
     *        lu,kn_num,mod_num,
     *        knoten(8),nnn,num_max,
     *        el_num,luerr

      real    coord,vek(3),ddd

      logical  format_read

      character*80 geo_name,reihe,file_name,comment

      parameter (lu=50)

      dimension coord_num(npoin_max),coord(npoin_max,ncd),
     *          coord_mod(npoin_max),
     *          lnods_num(nelem_max),lnods(nelem_max,nkd),
     *          lnods_mod(nelem_max)

      dimension coord_zeig(knmax_num),lnods_zeig(elmax_num)
c     ****************************************************************


      file_name=geo_name
      open(lu,file=file_name,status='old',err=777)
      format_read=.true.
      CALL HEAD_READ(lu,file_name,format_read,reihe)



c     ****************************************************************
c     EINLESEN DER KNOTEN:

      do 5 i=1,knmax_num
        coord_zeig(i)=0
 5    continue
      
      num_max=0
      do 10 i=1,npoin
        read(lu,*) kn_num,(vek(j),j=1,3),mod_num

        CALL KN_CHECK(kn_num,i,coord_num,coord_zeig,coord,vek)

        coord_zeig(kn_num)=i
        coord_num(i)=kn_num
        coord_mod(i)=mod_num
        if (ncd.eq.2) then
           coord(i,1)=vek(1)
           coord(i,2)=vek(2)
        else if (ncd.eq.3) then
           coord(i,1)=vek(1)
           coord(i,2)=vek(2)
           coord(i,3)=vek(3)
        endif
        num_max=MAX(num_max,kn_num)

 10   continue

c     Maximale Knotennummer:
      if (parallel) then
        kn_num=num_max
        CALL MPI_ALLREDUCE(kn_num,num_max,1,MPI_INTEGER,
     *                     MPI_MAX,MPI_COMM_WORLD,ierr)  
      endif

      if (num_max.ne.knmax_num) then
        call erro_init(myid,parallel,luerr)
        write(luerr,*)'Fehler in Routine GEO_LES'
        write(luerr,*)'Die tatsaechliche maximale Knotennummer stimmt'
        write(luerr,*)'mit der im Geometrie-File angegebenen     '
        write(luerr,*)'maximalen Knotennummer nicht ueberein.        '
        write(luerr,*)'Tatsaechliche max. Knotennummer:',num_max     
        write(luerr,*)'Angegebene    max. Knotennummer:',knmax_num
        call erro_ende(myid,parallel,luerr)
      endif
c     ****************************************************************


c     ****************************************************************
c     EINLESEN DER ELEMENTE:
      do 6 i=1,elmax_num
        lnods_zeig(i)=0
 6    continue

      num_max=0
      do 20 i=1,nelem
             read(lu,*) el_num,(knoten(j),j=1,nkd),mod_num

          CALL EL_CHECK(el_num,knoten,coord_zeig,lnods_zeig,
     *                  coord_num,lnods)

          lnods_zeig(el_num)=i
          lnods_num(i)=el_num
          lnods_mod(i)=mod_num
          num_max=MAX(el_num,num_max)
          do 21 k=1,nkd
            lnods(i,k)=coord_zeig(knoten(k))
 21       continue
 20   continue


c     Bestimmen der maximalen Element-Nummer elmax_num:
      if (parallel) then
        el_num=num_max
        CALL MPI_ALLREDUCE(el_num,num_max,1,MPI_INTEGER,
     *                     MPI_MAX,MPI_COMM_WORLD,ierr)  
      endif                                    

      if (num_max.ne.elmax_num) then
        call erro_init(myid,parallel,luerr)
        write(luerr,*)'Fehler in Routine GEO_LES'
        write(luerr,*)'Die tatsaechliche maximale Elementnummer stimmt'
        write(luerr,*)'mit der im Geometrie-File angegebenen     '
        write(luerr,*)'maximalen Elementnummer nicht ueberein.        '
        write(luerr,*)'Tatsaechliche max. Elementnummer:',num_max     
        write(luerr,*)'Angegebene    max. Elementnummer:',elmax_num
        call erro_ende(myid,parallel,luerr)
      endif

      if (parallel) then
        if (elmax_num.ne.nelem_ges) then
          call erro_init(myid,parallel,luerr)
          write(luerr,*)'Fehler in Routine GEO_LES'
          write(luerr,*)'Die Element sind global gesehen '
          write(luerr,*)'nicht fortlaufend durch numeriert.'
          write(luerr,*)'Anzahl Elemente Orginal-Geometrie:',nelem_ges
          write(luerr,*)'Maximale Element-Nummer          :',elmax_num
          call erro_ende(myid,parallel,luerr)
        endif
      endif                                    
c     ****************************************************************


      close(lu)


c     ****************************************************************
c     BESTIMMUNG DER EXTREMWERTE DER KNOTENABSTAENDE FUER TOLERANZEN:
c
      gitter_mit=0.0
      gitter_min=+1.e+12   
      gitter_max=-1.e+12
c     nnn=0
c     do 500 ielem=1,nelem
c        do 510 i=1,nkd   
c           do 520 j=i+1,nkd   
c              if (ncd.eq.2) then
c                vek(1)=coord(lnods(ielem,i),1)-coord(lnods(ielem,j),1)
c                vek(2)=coord(lnods(ielem,i),2)-coord(lnods(ielem,j),2)
c                vek(3)=0.0
c              else if (ncd.eq.3) then
c                vek(1)=coord(lnods(ielem,i),1)-coord(lnods(ielem,j),1)
c                vek(2)=coord(lnods(ielem,i),2)-coord(lnods(ielem,j),2)
c                vek(3)=coord(lnods(ielem,i),3)-coord(lnods(ielem,j),3)
c              endif
c              ddd=SQRT(vek(1)**2+vek(2)**2+vek(3)**2)
c              gitter_min=MIN(gitter_min,ddd)
c              gitter_max=MAX(gitter_max,ddd)
c              gitter_mit=gitter_mit+ddd
c              nnn=nnn+1
c520        continue
c510     continue
c500  continue 
c
c     gitter_mit=gitter_mit/REAL(nnn)
c
c
c     if (parallel) then
c
c         ddd=gitter_min
c         CALL MPI_ALLREDUCE(ddd,gitter_min,1,MPI_REAL,
c    *                       MPI_MIN,MPI_COMM_WORLD,ierr)
c         ddd=gitter_max
c         CALL MPI_ALLREDUCE(ddd,gitter_max,1,MPI_REAL,
c    *                       MPI_MAX,MPI_COMM_WORLD,ierr)
c         ddd=gitter_mit
c         CALL MPI_ALLREDUCE(ddd,gitter_mit,1,MPI_REAL,
c    *                       MPI_SUM,MPI_COMM_WORLD,ierr)
c
c         gitter_mit=gitter_mit/REAL(numprocs)
c
c     endif
c     *****************************************************************


c     ****************************************************************
c     FEHLERMELDUNG FUER FALSCH GEOEFFNETE FILES:
      goto 888
 777  continue      
      comment='Fehler beim Oeffnen von File (geo_les):'
      call erro_init(myid,parallel,luerr)
      write(luerr,*)'Fehler in Routine GEO_LES'
      call char_druck(comment,file_name,luerr)
      call erro_ende(myid,parallel,luerr)
 888  continue      
c     ****************************************************************

      return
      end
