C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE PRINT_DIM(comment,nknra_adr,nelra_adr)

      implicit none

      include 'mpif.h'
      include 'common.zer'

      integer   nknra_adr,nelra_adr
      integer   dim_geb_geo(5),dim_geo(5,512),
     *          dim_geb_rbe(7),dim_rbe(7,512)

      integer   icom,j,i,ndisp,luerr,lentb

      character*80 comment
      character*42 zeil_geo(12),zeil_rbe(10)
      character*120 text_1,text_2

      dimension nknra_adr(ntyp_knra+1),nelra_adr(ntyp_elra+1)
c     ****************************************************************
      
      if (myid.eq.0) then
         if (numprocs.gt.512) then
            call erro_init(myid,parallel,luerr)
            write(luerr,*)'Dimension der Hilfsfelder in DIM_PRINT '
            write(luerr,*)'ist zu klein !                         '
            call erro_ende(myid,parallel,luerr)
         endif
      endif


c     ****************************************************************
      if (parti_rbe) then
         ndisp=nknra_adr(2)-nknra_adr(1)
         ntemp=nknra_adr(3)-nknra_adr(2)
         nwand=nelra_adr(2)-nelra_adr(1)
         npres=nelra_adr(3)-nelra_adr(2)
         nsyme=nelra_adr(4)-nelra_adr(3)
         nzykl=nelra_adr(5)-nelra_adr(4)
         nconv=nelra_adr(6)-nelra_adr(5)
      else 
         ndisp=0                          
         ntemp=0                          
         nwand=0                          
         npres=0                          
         nsyme=0                          
         nzykl=0                          
         nconv=0                          
      endif

      dim_geb_geo(1)=npoin
      dim_geb_geo(2)=nelem

      dim_geb_rbe(1)=ndisp                        
      dim_geb_rbe(2)=nwand                         
      dim_geb_rbe(3)=npres
      dim_geb_rbe(4)=nsyme 
      dim_geb_rbe(5)=nzykl
      dim_geb_rbe(6)=nconv
      dim_geb_rbe(7)=ntemp

      if (parallel) then
         CALL MPI_GATHER(dim_geb_geo,5,MPI_INTEGER,
     *                 dim_geo,5,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
         CALL MPI_GATHER(dim_geb_rbe,7,MPI_INTEGER,
     *                 dim_rbe,7,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
      else
         do 101 j=1,5      
           dim_geo(j,1)=dim_geb_geo(j)
 101     continue

         do 102 j=1,7      
           dim_rbe(j,1)=dim_geb_rbe(j)
 102     continue

      endif
c     ****************************************************************
      
c     ****************************************************************
      zeil_geo(1)='Prozessor'
      zeil_geo(2)='npoin'
      zeil_geo(3)='nelem'

      zeil_rbe(1)='Prozessor'
      zeil_rbe(2)='ndisp'
      zeil_rbe(3)='nwand'
      zeil_rbe(4)='npres'
      zeil_rbe(5)='nsyme'
      zeil_rbe(6)='nzykl'
      zeil_rbe(7)='nconv'
      zeil_rbe(8)='ntemp'                   

        do 120 i=2,120
         text_1(i-1:i)='*'
         text_2(i-1:i)='-'
 120    continue

      if (myid.eq.0) then

          icom=lentb(comment)
          write(lupro,*)
          write(lupro,777) text_1
          write(lupro,555)comment(1:icom)
          write(lupro,555)text_2(1:icom)
          write(lupro,*) 
          write(lupro,888) (zeil_geo(j),j=1,3)
          do 201 i=1,numprocs
            write(lupro,999) i,(dim_geo(j,i),j=1,2)
 201      continue

          if (parti_rbe) then
             write(lupro,*) 
             write(lupro,888) (zeil_rbe(j),j=1,8)
             do 301 i=1,numprocs
               write(lupro,999) i,(dim_rbe(j,i),j=1,7)
 301         continue
          endif

          write(lupro,777) text_2

          zeil_geo(1)='Maximale Knotennummer :'
          zeil_geo(2)='Gesamtanzahl Knoten   :'
          zeil_rbe(1)='Maximale Elementnummer:'
          zeil_rbe(2)='Gesamtanzahl Elemente :'

          write(lupro,666) zeil_geo(1),knmax_num,zeil_geo(2),npoin_ges
          write(lupro,666) zeil_rbe(1),elmax_num,zeil_rbe(2),nelem_ges

          write(lupro,777) text_1
          write(lupro,*)

 444      format(1x,2(A31,1x,i8,2x))
 666      format(1x,2(A23,1x,i8,2x))
 777      format(1x,A70)
 555      format(1x,A)
 888      format(1x,A9,4x,10(A5,3x))
 999      format(1x,i3,8x,10(i7,1x))

 885      format(1x,A9,4x,2(A5,3x),2(A6,2x))

      endif
c     ****************************************************************

      return
      end
