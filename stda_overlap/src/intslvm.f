! This file is part of stda.
!
! Copyright (C) 2013-2019 Stefan Grimme
!
! stda is free software: you can redistribute it and/or modify it under
! the terms of the GNU Lesser General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
!
! stda is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU Lesser General Public License for more details.
!
! You should have received a copy of the GNU Lesser General Public License
! along with stda.  If not, see <https://www.gnu.org/licenses/>.
!
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      subroutine intslvm(ncent,nbf,nprims) 
      use stdacommon
      use intpack
      implicit real*8(a-h,o-z)                                                  

      real*8, allocatable ::r0(:)
      real*8, allocatable ::r1(:)
      real*8, allocatable ::r2(:)
      real*8, allocatable ::r3(:)
      real*8, allocatable ::r4(:)
      real*8, allocatable ::r5(:)
      real*8, allocatable ::r6(:)
      real*8, allocatable ::r7(:)
      real*8, allocatable ::r8(:)
      real*8, allocatable ::r9(:)
      integer*8 memneed,mp,nrecordlen,k,i1,lin8
      common/ prptyp / mprp 
      common /cema   / cen(3),xmolw
      common /amass  / ams(107)

      dimension v(3),point(3)

      call header('A O   I N T E G R A L S',0)

c overlap based neglect prim prefactor threshold      
      thr=1.d-7

c center of nuclear charge and molar mass
      sumwx=0.d0                                                                
      sumwy=0.d0                                                                
      sumwz=0.d0                                                                
      sumw=0.0d0
      xmolw=0.0d0
      do 10 i=1,ncent                                                        
         atmass=co(i,4)                                                      
         sumw=sumw+atmass                                                    
         sumwx=sumwx+atmass*co(i,1)                                       
         sumwy=sumwy+atmass*co(i,2)                                       
         sumwz=sumwz+atmass*co(i,3)                                       
         xmolw=xmolw+ams(idint(atmass))
   10 continue                                                               
      cen(1)=sumwx/sumw
      cen(2)=sumwy/sumw 
      cen(3)=sumwz/sumw  

      if(nbf.eq.0) then
         do i=1,nprims
            iaoat(i)=ipat(i)
         enddo
         nao=nprims
      else
         do i=1,nprims
            ii=ipat(i)
            iaoat(ipao(i))=ii
         enddo
         nao=nbf
      endif        

      mp=nao
      mp=mp*(mp+1)/2

      memneed=10*8*mp

      allocate(r1(mp),r2(mp),r3(mp),  
     .         r4(mp),r5(mp),r6(mp),  
     .         r7(mp),r8(mp),r9(mp),r0(mp),
     .         stat=ierr)
      if(ierr.ne.0)stop 'allocation failed in intslvm for AOs'

      open(unit=40,file='sint' ,status='replace')   
      open(unit=31,file='xlint',status='replace')
      open(unit=32,file='ylint',status='replace')
      open(unit=33,file='zlint',status='replace')
      open(unit=34,file='xmint',status='replace')
      open(unit=35,file='ymint',status='replace')
      open(unit=36,file='zmint',status='replace')
      open(unit=37,file='xvint',status='replace')
      open(unit=38,file='yvint',status='replace')
      open(unit=39,file='zvint',status='replace')

ccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c             overlap and dipole 
c
ccccccccccccccccccccccccccccccccccccccccccccccccccc
      
      point=0.0d0

      r0=0.0d0 
      r1=0.0d0
      r2=0.0d0
      r3=0.0d0
      r4=0.0d0
      r5=0.0d0
      r6=0.0d0
      r7=0.0d0
      r8=0.0d0
      r9=0.0d0

      do i=1,nprims
         iai=ipao(i)
         c1=cxip(i)
         do j=1,i-1
            iaj=ipao(j)
            iaa=max(iaj,iai)
            iii=min(iaj,iai)
            ij=iii+iaa*(iaa-1)/2 
            cf=c1*cxip(j)*2.0d0
c           prefactor            
            call propa0(opad1,point,v,1,i,j,s)
            if(s.gt.thr)then
              mprp=0
c             S            
              call propa1(opad1,point,v,1,i,j,s)
              r0(ij)=r0(ij)+v(1)*cf
c             R                    
              call propa1(opab1,point,v,3,i,j,s)
              r1(ij)=r1(ij)+v(1)*cf                
              r2(ij)=r2(ij)+v(2)*cf                
              r3(ij)=r3(ij)+v(3)*cf                
C             L              
              mprp=16
              call propa1(opam,point,v,3,i,j,s)
              ! note that s is changed by propa1 in this very case
              r4(ij)=r4(ij)+v(1)*cf                
              r5(ij)=r5(ij)+v(2)*cf                
              r6(ij)=r6(ij)+v(3)*cf                
C             V              
              mprp=0  
              call velo(i,j,v)
              r7(ij)=r7(ij)-v(1)*cf                
              r8(ij)=r8(ij)-v(2)*cf                
              r9(ij)=r9(ij)-v(3)*cf                
            endif
         enddo
         mprp=0
         call propa0(opad1,point,v,1,i,i,s)

         call propa1(opad1,point,v,1,i,i,s)
         ij=iai+iai*(iai-1)/2 
         cf=c1*c1
         r0(ij)=r0(ij)+v(1)*cf 
         call propa1(opab1,point,v,3,i,i,s)
         r1(ij)=r1(ij)+v(1)*cf    
         r2(ij)=r2(ij)+v(2)*cf    
         r3(ij)=r3(ij)+v(3)*cf    
      enddo
      
      ij=0
      do i=1,nao
         do j=1,i-1
            ij=lin8(i,j)
            r0(ij)=r0(ij)*0.50d0 
            r1(ij)=r1(ij)*0.50d0
            r2(ij)=r2(ij)*0.50d0
            r3(ij)=r3(ij)*0.50d0
            r4(ij)=r4(ij)*0.50d0
            r5(ij)=r5(ij)*0.50d0
            r6(ij)=r6(ij)*0.50d0
            r7(ij)=r7(ij)*0.50d0
            r8(ij)=r8(ij)*0.50d0
            r9(ij)=r9(ij)*0.50d0
         enddo         
      enddo


      ij=0
      do i=1,nao
         do j=1,i
           ij=lin8(i,j)
           if (i.eq.j) then
              write(40,"(2I8,F18.8)") i,j,r0(ij)
              write(31,"(2I8,F18.8)") i,j,r1(ij)
              write(32,"(2I8,F18.8)") i,j,r2(ij)
              write(33,"(2I8,F18.8)") i,j,r3(ij)
              write(34,"(2I8,F18.8)") i,j,r4(ij)
              write(35,"(2I8,F18.8)") i,j,r5(ij)
              write(36,"(2I8,F18.8)") i,j,r6(ij)
              write(37,"(2I8,F18.8)") i,j,r7(ij)
              write(38,"(2I8,F18.8)") i,j,r8(ij)
              write(39,"(2I8,F18.8)") i,j,r9(ij)
           else
               if (abs(r0(ij)).gt.1e-5) then
                  write(40,"(2I8,F18.8)") i,j,r0(ij)
               end if
               if (abs(r1(ij)).gt.1e-8) then
                  write(31,"(2I8,F18.8)") i,j,r1(ij)
               end if
               if (abs(r2(ij)).gt.1e-8) then
                  write(32,"(2I8,F18.8)") i,j,r2(ij)
               end if
               if (abs(r3(ij)).gt.1e-8) then
                  write(33,"(2I8,F18.8)") i,j,r3(ij)
               end if
               if (abs(r4(ij)).gt.1e-8) then
                  write(34,"(2I8,F18.8)") i,j,r4(ij)
               end if
               if (abs(r5(ij)).gt.1e-8) then
                  write(35,"(2I8,F18.8)") i,j,r5(ij)
               end if
               if (abs(r6(ij)).gt.1e-8) then
                  write(36,"(2I8,F18.8)") i,j,r6(ij)
               end if
               if (abs(r7(ij)).gt.1e-8) then
                  write(37,"(2I8,F18.8)") i,j,r7(ij)
               end if
               if (abs(r8(ij)).gt.1e-8) then
                  write(38,"(2I8,F18.8)") i,j,r8(ij)
               end if
               if (abs(r9(ij)).gt.1e-8) then
                  write(39,"(2I8,F18.8)") i,j,r9(ij)
               end if
           end if
         end do
      end do

      close(34)
      close(35)
      close(36)
      close(40) 
      close(31)
      close(32)
      close(33)
      close(37)
      close(38)
      close(39)
      
      deallocate(r0,r1,r2,r3,r4,r5,r6,r7,r8,r9)

      write(*,*) 'done.'
      end

****************************************************************************
* Given orbital indeces i1 and i2, lin() returns index in the linear array *
****************************************************************************

      integer*4 function lin(i1,i2)
      integer i1,i2
      integer*4 idum1,idum2
      idum1=max(i1,i2)
      idum2=min(i1,i2)
      lin=idum2+idum1*(idum1-1)/2
      return
      end

      integer*8 function lin8(i1,i2)
      integer i1,i2
      integer*8 idum1,idum2
      idum1=max(i1,i2)
      idum2=min(i1,i2)
      lin8=idum2+idum1*(idum1-1)/2
      return
      end


