/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include <math.h>
#include "fix_wall_lj126.h"
#include "atom.h"
#include "error.h"
#include <stdio.h>
#include <stdlib.h>
#include "comm.h"
#include "update.h"
#include "force.h"
#include "random_mars.h"
#include <string.h>
#include "group.h"
#include "respa.h"
#include "memory.h"

using namespace LAMMPS_NS;
using namespace FixConst;

#define EPSILON 1.0e-10

/* ---------------------------------------------------------------------- */

FixWallLJ126::FixWallLJ126(LAMMPS *lmp, int narg, char **arg) :
  FixWall(lmp, narg, arg) {}

/* ---------------------------------------------------------------------- */

void FixWallLJ126::precompute(int m)
{

  seed = myseed[m];

  delete random11;
  delete random12;
  delete random13;
  delete random21;
  delete random22;
  delete random23;
  delete random31;
  delete random32;
  delete random33;
  random11 = new RanMars(lmp,seed + 10000 + comm->me);
  random12 = new RanMars(lmp,seed + 20000 + comm->me);
  random13 = new RanMars(lmp,seed + 30000 + comm->me);
  random21 = new RanMars(lmp,seed + 40000 + comm->me);
  random22 = new RanMars(lmp,seed + 50000 + comm->me);
  random23 = new RanMars(lmp,seed + 60000 + comm->me);
  random31 = new RanMars(lmp,seed + 70000 + comm->me);
  random32 = new RanMars(lmp,seed + 80000 + comm->me);
  random33 = new RanMars(lmp,seed + 90000 + comm->me);

}

/* ----------------------------------------------------------------------
   interaction of all particles in group with a wall
   m = index of wall coeffs
   which = xlo,xhi,ylo,yhi,zlo,zhi
   error if any particle is on or behind wall
------------------------------------------------------------------------- */

void FixWallLJ126::wall_particle(int m, int which, double coord)
{
  // define further varaibles

  int i; 
  double delvx,delvy,delvz;
  double r,wd;
  double mysigma_C,mysigma_S,dot;
  
  double fc_x,fc_y,fc_z;
  double fD_trans_c_x,fD_trans_c_y,fD_trans_c_z;
  double fD_trans_s_x,fD_trans_s_y,fD_trans_s_z;
  double fD_rot_s_x,fD_rot_s_y,fD_rot_s_z;
  double fR_c_x,fR_c_y,fR_c_z;
  double fR_s_x,fR_s_y,fR_s_z;
  double nx,ny,nz;
  double omegasum_x,omegasum_y,omegasum_z;
  double cross_x,cross_y,cross_z;
  double randnum11,randnum12,randnum13,randnum21,randnum22,randnum23,randnum31,randnum32,randnum33;3;
  double fs_x,fs_y,fs_z;
  double torque_x,torque_y,torque_z;

  // calculate dpd_sigma

  double boltz = force->boltz;
  mysigma_C = sqrt(2.0*boltz*T_my[m]*mygamma_C[m]);
  mysigma_S = sqrt(2.0*boltz*T_my[m]*mygamma_S[m]);

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double **omega = atom->omega;
  double **torque = atom->torque;
  double *radius = atom->radius;
  double *rmass = atom->rmass;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double dtinvsqrt = 1.0/sqrt(update->dt);
  double delta_t = update->dt;

  int dim = which / 2;
  int side = which % 2;
  if (side == 0) side = -1;

  int onflag = 0;
  double rinv,rinv2,rinv6;

  for (int i = 0; i < nlocal; i++)
    // calculate the distance between wall and beads
    if (mask[i] & groupbit) {
      if (side < 0){
			r = x[i][dim] - coord;
	  }
      else{
			r = coord - x[i][dim];	  		  
	  }
      if (r >= cutoff[m]) continue;
      if (r <= 0.0) {		  
        onflag = 1;
        continue;
      }
    if (r < EPSILON) continue; 

      delvx = v[i][0] - myvx_wall[m];
      delvy = v[i][1];
      delvz = v[i][2];    
      wd = 1.0 - r/cutoff[m];
      // froce
        // conservative force = a0 * wd	
		rinv=1/r;
		rinv2=rinv*rinv;
		rinv6=rinv2*rinv2*rinv2;
        fc_x = 0.0;
		fc_y = -side*rinv6*(48*rinv6-24)*rinv;//mya0[m]*wd;
		fc_z = 0.0;
		
		nx = 0;
        ny = 1;
        nz = 0;
        dot = delvx*nx+delvy*ny+delvz*nz;
		
		// dissipative froce in translational       	
		// centeral
		fD_trans_c_x = -mygamma_C[m]*wd*wd*dot*nx;
		fD_trans_c_y = -mygamma_C[m]*wd*wd*dot*ny;
		fD_trans_c_z = -mygamma_C[m]*wd*wd*dot*nz;
		 
        		
		// shear		   
		fD_trans_s_x = -mygamma_S[m]*wd*wd*(delvx-dot*nx);
		fD_trans_s_y = -mygamma_S[m]*wd*wd*(delvy-dot*ny);
		fD_trans_s_z = -mygamma_S[m]*wd*wd*(delvz-dot*nz);
		   
		// dissipative froce in rotational
		omegasum_x = omega[i][0];
		omegasum_y = omega[i][1];
		omegasum_z = omega[i][2];
		cross_x = r*omegasum_z;
		cross_y = 0.0;
		cross_z = -omegasum_x*r;
        fD_rot_s_x = -mygamma_S[m]*wd*wd*cross_x;
		fD_rot_s_y = -mygamma_S[m]*wd*wd*cross_y;
		fD_rot_s_z = -mygamma_S[m]*wd*wd*cross_z;
		
		// random force
		randnum11 = random11->gaussian();
		randnum12 = random12->gaussian();
		randnum13 = random13->gaussian();
		randnum22 = random22->gaussian();
		randnum23 = random23->gaussian();
		randnum33 = random33->gaussian();
		// centeral
		double sqrt_dt=sqrt(delta_t);
		fR_c_x = wd*mysigma_C*(randnum11+randnum22+randnum33)/1.73205081/sqrt_dt;
		fR_c_y = wd*mysigma_C*(randnum11+randnum22+randnum33)/1.73205081/sqrt_dt;
		fR_c_z = wd*mysigma_C*(randnum11+randnum22+randnum33)/1.73205081/sqrt_dt;
		   
		// shear
		fR_s_x = wd*mysigma_S*((randnum12-randnum21)*ny+(randnum13-randnum31)*nz)/1.414213562373/sqrt_dt;
		fR_s_y = wd*mysigma_S*((randnum21-randnum12)*nx+(randnum23-randnum32)*nz)/1.414213562373/sqrt_dt;
		fR_s_z = wd*mysigma_S*((randnum31-randnum13)*nx+(randnum32-randnum23)*ny)/1.414213562373/sqrt_dt;
		   
        f[i][0] += fc_x+fD_trans_c_x+fD_trans_s_x+fD_rot_s_x+fR_c_x+fR_s_x;
        f[i][1] += fc_y+fD_trans_c_y+fD_trans_s_y+fD_rot_s_y+fR_c_y+fR_s_y;
        f[i][2] += fc_z+fD_trans_c_z+fD_trans_s_z+fD_rot_s_z+fR_c_z+fR_s_z;
		
		// torque
		fs_x = fD_trans_s_x+fD_rot_s_x+fR_s_x;
		fs_y = fD_trans_s_y+fD_rot_s_y+fR_s_y;
		fs_z = fD_trans_s_z+fD_rot_s_z+fR_s_z;
		torque_x = r*fs_z;
		torque_y = 0.0;
		torque_z = -fs_x*r;
		torque[i][0] -= torque_x;
		torque[i][1] -= torque_y;
		torque[i][2] -= torque_z;    
    }
  
  if (onflag) error->one(FLERR,"Particle on or inside fix wall surface");
}

