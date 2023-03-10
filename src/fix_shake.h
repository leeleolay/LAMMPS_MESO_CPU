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

#ifndef FIX_SHAKE_H
#define FIX_SHAKE_H

#include "fix.h"

namespace LAMMPS_NS {

class FixShake : public Fix {
 public:
  FixShake(class LAMMPS *, int, char **);
  ~FixShake();
  int setmask();
  void init();
  void setup(int);
  void pre_neighbor();
  void post_force(int);
  void post_force_respa(int, int, int);

  double memory_usage();
  void grow_arrays(int);
  void copy_arrays(int, int);
  int pack_exchange(int, double *);
  int unpack_exchange(int, double *);
  int pack_comm(int, int *, double *, int, int *);
  void unpack_comm(int, int, double *);

  int dof(int);
  void reset_dt();

 private:
  int me,nprocs;
  double PI;
  double tolerance;                      // SHAKE tolerance
  int max_iter;                          // max # of SHAKE iterations
  int output_every;                      // SHAKE stat output every so often
  int next_output;                       // timestep for next output

                                         // settings from input command
  int *bond_flag,*angle_flag;            // bond/angle types to constrain
  int *type_flag;                        // constrain bonds to these types
  double *mass_list;                     // constrain bonds to these masses
  int nmass;                             // # of masses in mass_list

  double *bond_distance,*angle_distance; // constraint distances

  int ifix_respa;                        // rRESPA fix needed by SHAKE
  int nlevels_respa;                     // copies of needed rRESPA variables
  int *loop_respa;
  double *step_respa;

  double **x,**v,**f;                    // local ptrs to atom class quantities
  double *mass,*rmass;
  int *type;
  int nlocal;
                                         // atom-based arrays
  int *shake_flag;                       // 0 if atom not in SHAKE cluster
                                         // 1 = size 3 angle cluster
                                         // 2,3,4 = size of bond-only cluster
  int **shake_atom;                      // global IDs of atoms in cluster
                                         // central atom is 1st
                                         // lowest global ID is 1st for size 2
  int **shake_type;                      // bondtype of each bond in cluster
                                         // for angle cluster, 3rd value
                                         //   is angletype
  double **xshake;                       // unconstrained atom coords

  int vflag;                            // virial flag
  double dtv,dtfsq;                     // timesteps for trial move
  double dtf_inner,dtf_innerhalf;       // timesteps for rRESPA trial move

  int *list;                            // list of clusters to SHAKE
  int nlist,maxlist;                    // size and max-size of list

                                        // stat quantities
  int *b_count,*b_count_all;            // counts for each bond type
  double *b_ave,*b_max,*b_min;          // ave/max/min dist for each bond type
  double *b_ave_all,*b_max_all,*b_min_all;   // MPI summing arrays
  int *a_count,*a_count_all;            // ditto for angle types
  double *a_ave,*a_max,*a_min;
  double *a_ave_all,*a_max_all,*a_min_all;

  void find_clusters();
  int masscheck(double);
  void unconstrained_update();
  void shake2(int);
  void shake3(int);
  void shake4(int);
  void shake3angle(int);
  void stats();
  int bondfind(int, int, int);
  int anglefind(int, int, int);
};

}

#endif
