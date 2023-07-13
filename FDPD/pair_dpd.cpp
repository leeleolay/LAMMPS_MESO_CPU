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

/* ----------------------------------------------------------------------
   Contributing author: Kurt Smith (U Pittsburgh)
------------------------------------------------------------------------- */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "pair_dpd.h"
#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "update.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "random_mars.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

#define EPSILON 1.0e-10

/* ---------------------------------------------------------------------- */

PairDPD::PairDPD(LAMMPS *lmp) : Pair(lmp)
{
  writedata = 1;
  random11 = NULL;
  random12 = NULL;
  random13 = NULL;
  random21 = NULL;
  random22 = NULL;
  random23 = NULL;
  random31 = NULL;
  random32 = NULL;
  random33 = NULL;
}

/* ---------------------------------------------------------------------- */

PairDPD::~PairDPD()
{
  if (allocated)
  {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(a0);
    memory->destroy(gamma_C);
    memory->destroy(gamma_S);
    memory->destroy(sigma_C);
    memory->destroy(sigma_S);
  }

  if (random11)
    delete random11;
  if (random12)
    delete random12;
  if (random13)
    delete random13;
  if (random21)
    delete random22;
  if (random22)
    delete random22;
  if (random23)
    delete random22;
  if (random31)
    delete random23;
  if (random32)
    delete random22;
  if (random33)
    delete random33;
}

/* ---------------------------------------------------------------------- */

void PairDPD::compute(int eflag, int vflag)
{
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, evdwl, fpair;
  double vxtmp, vytmp, vztmp, delvx, delvy, delvz;
  double rsq, r, rinv, dot, wd, factor_dpd;

  double fc_x, fc_y, fc_z;
  double fD_trans_c_x, fD_trans_c_y, fD_trans_c_z;
  double fD_trans_s_x, fD_trans_s_y, fD_trans_s_z;
  double fD_rot_s_x, fD_rot_s_y, fD_rot_s_z;
  double fR_c_x, fR_c_y, fR_c_z;
  double fR_s_x, fR_s_y, fR_s_z;
  double nx, ny, nz;
  double lamda_ij, lamda_ji, omegasum_x, omegasum_y, omegasum_z;
  double cross_x, cross_y, cross_z;
  double randnum11, randnum12, randnum13, randnum21, randnum22, randnum23, randnum31, randnum32, randnum33;
  double fs_x, fs_y, fs_z;
  double torque_x, torque_y, torque_z;

  int *ilist, *jlist, *numneigh, **firstneigh;

  evdwl = 0.0;
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  int *type = atom->type;
  double **omega = atom->omega;
  double **torque = atom->torque;
  double *radius = atom->radius;
  double *rmass = atom->rmass;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double dtinvsqrt = 1.0 / sqrt(update->dt);
  double delta_t = update->dt;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++)
  {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    vxtmp = v[i][0];
    vytmp = v[i][1];
    vztmp = v[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++)
    {
      j = jlist[jj];
      factor_dpd = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype])
      {
        r = sqrt(rsq);
        if (r < EPSILON)
          continue; // r can be 0.0 in DPD systems
        rinv = 1.0 / r;
        delvx = vxtmp - v[j][0];
        delvy = vytmp - v[j][1];
        delvz = vztmp - v[j][2];

        // weight function of conservative force
        wd = 1.0 - r / cut[itype][jtype];

        // unit vector
        nx = delx * rinv;
        ny = dely * rinv;
        nz = delz * rinv;
        // froce
        // conservative force = a0 * wd
        fc_x = a0[itype][jtype] * wd * nx;
        fc_y = a0[itype][jtype] * wd * ny;
        fc_z = a0[itype][jtype] * wd * nz;

        // dissipative froce in translational
        dot = delvx * nx + delvy * ny + delvz * nz;
        // centeral
        fD_trans_c_x = -gamma_C[itype][jtype] * wd * wd * dot * nx;
        fD_trans_c_y = -gamma_C[itype][jtype] * wd * wd * dot * ny;
        fD_trans_c_z = -gamma_C[itype][jtype] * wd * wd * dot * nz;

        // shear
        fD_trans_s_x = -gamma_S[itype][jtype] * wd * wd * (delvx - dot * nx);
        fD_trans_s_y = -gamma_S[itype][jtype] * wd * wd * (delvy - dot * ny);
        fD_trans_s_z = -gamma_S[itype][jtype] * wd * wd * (delvz - dot * nz);

        // dissipative froce in rotational
        lamda_ij = radius[i] / (radius[i] + radius[j]);
        lamda_ji = radius[j] / (radius[i] + radius[j]);
        omegasum_x = lamda_ij * omega[i][0] + lamda_ji * omega[j][0];
        omegasum_y = lamda_ij * omega[i][1] + lamda_ji * omega[j][1];
        omegasum_z = lamda_ij * omega[i][2] + lamda_ji * omega[j][2];
        cross_x = dely * omegasum_z - omegasum_y * delz;
        cross_y = delz * omegasum_x - omegasum_z * delx;
        cross_z = delx * omegasum_y - omegasum_x * dely;
        fD_rot_s_x = -gamma_S[itype][jtype] * wd * wd * cross_x;
        fD_rot_s_y = -gamma_S[itype][jtype] * wd * wd * cross_y;
        fD_rot_s_z = -gamma_S[itype][jtype] * wd * wd * cross_z;

        // random force
        randnum11 = random11->gaussian();
        randnum12 = random12->gaussian();
        randnum13 = random13->gaussian();
        randnum21 = random22->gaussian();
        randnum22 = random22->gaussian();
        randnum23 = random23->gaussian();
        randnum31 = random33->gaussian();
        randnum32 = random33->gaussian();
        randnum33 = random33->gaussian();
        // centeral
        double sqrt_dt = sqrt(delta_t);
        fR_c_x = wd * sigma_C[itype][jtype] * (randnum11 + randnum22 + randnum33) / 1.73205081 * nx / sqrt_dt;
        fR_c_y = wd * sigma_C[itype][jtype] * (randnum11 + randnum22 + randnum33) / 1.73205081 * ny / sqrt_dt;
        fR_c_z = wd * sigma_C[itype][jtype] * (randnum11 + randnum22 + randnum33) / 1.73205081 * nz / sqrt_dt;

        // shear
        fR_s_x = wd * sigma_S[itype][jtype] * ((randnum12 - randnum21) * ny + (randnum13 - randnum31) * nz) / 1.414213562373 / sqrt_dt;
        fR_s_y = wd * sigma_S[itype][jtype] * ((randnum21 - randnum12) * nx + (randnum23 - randnum32) * nz) / 1.414213562373 / sqrt_dt;
        fR_s_z = wd * sigma_S[itype][jtype] * ((randnum31 - randnum13) * nx + (randnum32 - randnum23) * ny) / 1.414213562373 / sqrt_dt;

        f[i][0] += fc_x + fD_trans_c_x + fD_trans_s_x + fD_rot_s_x + fR_c_x + fR_s_x;
        f[i][1] += fc_y + fD_trans_c_y + fD_trans_s_y + fD_rot_s_y + fR_c_y + fR_s_y;
        f[i][2] += fc_z + fD_trans_c_z + fD_trans_s_z + fD_rot_s_z + fR_c_z + fR_s_z;

        // torque
        fs_x = fD_trans_s_x + fD_rot_s_x + fR_s_x;
        fs_y = fD_trans_s_y + fD_rot_s_y + fR_s_y;
        fs_z = fD_trans_s_z + fD_rot_s_z + fR_s_z;
        torque_x = dely * fs_z - fs_y * delz;
        torque_y = delz * fs_x - fs_z * delx;
        torque_z = delx * fs_y - fs_x * dely;
        torque[i][0] -= lamda_ij * torque_x;
        torque[i][1] -= lamda_ij * torque_y;
        torque[i][2] -= lamda_ij * torque_z;
        if (newton_pair || j < nlocal)
        {
          f[j][0] -= fc_x + fD_trans_c_x + fD_trans_s_x + fD_rot_s_x + fR_c_x + fR_s_x;
          f[j][1] -= fc_y + fD_trans_c_y + fD_trans_s_y + fD_rot_s_y + fR_c_y + fR_s_y;
          f[j][2] -= fc_z + fD_trans_c_z + fD_trans_s_z + fD_rot_s_z + fR_c_z + fR_s_z;

          torque[j][0] -= lamda_ji * torque_x;
          torque[j][1] -= lamda_ji * torque_y;
          torque[j][2] -= lamda_ji * torque_z;
        }
      }
    }
  }
  if (vflag_fdotr)
    virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairDPD::allocate()
{
  int i, j;
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (i = 1; i <= n; i++)
    for (j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");

  memory->create(cut, n + 1, n + 1, "pair:cut");
  memory->create(a0, n + 1, n + 1, "pair:a0");
  memory->create(gamma_C, n + 1, n + 1, "pair:gamma_C");
  memory->create(gamma_S, n + 1, n + 1, "pair:gamma_S");
  memory->create(sigma_S, n + 1, n + 1, "pair:sigma_C");
  memory->create(sigma_C, n + 1, n + 1, "pair:sigma_S");
  for (i = 0; i <= atom->ntypes; i++)
    for (j = 0; j <= atom->ntypes; j++)
      sigma_C[i][j] = sigma_S[i][j] = gamma_C[i][j] = gamma_S[i][j] = 0.0;
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairDPD::settings(int narg, char **arg)
{
  if (narg != 3)
    error->all(FLERR, "Illegal pair_style command");

  temperature = force->numeric(FLERR, arg[0]);
  cut_global = force->numeric(FLERR, arg[1]);
  seed = force->inumeric(FLERR, arg[2]);

  // initialize Marsaglia RNG with processor-unique seed

  if (seed <= 0)
    error->all(FLERR, "Illegal pair_style command");
  delete random11;
  delete random12;
  delete random13;
  delete random21;
  delete random22;
  delete random23;
  delete random31;
  delete random32;
  delete random33;
  random11 = new RanMars(lmp, seed + 10000 + comm->me);
  random12 = new RanMars(lmp, seed + 20000 + comm->me);
  random13 = new RanMars(lmp, seed + 30000 + comm->me);
  random21 = new RanMars(lmp, seed + 40000 + comm->me);
  random22 = new RanMars(lmp, seed + 50000 + comm->me);
  random23 = new RanMars(lmp, seed + 60000 + comm->me);
  random31 = new RanMars(lmp, seed + 70000 + comm->me);
  random32 = new RanMars(lmp, seed + 80000 + comm->me);
  random33 = new RanMars(lmp, seed + 90000 + comm->me);

  // reset cutoffs that have been explicitly set

  if (allocated)
  {
    int i, j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j])
          cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairDPD::coeff(int narg, char **arg)
{
  if (narg < 5 || narg > 6)
    error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated)
    allocate();

  int ilo, ihi, jlo, jhi;
  force->bounds(FLERR, arg[0], atom->ntypes, ilo, ihi);
  force->bounds(FLERR, arg[1], atom->ntypes, jlo, jhi);

  double a0_one = force->numeric(FLERR, arg[2]);
  double gamma_one_C = force->numeric(FLERR, arg[3]);
  double gamma_one_S = force->numeric(FLERR, arg[4]);

  double cut_one = cut_global;
  if (narg == 6)
    cut_one = force->numeric(FLERR, arg[5]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++)
  {
    for (int j = MAX(jlo, i); j <= jhi; j++)
    {
      a0[i][j] = a0_one;
      gamma_C[i][j] = gamma_one_C;
      gamma_S[i][j] = gamma_one_S;
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0)
    error->all(FLERR, "Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairDPD::init_style()
{
  if (comm->ghost_velocity == 0)
    error->all(FLERR, "Pair dpd requires ghost atoms store velocity");

  // if newton off, forces between atoms ij will be double computed
  // using different random numbers

  if (force->newton_pair == 0 && comm->me == 0)
    error->warning(FLERR,
                   "Pair dpd needs newton pair on for momentum conservation");

  neighbor->request(this, instance_me);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairDPD::init_one(int i, int j)
{
  if (setflag[i][j] == 0)
    error->all(FLERR, "All pair coeffs are not set");

  sigma_C[i][j] = sqrt(2.0 * force->boltz * temperature * gamma_C[i][j]);
  sigma_S[i][j] = sqrt(2.0 * force->boltz * temperature * gamma_S[i][j]);

  cut[j][i] = cut[i][j];
  a0[j][i] = a0[i][j];
  gamma_C[j][i] = gamma_C[i][j];
  gamma_S[j][i] = gamma_S[i][j];
  sigma_C[j][i] = sigma_C[i][j];
  sigma_S[j][i] = sigma_S[i][j];

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairDPD::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i, j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++)
    {
      fwrite(&setflag[i][j], sizeof(int), 1, fp);
      if (setflag[i][j])
      {
        fwrite(&a0[i][j], sizeof(double), 1, fp);
        fwrite(&gamma_C[i][j], sizeof(double), 1, fp);
        fwrite(&gamma_S[i][j], sizeof(double), 1, fp);
        fwrite(&cut[i][j], sizeof(double), 1, fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairDPD::read_restart(FILE *fp)
{
  read_restart_settings(fp);

  allocate();

  int i, j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++)
    {
      if (me == 0)
        fread(&setflag[i][j], sizeof(int), 1, fp);
      MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
      if (setflag[i][j])
      {
        if (me == 0)
        {
          fread(&a0[i][j], sizeof(double), 1, fp);
          fread(&gamma_C[i][j], sizeof(double), 1, fp);
          fread(&gamma_S[i][j], sizeof(double), 1, fp);
          fread(&cut[i][j], sizeof(double), 1, fp);
        }
        MPI_Bcast(&a0[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&gamma_C[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&gamma_S[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&cut[i][j], 1, MPI_DOUBLE, 0, world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairDPD::write_restart_settings(FILE *fp)
{
  fwrite(&temperature, sizeof(double), 1, fp);
  fwrite(&cut_global, sizeof(double), 1, fp);
  fwrite(&seed, sizeof(int), 1, fp);
  fwrite(&mix_flag, sizeof(int), 1, fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairDPD::read_restart_settings(FILE *fp)
{
  if (comm->me == 0)
  {
    fread(&temperature, sizeof(double), 1, fp);
    fread(&cut_global, sizeof(double), 1, fp);
    fread(&seed, sizeof(int), 1, fp);
    fread(&mix_flag, sizeof(int), 1, fp);
  }
  MPI_Bcast(&temperature, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&cut_global, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&seed, 1, MPI_INT, 0, world);
  MPI_Bcast(&mix_flag, 1, MPI_INT, 0, world);

  // initialize Marsaglia RNG with processor-unique seed
  // same seed that pair_style command initially specified

  if (random11)
    delete random11;
  random11 = new RanMars(lmp, seed + 10000 + comm->me);
  if (random12)
    delete random12;
  random12 = new RanMars(lmp, seed + 20000 + comm->me);
  if (random13)
    delete random13;
  random13 = new RanMars(lmp, seed + 30000 + comm->me);
  if (random21)
    delete random21;
  random21 = new RanMars(lmp, seed + 40000 + comm->me);
  if (random22)
    delete random22;
  random22 = new RanMars(lmp, seed + 50000 + comm->me);
  if (random23)
    delete random23;
  random23 = new RanMars(lmp, seed + 60000 + comm->me);
  if (random31)
    delete random31;
  random31 = new RanMars(lmp, seed + 70000 + comm->me);
  if (random32)
    delete random32;
  random32 = new RanMars(lmp, seed + 80000 + comm->me);
  if (random33)
    delete random33;
  random33 = new RanMars(lmp, seed + 90000 + comm->me);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairDPD::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp, "%d %g %g %g\n", i, a0[i][i], gamma_C[i][i], gamma_S[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairDPD::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp, "%d %d %g %g %g %g\n", i, j, a0[i][j], gamma_C[i][j], gamma_S[i][j], cut[i][j]);
}

/* ---------------------------------------------------------------------- */

double PairDPD::single(int /*i*/, int /*j*/, int itype, int jtype, double rsq,
                       double /*factor_coul*/, double factor_dpd, double &fforce)
{
  double r, rinv, wd, phi;

  r = sqrt(rsq);
  if (r < EPSILON)
  {
    fforce = 0.0;
    return 0.0;
  }

  rinv = 1.0 / r;
  wd = 1.0 - r / cut[itype][jtype];
  fforce = a0[itype][jtype] * wd * factor_dpd * rinv;

  phi = 0.5 * a0[itype][jtype] * cut[itype][jtype] * wd * wd;
  return factor_dpd * phi;
}
