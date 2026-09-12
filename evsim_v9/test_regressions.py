"""Targeted regressions for observed energy, dynamics, clock and reporting failures."""
import copy
import unittest
import numpy as np
import env20 as E
import models as M
import route20 as R
from analyze import score, reference_means, VAL_OFFSETS, REP_OFFSETS


class Regressions(unittest.TestCase):
    def setUp(self):
        E.REWARD_MODE = 'L'; E.CURVE_MODE = 'envelope'; E.SHAPE_C = 0.

    def test_launch_energy_not_free(self):
        for v in (0.5, 1.5, 1.999, 2.0, 2.001):
            batt, fric, wheel = M.v1_power(v, 2.6, 0., .85, 288.15)
            self.assertGreater(batt - M.AUX.p_base_dc, wheel)
            self.assertEqual(fric, 0.)
        b0 = M.v1_power(1.999, 2.6, 0., .85, 288.15)[0]
        b1 = M.v1_power(2.001, 2.6, 0., .85, 288.15)[0]
        self.assertLess(abs(b1-b0), 20.)

    def test_low_speed_regen_still_disabled(self):
        batt, fric, wheel = M.v1_power(1.5, -3.5, 0., .85, 288.15)
        self.assertEqual(batt, M.AUX.p_base_dc)
        self.assertAlmostEqual(fric, -wheel)

    def test_infeasible_red_does_not_invent_brakes_or_teleport(self):
        e = E.Route20(offsets=[0.] * len(R.SIGNALS)); e.reset()
        e.x = R.SIGNALS[0]-1.; e.v=22.; e.t=30.; e.a=0.
        x0, v0 = e.x, e.v
        _, _, done, info = e.step(2.6)
        self.assertGreaterEqual(e.a, E.A_LO)
        self.assertAlmostEqual(e.v, v0 + E.DT*e.a)
        self.assertAlmostEqual(e.x, x0 + E.DT*(v0+e.v)/2)
        self.assertGreater(e.x, R.SIGNALS[0])
        self.assertTrue(done and info['red_crossing'])
        self.assertGreater(e.terminal_pen, 0.)

    def test_signal_is_checked_at_crossing_time(self):
        e = E.Route20(offsets=[0.] * len(R.SIGNALS)); e.reset()
        e.x=R.SIGNALS[0]-1.; e.v=4.; e.t=29.9; e.a=0.
        self.assertTrue(R.signal_green(R.SIGNALS[0], e.t, 0.))
        _, _, done, info = e.step(0.)
        self.assertTrue(done and info['red_crossing'])

    def test_clock_contains_both_deadlines(self):
        a=E.Route20(offsets=[0.] * len(R.SIGNALS)); a.reset()
        b=copy.deepcopy(a); b.t_end+=10.
        self.assertFalse(np.array_equal(a.obs(), b.obs()))
        b=copy.deepcopy(a); b.t+=10.; b.t_end+=10.
        self.assertEqual(a.obs()[11], b.obs()[11])
        self.assertNotEqual(a.obs()[12], b.obs()[12])
        self.assertEqual(len(a.obs()), E.OBS_DIM)

    def test_reset_from_clears_episode_ledger(self):
        e=E.Route20();e.reset();e.step(2.6)
        e.reset_from(1000., 10., 55.)
        self.assertEqual(e.steps, 0);self.assertEqual(e.e_batt, 0.)
        self.assertEqual(e.soc, e.soc0)

    def test_reward_ledger_and_integration_step_jerk(self):
        e=E.Route20(offsets=[22.5]*len(R.SIGNALS));e.reset()
        total=0.; j=[]
        while True:
            old=e.a
            _,r,d,i=e.step(float(np.clip((20-e.v)/E.DT,-3.5,2.6)))
            j.append((e.a-old)/E.DT);total+=r
            self.assertGreaterEqual(e.a,E.A_LO-1e-10)
            self.assertLessEqual(e.a,E.A_HI+1e-10)
            if d:break
        truth=-e.e_batt/1e5-E.LAM_T*e.t-e.e_overspeed_pen-e.terminal_pen
        self.assertAlmostEqual(total,truth,places=8)
        self.assertAlmostEqual(e.jerk_sq,float(np.square(j).sum()),places=7)
        self.assertGreater(e.jerk_max,E.J_MAX)  # emergency comfort override is disclosed

    def test_split_and_reference_are_metadata_based(self):
        grid=[]; refs={'grid':{}}
        for soc,T in [(0.85,288.15),(0.90,263.15),(0.95,263.15)]:
            for off in [0.,22.5,45.,67.5]:
                r=10. if off in VAL_OFFSETS else -10.
                grid.append(dict(soc=soc,T=T,offset=off,R=r,arr=True))
                refs['grid'][f'{soc}_{T}_{off}']={'ref':{'R':r}, 'humans':{n:{'R':r} for n in ('attentive','normal','distracted')}}
        np.random.default_rng(0).shuffle(grid)
        self.assertEqual(score({'grid':grid},VAL_OFFSETS),(10.,6))
        rep=[x for x in grid if x['offset'] in REP_OFFSETS]
        self.assertEqual(reference_means(refs,rep)['normal'],-10.)


if __name__=='__main__':unittest.main(verbosity=2)
