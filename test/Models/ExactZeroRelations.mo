package ExactZeroRelations "A cluster relation whose crossing function is exactly zero takes its literal value (MSL ThyristorBridge2mPulse_DC_Drive's fire_n = 0 > 0 after a timer reset); values from omc 1.27.1"
  model Latch "x < 0 with the state x reset to exactly 0 at 0.5 and held there: false from then, so the fire at 0.7 turns the latch off"
    Real x(start = -1, fixed = true);
    Boolean fire = time > 0.7;
    Boolean off(start = true, fixed = true);
  equation
    der(x) = if time < 0.5 then 1 else 0;
    when time > 0.5 then
      reinit(x, 0);
    end when;
    off = x < 0 or pre(off) and not fire;
  end Latch;
end ExactZeroRelations;
