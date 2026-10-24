package WhenBodyCalls "A call a when body reads in several statements, evaluated once"
  function stats "The sum of u, and u times k"
    input Real u[:];
    input Real k;
    output Real s;
    output Real y[size(u, 1)];
  algorithm
    s := sum(u);
    y := k * u;
  end stats;

  model Sampled "As MSL's RealFFT: samples into buf, then a tuple call of buf in an if-statement"
    Real buf[4](each start = 0, each fixed = true);
    Integer n(start = 0, fixed = true);
    Real s(start = 0, fixed = true);
    Real y[4](each start = 0, each fixed = true);
    Real s2(start = 0, fixed = true);
    Real y2[4](each start = 0, each fixed = true);
  algorithm
    when sample(0.1, 0.1) then
      n := pre(n) + 1;
      if n <= 4 then
        buf[n] := n;
      end if;
    end when;
    when time > 0.55 then
      if n >= 4 then
        (s2, y2) := stats(buf, 3.0);
      end if;
    end when;
    when terminal() then
      if n < 4 then
        s := -1;
      else
        (s, y) := stats(buf, 2.0);
      end if;
    end when;
  end Sampled;

  model TwoBodies "Bodies that share a scope: two terminal bodies, a when's elsewhen arms"
    Real buf[4](each start = 0, each fixed = true);
    Integer n(start = 0, fixed = true);
    Real s(start = 0, fixed = true);
    Real y[4](each start = 0, each fixed = true);
    Real s3(start = 0, fixed = true);
    Real y3[4](each start = 0, each fixed = true);
    Real s4(start = 0, fixed = true);
    Real y4[4](each start = 0, each fixed = true);
    Real s5(start = 0, fixed = true);
    Real y5[4](each start = 0, each fixed = true);
  algorithm
    when sample(0.1, 0.1) then
      n := pre(n) + 1;
      if n <= 4 then
        buf[n] := n;
      end if;
    end when;
    when time > 0.55 then
      if n >= 4 then
        (s4, y4) := stats(buf, 4.0);
      end if;
    elsewhen time > 0.75 then
      if n >= 4 then
        (s5, y5) := stats(buf, 5.0);
      end if;
    end when;
    when terminal() then
      if n >= 4 then
        (s, y) := stats(buf, 2.0);
      end if;
    end when;
    when terminal() then
      if n >= 4 then
        (s3, y3) := stats(buf, 3.0);
      end if;
    end when;
  end TwoBodies;
end WhenBodyCalls;
