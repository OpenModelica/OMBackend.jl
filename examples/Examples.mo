// The example models of OMBackend: flattened by OMFrontend, they are the backend's input
// (ExampleModels.jl), for the tests and the demo (simulate_examples.jl).
package Examples
  model HelloWorld "der(x) = -a*x: x(t) = exp(-t)"
    Real x(start = 1, fixed = true);
    parameter Real a = 1;
  equation
    der(x) = -a*x;
  end HelloWorld;

  model LotkaVolterra "predator-prey"
    parameter Real alpha = 0.1 "Reproduction rate of prey";
    parameter Real beta = 0.02 "Mortality rate of predator per prey";
    parameter Real gamma = 0.4 "Mortality rate of predator";
    parameter Real delta = 0.02 "Reproduction rate of predator per prey";
    Real x(start = 10, fixed = true) "Prey population";
    Real y(start = 10, fixed = true) "Predator population";
  equation
    der(x) = x*(alpha - beta*y);
    der(y) = y*(delta*x - gamma);
  end LotkaVolterra;

  model VanDerPol "Van der Pol oscillator"
    Real x(start = 1, fixed = true);
    Real y(start = 1, fixed = true);
    parameter Real lambda = 0.3;
  equation
    der(x) = y;
    der(y) = -x + lambda*(1 - x*x)*y;
  end VanDerPol;

  model BouncingBall "hybrid: a ball bouncing on the ground"
    parameter Real e = 0.7 "coefficient of restitution";
    parameter Real g = 9.81 "gravity acceleration";
    Real h(start = 1, fixed = true) "height of ball";
    Real v(fixed = true) "velocity of ball";
  equation
    der(v) = -g;
    der(h) = v;
    when h <= 0 then
      reinit(v, -e*pre(v));
    end when;
  end BouncingBall;

  model SimpleMechanicalSystem "two inertias and a spring"
    parameter Real J1 = 1, J2 = 1;
    constant Real C = 1;
    Real u;
    Real Phi_1, Phi_2;
    Real omega_1, omega_2;
    Real tau_1, tau_2, tau_3;
  initial equation
    omega_1 = 0;
    omega_2 = 0;
    Phi_1 = 0;
    Phi_2 = 0;
  equation
    u = time;
    tau_1 = u;
    tau_2 = C*(Phi_2 - Phi_1);
    tau_3 = -tau_2;
    der(Phi_1) = omega_1;
    der(Phi_2) = omega_2;
    der(omega_1) = (tau_1 + tau_2)/J1;
    der(omega_2) = tau_3/J2;
  end SimpleMechanicalSystem;

  model SimpleCircuit "a source, two resistors, a capacitor and an inductor"
    parameter Real A = 12.0;
    parameter Real C = 1;
    parameter Real L = 0.01;
    parameter Real R1 = 10.0;
    parameter Real R2 = 100.0;
    parameter Real f = 1.0;
    Real i1;
    Real i2;
    Real i;
    Real u1;
    Real u2;
    Real u3;
    Real u4(start = 0.007);
    Real u;
  equation
    u3 = R2*i2;
    u = A*sin(2*3.1415*f*time);
    u = u1 + u2;
    u1 = R1*i1;
    i = i2 + i1;
    der(u2) = i1/C;
    u = u4 + u3;
    der(i2) = u4/L;
  end SimpleCircuit;
end Examples;
