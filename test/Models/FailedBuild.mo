package FailedBuild "A build that fails after its module made an external table"
  class TimeTable41 "Modelica.Blocks.Types.ExternalCombiTimeTable of MSL 4.1: its constructor calls init3"
    extends ExternalObject;
    function constructor
      input String tableName;
      input String fileName;
      input Real table[:, :];
      input Real startTime;
      input Integer columns[:];
      input Integer smoothness;
      input Integer extrapolation;
      input Real shiftTime;
      input Integer timeEvents;
      input Boolean verboseRead;
      input String delimiter;
      input Integer nHeaderLines;
      output TimeTable41 externalCombiTimeTable;
    external "C" externalCombiTimeTable = ModelicaStandardTables_CombiTimeTable_init3(fileName, tableName,
      table, size(table, 1), size(table, 2), startTime, columns, size(columns, 1), smoothness, extrapolation,
      shiftTime, timeEvents, verboseRead, delimiter, nHeaderLines)
      annotation(Library = {"ModelicaStandardTables", "ModelicaIO", "ModelicaMatIO", "zlib"});
    end constructor;

    function destructor
      input TimeTable41 externalCombiTimeTable;
    external "C" ModelicaStandardTables_CombiTimeTable_close(externalCombiTimeTable)
      annotation(Library = {"ModelicaStandardTables", "ModelicaIO", "ModelicaMatIO", "zlib"});
    end destructor;
  end TimeTable41;

  function getTimeTableValue "Modelica.Blocks.Tables.Internal.getTimeTableValueNoDer"
    input TimeTable41 tableID;
    input Integer icol;
    input Real timeIn;
    input Real nextTimeEvent;
    input Real pre_nextTimeEvent;
    output Real y;
  external "C" y = ModelicaStandardTables_CombiTimeTable_getValue(tableID, icol, timeIn, nextTimeEvent, pre_nextTimeEvent)
    annotation(Library = {"ModelicaStandardTables", "ModelicaIO", "ModelicaMatIO", "zlib"});
  end getTimeTableValue;

  model TableAtFailedBuild "a time table and an initial equation no start satisfies: the build fails after its module made the table"
    parameter Real table[2, 2] = [0, 0; 1, 2];
    parameter TimeTable41 tab = TimeTable41("NoName", "NoName", table, 0.0, {2}, 1, 2, 0.0, 3, false, ",", 0);
    Real y = getTimeTableValue(tab, 1, time, 1e60, 1e60);
    Real x(start = 1);
  initial equation
    x * x = -1;
  equation
    der(x) = y - x;
  end TableAtFailedBuild;

end FailedBuild;
