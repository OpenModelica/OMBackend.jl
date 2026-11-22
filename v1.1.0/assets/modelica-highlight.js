// Modelica for highlight.js (no Modelica in highlight.js's language set). Documenter loads highlight.js through
// RequireJS and calls hljs.highlightAll() when the page is ready; this script waits for hljs, registers the
// language, and highlights the ```modelica blocks that are not highlighted yet.
(function () {
  function modelica(hljs) {
    var IDENT = /[A-Za-z_][A-Za-z0-9_]*/;
    return {
      name: "Modelica",
      aliases: ["mo"],
      keywords: {
        keyword: [
          "algorithm", "and", "annotation", "block", "break", "class", "connect", "connector", "constant",
          "constrainedby", "discrete", "each", "else", "elseif", "elsewhen", "encapsulated", "end", "enumeration",
          "equation", "expandable", "extends", "external", "final", "flow", "for", "function", "if", "import",
          "impure", "in", "initial", "inner", "input", "loop", "model", "not", "operator", "or", "outer", "output",
          "package", "parameter", "partial", "protected", "public", "pure", "record", "redeclare", "replaceable",
          "return", "stream", "then", "type", "when", "while", "within"
        ],
        type: ["Real", "Integer", "Boolean", "String", "Clock", "ExternalObject", "StateSelect", "AssertionLevel"],
        literal: ["true", "false"],
        built_in: [
          "abs", "sign", "sqrt", "div", "mod", "rem", "ceil", "floor", "integer", "sin", "cos", "tan", "asin",
          "acos", "atan", "atan2", "sinh", "cosh", "tanh", "exp", "log", "log10", "der", "delay", "cardinality",
          "homotopy", "semiLinear", "inStream", "actualStream", "spatialDistribution", "getInstanceName", "sample",
          "pre", "edge", "change", "reinit", "terminal", "noEvent", "smooth", "assert", "terminate", "size",
          "ndims", "scalar", "vector", "matrix", "identity", "diagonal", "zeros", "ones", "fill", "linspace", "min",
          "max", "sum", "product", "transpose", "outerProduct", "symmetric", "cross", "skew", "cat", "array",
          "previous", "hold", "subSample", "superSample", "shiftSample", "backSample", "firstTick", "interval",
          "time"
        ]
      },
      contains: [
        hljs.C_LINE_COMMENT_MODE,
        hljs.C_BLOCK_COMMENT_MODE,
        { scope: "string", begin: '"', end: '"', contains: [hljs.BACKSLASH_ESCAPE] },
        { scope: "number", match: /\b\d+(\.\d*)?([eE][+-]?\d+)?\b/ },
        { scope: "symbol", begin: "'", end: "'" },
        {
          begin: [/\b(?:model|class|block|connector|record|package|function|type|operator)\s+/, IDENT],
          beginScope: { 1: "keyword", 2: "title.class" }
        }
      ]
    };
  }

  function apply() {
    var hljs = window.hljs;
    if (!hljs) { return false; }
    if (!hljs.getLanguage("modelica")) { hljs.registerLanguage("modelica", modelica); }
    document.querySelectorAll("pre code.language-modelica").forEach(function (block) {
      if (!block.dataset.highlighted) { hljs.highlightElement(block); }
    });
    return true;
  }

  function waitForHljs(tries) {
    if (apply() || tries <= 0) { return; }
    setTimeout(function () { waitForHljs(tries - 1); }, 50);
  }

  window.addEventListener("load", function () { waitForHljs(200); });
})();
