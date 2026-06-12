"""
```
debugWrite(filename::String, data::String)
```
  If logging is enabled dumps this as a string.
"""
function debugWrite(filename::String, data::String)
  write(filename, data)
end
