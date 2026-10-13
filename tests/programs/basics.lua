-- Deterministic Lua workload; output is compared with the native build's (basics.expected).
local function fib(n) if n < 2 then return n end return fib(n - 1) + fib(n - 2) end
print("fib", fib(25))

local t = {}
for i = 1, 2000 do t[i] = (i * 7919) % 2003 end
table.sort(t)
print("sorted", t[1], t[1000], t[2000], #t)

local words = {}
for w in ("the quick brown fox jumps over the lazy dog the end"):gmatch("%a+") do
  words[w] = (words[w] or 0) + 1
end
local keys = {}
for k in pairs(words) do keys[#keys + 1] = k end
table.sort(keys)
for _, k in ipairs(keys) do io.write(k, "=", words[k], " ") end
print()

print(string.format("%.17g %.3e %5.2f %x %q", math.pi, 1 / 3, -2.5, 255, "a\nb"))
print(("hello world"):gsub("o", "0"), ("%d items"):format(3), ("abc"):rep(3, "-"))
print(math.sqrt(2), math.floor(-3.5), math.fmod(7, 3), 7 // 2, 7 % -3, 2 ^ 53, 1e300 * 1e10)
print(math.maxinteger + 1 == math.mininteger, 0x7fffffff * 0x7fffffff, 3 / 0, -(0 / 0) ~= -(0 / 0))

local co = coroutine.wrap(function(a)
  for i = 1, 3 do a = coroutine.yield(a * i) end
  return "done"
end)
print("co", co(2), co(10), co(100), co(0))

local function counter()
  local n = 0
  return function() n = n + 1 return n end
end
local c1, c2 = counter(), counter()
c1() c1()
print("closures", c1(), c2())

print("pcall", pcall(error, {code = 42}), select("#", pcall(error)))
local ok, msg = pcall(function() local x = nil; return x.y end)
print("error", ok, (msg:gsub("^.-:%d+: ", "")))

local mt = {__add = function(a, b) return a.v + b.v end, __tostring = function(s) return "V" .. s.v end}
local a, b = setmetatable({v = 1}, mt), setmetatable({v = 2}, mt)
print("meta", a + b, tostring(a))

local big = {}
for i = 1, 200000 do big[i] = tostring(i) end
print("concat", #table.concat(big))
print("utf8", utf8.len("héllo wörld"), utf8.char(0x4e2d, 0x6587))
print("pack", string.unpack("<i4", string.pack("<i4", -2)), #string.pack("z", "abc"))
collectgarbage()
print("gc", collectgarbage("count") > 0)
