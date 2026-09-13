local torch = require 'torch'
local autograd = require 'autograd'
local tester = torch.Tester()
local tests = torch.TestSuite()

local cases = {
   {{1,3,1}, {2,1,3}}, -- preserve singleton dimensions
   {{1,3}, {2,3,2}},   -- add a leading dimension
   {{1,1}, {2,3}},
   {{3}, {2,3}},
   {{2,3}, {1,1}},     -- no repetitions
}

for _, optimized in ipairs{false,true} do
   for _, typename in ipairs{'torch.DoubleTensor','torch.FloatTensor'} do
      local mode, tensorType = optimized, typename
      local name = (mode and 'Optimized_' or 'Direct_') .. tensorType
      tests[name] = function()
         autograd.optimize(mode)
         torch.manualSeed(12345)
         for _, case in ipairs(cases) do
            local x = torch.randn(table.unpack(case[1])):type(tensorType)
            local repeats = case[2]
            local copies = 1
            for _, n in ipairs(repeats) do copies = copies * n end
            local df = autograd(function(v)
               local repeated = torch.repeatTensor(v, table.unpack(repeats))
               return torch.sum(torch.cmul(repeated, repeated))
            end)
            local g = df(x)
            tester:assert(g:isSameSizeAs(x), 'gradient must retain the input shape')
            tester:assertTensorEq(g, x * (2 * copies), 1e-4, 'incorrect repeat gradient')
         end

         local x = torch.randn(2,3):type(tensorType):t()
         local df = autograd(function(v)
            return torch.sum(torch.repeatTensor(v,2,3))
         end)
         local g = df(x)
         tester:assert(g:isSameSizeAs(x), 'noncontiguous input shape changed')
         tester:assertTensorEq(g, x:clone():fill(6), 0, 'incorrect noncontiguous gradient')
      end
   end
end

autograd.protected(true)
tester:add(tests):run()
