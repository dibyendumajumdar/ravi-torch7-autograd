-- Various torch additions, should move to torch itself

torch.select = function (A, dim, index)
   return A:select(dim, index)
end

torch.index = function (A, dim, index)
   return A:index(dim, index)
end

torch.narrow = function(A, dim, index, size)
   return A:narrow(dim, index, size)
end

torch.clone = function(A)
   local B = newTensor(A, A:size())
   return B:copy(A)
end

torch.contiguous = function(A)
   return A:contiguous()
end

torch.copy = function(A,B)
   local o = A:copy(B)
   return o
end

torch.size = function(A, dim)
   return A:size(dim)
end

torch.nDimension = function(A)
   return A:nDimension()
end

torch.nElement = function(A)
   return A:nElement()
end

torch.isSameSizeAs = function(A, B)
   return A:isSameSizeAs(B)
end

torch.transpose = function(A)
   return A:t()
end

torch.narrow = function(dim, index, size)
   return A:narrow(dim, index, size)
end

-- Allow number * tensor style operations

local numberMetatable = {
   __add = function(a,b)
      if type(a) == "number" and isTensor(b) then
         return b + a
      else
         return a + b
      end
   end,
   __sub = function(a,b)
      if type(a) == "number" and isTensor(b) then
         return -b + a
      else
         return a - b
      end
   end,
   __mul = function(a,b)
      if type(a) == "number" and isTensor(b) then
         return b * a
      else
         return a * b
      end
   end
}

debug.setmetatable(1.0, numberMetatable)