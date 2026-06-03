
"""
Operation types:
    Unary -> take a Tensor return Tensor of the same shape
    Binary -> take two Tensors and return a Tensor
    SReduce (Scalar reduce) take a Tensor and return single element Tensor (implicit autograd allowed)
"""

from .utils import *
from .dtype import int32, Null, Dtype
from typing import Tuple

from .blob import Blob
from ._C import ops
import math
from enum import Enum, auto
from dataclasses import dataclass

#This is used for checking ops and stuff
#But planning to use them to hide the op implementation when we migrate to C
class Ops(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    POW = auto()
    SUM = auto()
    MEAN = auto()

    SMUL = auto()
    SPOW = auto()

    TRANSPOSE = auto()
    VIEW = auto()
    BROADCAST = auto()
    SLICE = auto()


    MATMUL = auto()




class Operation:
    def __init__(self, name: str, type_: str, op: int, *args):
        from .tensor import  Tensor
        self.name = name
        self.op = op
        self.op_type = type_
        #The dtype this op works at. the result has this type and operands are casted to it before kernel call
        self.dtype = max(operand.dtype if isinstance(operand, Tensor) else Null for operand in args)

        
        #cast every operand to the highes level operand
        casted_operands = []
        for operand in args:
            casted_v = operand.to_(self.dtype) if isinstance(operand, Tensor) and not operand.dtype is  self.dtype else operand
            casted_operands.append(casted_v)

        self.operands = tuple(casted_operands)
        self.requires_grad =  any(operand.requires_grad if isinstance(operand, Tensor) else False for operand in self.operands)


    def detach(self, operand):
        assert operand in self.operands, f"operand not attached to {self}"
        operands = list(self.operands)
        operands.remove(operand)
        self.operands =tuple(operands)

    def __call__(self): 
        from .tensor import Tensor
        data, shape, strides = self.forward()

        #Don't include in the Op graph unless it requires gradient
        if self.requires_grad:
            src = self
        else:
            src = None
        self.res = Tensor(*shape, data=data, strides=strides, src=src, dtype=self.dtype, requires_grad=self.requires_grad)
        return self.res
        
    

    def backward(self, upstream_m):
        from lala.tensor import Tensor
        for operand in self.operands:
            if isinstance(operand, Tensor) and operand.requires_grad:
                if self.op_type == "BinaryOp":
                    grad_b =   self.gradient(operand)
                elif self.op_type == "ViewOp":
                    grad_b =   self.gradient(upstream_m)
                elif self.op_type == "UnOp" or self.op_type == "ReduceOp":
                    grad_b =   self.gradient()
                else:
                    grad_b =   self.gradient(operand, upstream_m)

                grad = Tensor(*operand.shape, data=grad_b)
                #Remember VewOps don't participate in the chan rule. They just do a reverse op on the upstream and pass it up
                if upstream_m is not None and self.op_type != "ViewOp" and self.name != "MMul":
                    grad *= upstream_m

                #Set the grad or accumulate if the tensor already has a grad 
                if operand.grad is None:
                    operand.grad = grad
                else: 
                    operand.grad += grad

                #continue up the autodiffer chain
                operand.backward(operand.grad)


"""
We break down ops into these 5 ops so we can optimize the forward, broadcasting and backward optimized accordingly ...
"""

class UnOps(Operation): 
    """
    These are operantions on a tensor that take a tensor and do the same op 
    (function) on every element
    ex. smul. spow, sadd, ssub
    like ops with other types in python int, bool. float
    """
    def __init__(self, name, op, lhs, other): 
        super().__init__(name, "UnOp", op, lhs, other)


class BinaryOps(Operation): 
    """
    These are Ops that take two tensors and return a tensor of the same shape
    ex. +, -, ** and most other ops

    """
    def __init__(self, name, op, *args): 
        super().__init__(name, "BinaryOp", op, *args)


class ReduceOps(Operation):
    def __init__(self, name, op, *args): 
        super().__init__(name, "ReduceOp", op, *args)

    """
    There are ops that return a tensor of smaller shape
    ex. mean(<dim>), sum(<dim>), along some dim
    """

class ViewOps(Operation): 
    """
    These are just different ways of looking at the storage(memory)
    ex. transpose, view, expand...
    The gradient of ViewOps is just the reverse if the changes
    """, 
    def __init__(self, name, op, t, new_shape): 
        from .tensor import Tensor
        super().__init__(name, "ViewOp", op, t, new_shape)



#=======================================here are all the ops supported by  lalagrad=============================
class Mean(ReduceOps):
    def __init__(self, tensor, dim): 
        super().__init__("MEAN", Ops.MEAN, tensor, dim)

    def forward(self):
        m, dim = self.operands
        res = Blob(m.dtype.bytes)
        #TODO: Make the mean kernel take a dimension
        ops[self.dtype.name].mean_t(m.storage, res)
        return res, (), None
    
    def gradient(self):
        m = self.operands[0]
        grad_b = Blob(nbytes=m.storage.nbytes, fill=1/m.numel())
        return grad_b
        
   

class ScalarPower(UnOps):
    def __init__(self, *args): 
        super().__init__("ElPow", Ops.SPOW, *args)

    def forward(self): 
        lhs, exp = self.operands
        res = Blob(nbytes=lhs.storage.nbytes)
        ops[self.dtype.name].exp(lhs.storage, exp, res)
        return res, lhs.shape, None
        
    def gradient(self): 
        lhs, exp = self.operands
        grad_b = Blob(nbytes=lhs.storage.nbytes)
        ops[self.dtype.name].mul_s(lhs.storage, exp, grad_b)
        return grad_b

    

class ScalarMul(UnOps):
    def __init__(self, *args): 
        super().__init__("SMul", Ops.SMUL, *args)

    def forward(self): 
        lhs, s = self.operands
        res_b = Blob(lhs.storage.nbytes)
        ops[self.dtype.name].mul_s(lhs.storage, s,  res_b)

        return res_b, lhs.shape, None
        
    def gradient(self): 
        w_r_t, scalar = self.operands
        grad_b = Blob(nbytes=w_r_t.storage.nbytes, fill=scalar)
        return grad_b

    
class Add(BinaryOps):
    def __init__(self, lhs, rhs):
        super().__init__("Add", Ops.ADD, lhs, rhs)
    
    def forward(self):
        rhs, lhs = self.operands
        res = Blob(nbytes=rhs.storage.nbytes)
        ops[self.dtype.name].add_t(rhs.storage, lhs.storage, res)
        return res, rhs.shape, None

    def gradient(self, w_r_t): 
        assert w_r_t in self.operands, "w_r_t is not an operand of this grad_fn"
        grad_b = Blob(nbytes=w_r_t.storage.nbytes, fill=1.0)
        return  grad_b
    



class Sub(BinaryOps):
    def __init__(self, *args):
        super().__init__("Sub", Ops.SUB, *args)
    
    def forward(self):
        rhs, lhs = self.operands
        res = Blob(nbytes=rhs.storage.nbytes)
        ops[self.dtype.name].sub_t(rhs.storage, lhs.storage, res)
        return res, rhs.shape, None
    

    def gradient(self, w_r_t): 
        assert w_r_t in self.operands, "w_r_t is not an operand of this grad_fn"
        lhs, _ = self.operands
        if w_r_t is lhs:
            fill_v = 1.0
        else:
            fill_v = -1.0
        return Blob(lhs.storage._get_size(), fill=fill_v)
            


class Sum(ReduceOps):
    """
    Sum elements of a tensor along a dim
    The gradient is just 1s of the same shape
    """
    def __init__(self, tensor, dim=None): 
        super().__init__("Sum", Ops.SUM, tensor, dim)
        

    def forward(self, dim=None): 
        m, dim = self.operands
        #TODO: remove the condition and write a generic kernel that handles this
        if dim is None:
            rhs, _ = self.operands
            res = Blob(nbytes=m.dtype.bytes)
            ops[self.dtype.name].sum_t(rhs.storage, res)
            return res, (), None
            #TODO: Implement sum allong dim with strides
        else: 
            raise NotImplementedError("Sum along a dim not Implemented yet")
    
    def gradient(self):
        lhs, dim = self.operands
        grad_b = Blob(lhs.storage._get_size(), fill=1.0)
        return grad_b

        

class Mul(BinaryOps):
    """
    Does element-wise multiplication
    backward with respect to one of the operands is the other
    """
    def __init__(self, *args): super().__init__("ElMul", Ops.MUL, *args)

    def forward(self): 
        rhs, lhs = self.operands
        res = Blob(nbytes=rhs.storage.nbytes)
        ops[self.dtype.name].mul_t(rhs.storage, lhs.storage, res)
        return res, rhs.shape, None
    
    def gradient(self, w_r_t): 
        lhs, rhs = self.operands
        other_b = rhs.storage if lhs is w_r_t else lhs.storage
        grad_b = Blob(nbytes=other_b.nbytes)
        other_b._copy(grad_b)
        return grad_b



class Relu(Operation):
    """
    Does the relu activation on on each element of a tensor
    the gradient of relu is just relu
    """
    def __init__(self, *args): super().__init__("Relu", *args)

    def forward(self):
        m = self.operands[0]
        res = Blob.empty(*m.shape, dtype=int32)
        
        return res, m.shape, None 
    
    def gradient(self):
        m = self.operands[0]
        return [[1 if e > 0 else 0 for e in row] for row in m.data]


        
class Transpose(ViewOps):
    """exchanges the elements of two dims"""
    def __init__(self, lhs, dim0: int, dim1: int): 
        super().__init__("Transpose", Ops.TRANSPOSE, lhs, (dim0, dim1))


    def forward(self): 
        m, dims = self.operands
        dim0, dim1 = dims

        new_shape = list(m.shape)
        a = new_shape[dim0]
        new_shape[dim0] = new_shape[dim1]
        new_shape[dim1] = a

        new_stride = list(m.stride())
        a = new_stride[dim0]
        new_stride[dim0] = new_stride[dim1]
        new_stride[dim1] = a
        return m.storage, tuple(new_shape), tuple(new_stride)
    
    def gradient(self, upstream_m):
        _, dims = self.operands
        dim0, dim1 = dims
        return upstream_m.transpose(dim1, dim0).contiguous().storage
     
        
class View(ViewOps):
    """
    This is a litteral different view of a tensor. 
    No elemt postion swap is done just just reinterpret the buffer
    """
    def __init__(self, lhs, new_shape):
        super().__init__("View", Ops.VIEW,   lhs, new_shape)
    
    def forward(self):
        t, new_shape = self.operands
        return t.storage, new_shape, None

    def gradient(self):
        lhs = self.operands
        grad_b = Blob(lhs.storage.nbytes)
        grad_b._copy(lhs.storage)
        return grad_b


class BroadCast(ViewOps):
    def __init__(self, lhs, new_shape: Tuple[int]):
        super().__init__("Broadcast", Ops.BROADCAST, lhs, new_shape)

    def forward(self):
        lhs, shape = self.operands
        new_dims = len(shape) - len(lhs.shape)
        assert new_dims >= 0, "can't broadcast to a smaller dim shape"
        assert shape != lhs.shape, "can't broadcast to old shape"
        assert all(shape[new_dims+i] == lhs.shape[i] or lhs.shape[i]==1  for i in range(lhs.dim())), "new shape can ony replace size 1 dims from existing dims"
        new_shape, new_stride = shape, tuple(0 for _ in range(new_dims)) + lhs.stride()
        return lhs.storage, new_shape, new_stride


class Slice(ViewOps):
    """
    Creates a new Tensor by slicing a tensor 
    No data copy just a new view
    """
    def __init__(self, lhs, args):
        super().__init__("Slice", Ops.SLICE, lhs, args)

    def forward(self):
        lhs, args = self.operands
        assert len(args) == lhs.dim()

        strides = list(lhs.strides)
        shape = list(lhs.shape)
        offset = 0

        for dim, s in enumerate(args):
            if isinstance(s, int): 
                start = s
                stop = s + 1
                step = 1
            elif isinstance(s, slice): 
                start = 0 if s.start is None else s.start
                stop = shape[dim] if s.stop is None else s.stop
                step = 1 if s.step is None else s.step
            else: raise IndexError(f"Invalid Indexing element {s}")

            offset += start * strides[dim]
            shape[dim] = (stop - start + step - 1) // step
            strides[dim] = strides[dim] * step


        return  Blob(ptr=lhs.storage._get_pointer(lhs.dtype.ptr_t)+offset, nbytes=0), shape, strides
    
#this does casting to a dtype
#for now that is just to float32
#doesn't support backward
class CastOp:
    @staticmethod
    def forward(lhs, res, dtype: Dtype):
        ops[dtype.name].cast(lhs.storage, res.storage, lhs.numel())


        
class Matmul(Operation):
    """
    Matmul doesn't fit in the any of the above ops as  my may return tensors of different and
    also the gradient is different from other Binary ops

    This is also the only op that returns a pointer for the result tensor in case of 
    batched matmul as the mem size and allocation are handled in the kernel
    """
    
    def __init__(self, *args):
        super().__init__("MMul", "MBinary", Ops.MATMUL, *args)
        
    def forward(self):
        lhs, rhs = self.operands
        #this dims are along which we do matmuls 
        #common_dims is > 0 for batch matmul and 0 for single matmul
        print
        rhs_rows, rhs_cols = rhs.shape[-2:]

        lhs_rows, lhs_cols = lhs.shape[-2:]
        dims = lhs.dims

        assert lhs_cols == rhs_rows and lhs.shape[:dims-2] == rhs.shape[:dims-2]
        
        res_shape = (lhs_rows, rhs_cols)
        if(dims > 2):
            res_shape = lhs.shape[:dims-2] + res_shape
        rhs_strides = rhs.stride()
        lhs_strides = lhs.stride()

        res_strides = list(get_strides(res_shape))
        
        res = Blob(nbytes=math.prod(res_shape)*self.dtype.bytes)

        ops[self.dtype.name].batch_matmul(lhs.storage, rhs.storage, res, res_shape, res_strides, lhs.shape, lhs_strides, rhs.shape, rhs_strides, dims)
        for i in range(dims):
            if not lhs_strides[i] and not rhs_strides[i]: res_strides[i] = 0
        return res, res_shape, tuple(res_strides)

    def gradient(self, w_r_t, upstream_m): 
        lhs, rhs = self.operands

        if w_r_t is lhs:
            rhs_t = rhs.T.contiguous()
            res = Matmul(upstream_m, rhs_t)
            lgrad_b,_, __ = res.forward()
            res.detach(rhs_t)
            return lgrad_b
        
        else:
            lhs_t = lhs.T.contiguous()
            rgrad_b, _, __ = Matmul(lhs_t, upstream_m).forward()
            return rgrad_b
