
from typing import List, Optional, Union, Tuple
from lala._C import ffi
import math
from .utils import graph_html
from .ops import *
from .blob import Blob
import numpy as np
from .dtype import Dtype
from lala.dtype import float32

def isTensor(obj): return isinstance(obj, Tensor)

class Tensor: 
    """
    A Tensor is an n dimensional array of values of the same type
    The underlying memory region is just an big 1D buffer(Blob) of values
    """
    def __init__(self, *args, data: Optional[Union[List[List[int]] | "Tensor" | Blob]]=None, dtype=float32, label=None, offset=0,  src: Optional[Operation]=None, strides: Optional[Tuple[int]]=None, requires_grad=False):
        assert (not requires_grad) or (dtype is float32), "requires_grad allowed for dtype=float32"
        assert src is None or isinstance(src, Operation), "A Tensor src can only be an Op or None "
        if data is not None:
            if isinstance(data, list):
                #use numpy to convert the list to a Contiguous memory block and get a void pointer to it
                #then delete the numpy object (we don't need it)
                #TODO: Implement our own list to buffer of dtyper in C
                np_dtype = np.float32 if dtype is float32 else np.int32
                arr = np.array(data, dtype=np_dtype, order="C")
                shape = arr.shape
                nbytes = arr.size * dtype.bytes
                ptr = arr.ctypes.data
                blob = Blob(ptr=ptr, nbytes=nbytes)
                #We need to keep the numpy object as it owns the buffer and gc may free the buffer if the 
                #np.array object gets deleted by the python Garbage Collector
                #TODO: ofcource  we will fix this
                blob.__setattr__("numpy", arr)

            #from numpy ndarray
            elif isinstance(data, np.ndarray):
                shape = args if len(args) else data.shape
                nbytes = data.size * dtype.bytes
                ptr = data.ctypes.data
                blob = Blob(ptr=ptr, nbytes=nbytes)
                dtype = float32 if data.dtype == np.float32 else int32 
                blob.__setattr__("numpy", data)

            #you can also just pass a Storage Blob and build a tensor on top
            #it doesn't copy so any change to this the tensor changes the storage 
            #ultimately chaging the data of any other tensor base on that passed blob
            elif isinstance(data, Blob):
                shape = args
                blob = data
            elif isinstance(data, int):
                shape = ()
                dtype = int32
                blob = Blob(nbytes=4)
                blob._get_pointer("int*")[0] = data
            elif isinstance(data, float):
                shape = ()
                dtype = int32
                blob = Blob(nbytes=4)
                blob._get_pointer("float*")[0] = data

            else: 
                raise TypeError("data must be One of this types Tensor, list, Blob, float, int, bool")
        else: 
            assert len(args) > 0, "shape or data is required"
            shape = args
            blob = Blob(math.prod(args)*self.numel())

        #this is basically what a tensor if
        self.storage = blob #a memory blob where the data is stored
        self.dtype = dtype  #a datatype for determing how to interpret that data
        self.shape = shape  #a shape for knowing the dimentions of that data

        #this are ue for buiding the autograd graph for backward gradient calculation
        self.src = src #the src of a tensor (None for leaf tensors and a Function object for non leaf tensors)
        self.requires_grad = requires_grad  #used to decide on wether we need to include this tesnor in the autograd graph
        self.grad = None

        self.dims = len(self.shape) 

        self.label = label
        if strides is None:
            if self.dims:
                s = [1]
                r = tuple(reversed(self.shape))
                for i in range(self.dims - 1):
                    s.append(s[i] * r[i])
                self.strides = tuple(reversed(s))
            else:
                self.strides = ()
        else: 
            self.strides = strides

    @property
    def T(self):
        if self.dims < 2: raise RuntimeError(f"tensor of dim {self.dims} can't be transposed")
        return self.transpose(-1, -2)

    @classmethod
    def empty(cls, *args, dtype=float32, label=None, requires_grad=False):
        blob = Blob(nbytes=math.prod(args)*dtype.bytes)
        return cls(*args, data=blob, dtype=dtype, label=label, requires_grad=requires_grad)
    
    @classmethod
    def fill(cls, *args, value, dtype=None, label=None, requires_grad=False):
        if dtype is None:
            if isinstance(value, int): dtype = int32
            elif isinstance(value, float): dtype = float32
        blob = Blob(nbytes=math.prod(args)*dtype.bytes, fill=value)
        return cls(*args, data=blob, dtype=dtype, label=label, requires_grad=requires_grad)
    
    @classmethod
    def rand(cls, *args, dtype=float32, label=None, requires_grad=False):
        assert len(args), "shape args required"
        #Once again use numpy for random generation
        rand = np.random.uniform(-1.0, 1.0, size=math.prod(args)).astype(np.float32)
        
        return cls(*args, data=rand, dtype=dtype, label=label, requires_grad=requires_grad)
    
    @classmethod
    def zeros(cls, *args, dtype=float32, label=None, requires_grad=False):
        b = Blob(nbytes=dtype.bytes * math.prod(args), zero_init=True)
        return cls(data=b, *args, dtype=dtype, label=label, requires_grad=requires_grad)

    @classmethod
    def zeros_like(cls, t: "Tensor", label=None, dtype=float32,requires_grad=False):
        b = Blob(nbytes=t.storage.nbytes, zero_init=True)
        return cls(data=b, *t.shape, dtype=dtype, label=label,  requires_grad=requires_grad)
    
    @classmethod
    def ones(cls, *args, dtype=float32, label=None, requires_grad=False):
        b = Blob(nbytes=dtype.bytes * math.prod(args), fill=1.0 if dtype is float32 else 1)
        return cls(data=b, *args, dtype=dtype, label=label, requires_grad=requires_grad)
    
    @classmethod
    def ones_like(cls, t: "Tensor", label=None, dtype=float32,requires_grad=False):
        blob = Blob(nbytes=t.storage.nbytes, fill=1)
        new = cls(*t.shape, data=blob, dtype=dtype, label=label, requires_grad=requires_grad)
        new.fill(1).detach()
        return new
    
    def __getitem__(self, args): return Slice(self, args)()

    #this is just a dummy place holder to be used where we need to use a tensor

    @classmethod
    def dummy(cls, label: str="Dummy"):
        return cls(0, label=label)
    

    def data_ptr(self):
        return int(self.storage._get_pointer("uintptr_t"))
    
    def detach(self):
        assert self.src is not None, "trying to detach an unattached tensor"
        self.src = None
        
    def contiguous(self):
        """
        returns a new Tensor that is stored in a new buffer in a row-major order
        This also makes sure the storage for this new Tensor is not shared with any other tensor 

        We currently just convert the tensor to a python list and create a new tensor from it
        """
        return Tensor(*self.shape, data=self.tolist())
    
    def is_contiguous(self):
        pass

    def clone(self):
        b = Blob(self.storage.nbytes)
        self.storage._copy(b)
        return Tensor(*self.shape, data=b,label=f"{self.label}-clone", dtype=self.dtype, requires_grad=self.requires_grad)

    def __repr__(self):
        return f"Tensor(shape=<{self.shape}> dtype=<{self.dtype.name}> grad=<{None if self.src is None else self.src.name}>)"
    
    def is_scalar(self):
        return len(self.shape) == 0
    
    def view(self, *args): 
        if len(args):
            return View(self, args)()
        return View(self, self.shape)()  
        
        
    def to(self, dtype: Dtype): 
        new = self.clone()
        print("New", new.requires_grad)
        if self.dtype is dtype: return self
        
        CastOp.forward(self, new, dtype)
        new.dtype = dtype
        return new
    

    def to_(self, dtype: Dtype):
        CastOp.forward(self, self, dtype)
        print(self.requires_grad)
        return self
    

    
    def visualize(self, file_name="graph.html"):
        graph_html(self, filename=file_name)
        print(f"Computaion Graph exported as {file_name}")

    

    def dot(self, other): return (self * other).sum(1)

    def __add__(self, other): return Add(self, other)()
    def __sub__(self, other): return Sub(self, other)()
    def __mul__(self, other): return Mul(self, other)()
    def __matmul__(self, other): return Matmul(self, other)()

    def sum(self, dim=None): return Sum(self, dim)()
    def mean(self, dim=None): return Mean(self, dim)()
    def smul(self, scalar): return ScalarMul(self, scalar)()

    def transpose(self, dim0, dim1): return Transpose(self, dim0, dim1)()
    def broadcast_to(self, *args): return BroadCast(self, args)()
    def expand(self, *args): return BroadCast(self, args)()
        
    
    def spow(self, exp): return ScalarPower(self, exp)()

    

    def item(self): 
        assert not len(self.shape), "get_item only defined for scalar "
        return self.storage._get_pointer(self.dtype.ptr_t)[0]

    def tolist(self): 
        if len(self.shape):
            return self.storage._to_python_list(self.shape, self.stride(), self.dtype.ptr_t)
        return self.get_item()

    def numel(self): return math.prod(self.shape)
    def dim(self): return self.dims
    def stride(self): return self.strides


    def backward(self, upstream_m: Optional["Tensor"]=None):
        assert self.requires_grad, "matrice doesn't requre_grad"
            
        if self.src:
            #TODO: Better to check if tensor is Scalar than relay on op_type

            if not upstream_m and not self.is_scalar():
                raise RuntimeError("implicit backward only defined for single elemnt matrices")
            self.src.backward(upstream_m)
        else:
            assert upstream_m, "implicit backward only defined for single elemnt matrices"
            assert self.shape == upstream_m.shape, "invalid shape upstream tensor"
            self.grad = upstream_m


