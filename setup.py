import os


supported = ['posix']

if os.name not in supported:
    raise EnvironmentError(f"system {os.name} not supported yet")


from os import path
from cffi import FFI
import subprocess

_c_lib_kernel_header = "_c_lib.h"
_c_lib_shared = "./lala/_C/shared"
_c_lib_name = "_c_lib_tensor" #final cffi Python extention code generated according to the Python C API


#TODO: also add options for clang compile
compiler = "gcc"
compiler_args = ["-shared", "-fopenmp", "-fPIC",  path.join(_c_lib_shared , "lib_tensor.c"), "-o", path.join(_c_lib_shared , "lib_c_tensor.so")]


try:
    s = subprocess.call([compiler] + compiler_args)
    print("C backend compiled")



    with open(path.join(_c_lib_shared, _c_lib_kernel_header)) as header:
        signs = header.read()


    ffi = FFI()

    #this is required so cffi knows about the functions defined in the lib
    ffi.cdef(signs)


    ffi.set_source(
        _c_lib_name,
        """
            #include "_c_lib.h"
        """,
        libraries = ["_c_tensor"],
        extra_objects=[path.abspath(f"{_c_lib_shared}/lib_c_tensor.so")]
    )

    ffi.compile(tmpdir=_c_lib_shared , verbose=True)
    print("Setup complete")

except FileNotFoundError:
    print('gcc not found on you system')
    exit()
except:
    #TODO: give the option to download a the _c_lib_tensor.so in case of a compiler missing
    def download_lib(): pass
    print("Error compiling lib")
    print("Downloading lib binary")
    download_lib()



