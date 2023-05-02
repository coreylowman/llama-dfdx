use dfdx::{
    shapes::{HasShape, Shape, Unit},
    tensor::{CopySlice, Cpu, Tensor, TensorFromVec, ZerosTensor},
};

use std::any::{Any, TypeId};

#[derive(Debug)]
#[non_exhaustive]
pub enum LazyTensor<S: Shape, E: Unit> {
    Disk {
        path: std::path::PathBuf,
        shape: S,
        move_to_ram: bool,
    },
    Cpu(Tensor<S, E, Cpu>),
    #[cfg(feature = "cuda")]
    Cuda(Tensor<S, E, dfdx::tensor::Cuda>),
}

impl<S: Shape, E: Unit> LazyTensor<S, E> {
    pub fn defer_load(&mut self) {
        if let Self::Disk {
            path: _,
            shape: _,
            move_to_ram,
        } = self
        {
            *move_to_ram = true;
        }
    }
}

impl<S: Shape, E: Unit> LazyTensor<S, E> {
    pub fn num_bytes(&self) -> usize {
        self.shape().num_elements() * std::mem::size_of::<E>()
    }

    pub fn is_on_disk(&self) -> bool {
        matches!(self, Self::Disk { .. })
    }

    fn shape(&self) -> S {
        match self {
            Self::Disk {
                path: _,
                shape,
                move_to_ram: _,
            } => *shape,
            Self::Cpu(tensor) => *tensor.shape(),
            #[cfg(feature = "cuda")]
            Self::Cuda(tensor) => *tensor.shape(),
        }
    }

    pub fn get_on<D: ZerosTensor<E> + TensorFromVec<E> + CopySlice<E>>(
        &mut self,
        device: &D,
    ) -> Tensor<S, E, D> {
        let shape = self.shape();
        let numel = shape.num_elements();

        match &self {
            Self::Disk {
                path,
                shape,
                move_to_ram,
            } => {
                let mut loaded = device.zeros_like(shape);
                let file = std::fs::File::open(path).unwrap();
                let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
                let bytes: &[u8] = &mmap;
                let ptr = bytes.as_ptr() as *const E;
                assert!(bytes.len() < (isize::MAX as usize));
                assert_eq!(bytes.len(), numel * std::mem::size_of::<E>());
                assert_eq!(ptr.align_offset(std::mem::align_of::<E>()), 0);
                // # Safety
                // - assertion checks for byte length
                // - non-null because we created from bytes slice
                // - aligned due to assertion
                let slice = unsafe { std::slice::from_raw_parts(ptr, numel) };
                loaded.copy_from(slice);

                if *move_to_ram {
                    if TypeId::of::<D>() == TypeId::of::<Cpu>() {
                        let tensor: Box<dyn Any> = Box::new(loaded.clone());
                        *self = Self::Cpu(*tensor.downcast().unwrap());
                    } else {
                        #[cfg(feature = "cuda")]
                        if TypeId::of::<D>() == TypeId::of::<dfdx::tensor::Cuda>() {
                            let tensor: Box<dyn Any> = Box::new(loaded.clone());
                            *self = Self::Cuda(*tensor.downcast().unwrap());
                        } else {
                            panic!("Unsupported device found (not Cpu/Cuda");
                        }

                        #[cfg(not(feature = "cuda"))]
                        panic!("Unsupported device found (not Cpu/Cuda");
                    }
                }
                loaded
            }
            Self::Cpu(tensor) => {
                if TypeId::of::<D>() == TypeId::of::<Cpu>() {
                    // Here since we know `D` is of type `Cpu`, we can just clone the tensor.
                    // However we can't easily return `tensor.clone()` because of the generic
                    // type.
                    //
                    // One idea might be to use std::mem::transmute, however that gives us
                    // an error about depedendly sized types for some reason.
                    //
                    // Instead we can go through Box<Any> and downcast it, which basically
                    // goes through pointers to do this.
                    let t: Box<dyn Any> = Box::new(tensor.clone());
                    *t.downcast().unwrap()
                } else {
                    let mut loaded = device.zeros_like(tensor.shape());
                    let buf = tensor.as_vec();
                    loaded.copy_from(&buf);
                    loaded
                }
            }
            #[cfg(feature = "cuda")]
            Self::Cuda(tensor) => {
                if TypeId::of::<D>() == TypeId::of::<dfdx::tensor::Cuda>() {
                    // See comment in corresponding Self::CPU branch.
                    let t: Box<dyn Any> = Box::new(tensor.clone());
                    *t.downcast().unwrap()
                } else {
                    let mut loaded = device.zeros_like(tensor.shape());
                    let buf = tensor.as_vec();
                    loaded.copy_from(&buf);
                    loaded
                }
            }
        }
    }
}
