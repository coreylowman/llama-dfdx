use dfdx::{
    shapes::{HasShape, Shape, Unit},
    tensor::{CopySlice, Cpu, Tensor, TensorFromVec, ZerosTensor},
};

use std::any::{Any, TypeId};

#[derive(Debug)]
pub struct MemoryMappedTensor<S: Shape, E: Unit> {
    tensor: Tensor<S, E, Cpu>,
    mmap: memmap2::Mmap,
}

#[derive(Debug)]
#[non_exhaustive]
pub enum LazyTensor<S: Shape, E: Unit> {
    Disk {
        path: std::path::PathBuf,
        shape: S,
        move_to_ram: bool,
    },
    MemoryMapped(Option<MemoryMappedTensor<S, E>>),
    #[cfg(feature = "cuda")]
    Cuda(Tensor<S, E, dfdx::tensor::Cuda>),
}

impl<S: Shape, E: Unit> LazyTensor<S, E> {
    pub fn defer_load(&mut self) {
        if let Self::Disk { move_to_ram, .. } = self {
            *move_to_ram = true;
        }
    }
}

impl<S: Shape, E: Unit> Drop for LazyTensor<S, E> {
    fn drop(&mut self) {
        match self {
            Self::Disk { .. } => {}
            Self::MemoryMapped(tensor) => {
                if let Some(MemoryMappedTensor { tensor, .. }) = std::mem::take(tensor) {
                    // since this tensor doesn't own the vec, we need to forget it so it doesn't get dropped
                    std::mem::forget(tensor);
                }
            }
            #[cfg(feature = "cuda")]
            Self::Cuda(tensor) => {}
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
            Self::Disk { shape, .. } => *shape,
            Self::MemoryMapped(tensor) => *tensor.as_ref().unwrap().tensor.shape(),
            #[cfg(feature = "cuda")]
            Self::Cuda(tensor) => *tensor.shape(),
        }
    }

    pub fn move_to_ram<D: ZerosTensor<E> + TensorFromVec<E> + CopySlice<E>>(&mut self, device: &D) {
        if let Self::Disk { move_to_ram, .. } = self {
            if *move_to_ram {
                self.get_on(device);
            }
        }
    }

    pub fn get_on<D: ZerosTensor<E> + TensorFromVec<E> + CopySlice<E>>(
        &mut self,
        device: &D,
    ) -> Tensor<S, E, D> {
        let shape = self.shape();
        let numel = shape.num_elements();

        match self {
            Self::Disk {
                path,
                shape,
                move_to_ram,
            } => {
                if !*move_to_ram {
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
                    loaded
                } else {
                    if TypeId::of::<D>() == TypeId::of::<Cpu>() {
                        let file = std::fs::File::open(path).unwrap();
                        let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
                        let bytes: &[u8] = &mmap;
                        let ptr = bytes.as_ptr() as *mut E;
                        assert!(bytes.len() < (isize::MAX as usize));
                        assert_eq!(bytes.len(), numel * std::mem::size_of::<E>());
                        assert_eq!(ptr.align_offset(std::mem::align_of::<E>()), 0);
                        // # Safety
                        // TODO
                        let vec = unsafe { Vec::from_raw_parts(ptr, numel, numel) };
                        let loaded = device.tensor_from_vec(vec, *shape);
                        let tensor: Box<dyn Any> = Box::new(loaded.clone());
                        *self = Self::MemoryMapped(Some(MemoryMappedTensor {
                            tensor: *tensor.downcast().unwrap(),
                            mmap,
                        }));
                        loaded
                    } else {
                        #[cfg(feature = "cuda")]
                        if TypeId::of::<D>() == TypeId::of::<dfdx::tensor::Cuda>() {
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
                            let t: Box<dyn Any> = Box::new(tensor.as_ref().unwrap().tensor.clone());
                            self = Self::Cuda(*tensor.downcast().unwrap());
                            loaded
                        } else {
                            panic!("Unsupported device found (not Cpu/Cuda");
                        }

                        #[cfg(not(feature = "cuda"))]
                        panic!("Unsupported device found (not Cpu/Cuda)");
                    }
                }
            }
            Self::MemoryMapped(tensor) => {
                // Here since we know `D` is of type `Cpu`, we can just clone the tensor.
                // However we can't easily return `tensor.clone()` because of the generic
                // type.
                //
                // One idea might be to use std::mem::transmute, however that gives us
                // an error about depedendly sized types for some reason.
                //
                // Instead we can go through Box<Any> and downcast it, which basically
                // goes through pointers to do this.
                assert_eq!(TypeId::of::<D>(), TypeId::of::<Cpu>());
                let t: Box<dyn Any> = Box::new(tensor.as_ref().unwrap().tensor.clone());
                *t.downcast().unwrap()
            }
            #[cfg(feature = "cuda")]
            Self::Cuda(tensor) => {
                // See comment in corresponding Self::CPU branch.
                assert_eq!(TypeId::of::<D>(), TypeId::of::<dfdx::tensor::Cuda>());
                let t: Box<dyn Any> = Box::new(tensor.clone());
                *t.downcast().unwrap()
            }
        }
    }
}
