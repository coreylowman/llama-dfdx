use dfdx::{
    shapes::{HasShape, Shape, Unit},
    tensor::{CopySlice, Cpu, Tensor, TensorFromVec, ZerosTensor},
};

#[derive(Debug)]
pub enum LazyTensor<S: Shape, E: Unit> {
    Disk {
        path: std::path::PathBuf,
        shape: S,
    },
    CPU(Tensor<S, E, Cpu>),
    #[cfg(feature = "cuda")]
    CUDA(Tensor<S, E, dfdx::tenspr::Cuda>),
}

impl<S: Shape, E: Unit> LazyTensor<S, E> {
    pub fn load_into_cpu(&mut self, device: &Cpu) {
        let tensor = self.load_on(device);
        *self = Self::CPU(tensor);
    }

    #[cfg(feature = "cuda")]
    pub fn load_into_cuda(&mut self, device: &Cuda) {
        let tensor = self.load_on(device);
        *self = Self::CUDA(tensor);
    }
}

impl<S: Shape, E: Unit> LazyTensor<S, E> {
    fn shape(&self) -> S {
        match self {
            Self::Disk { path: _, shape } => *shape,
            Self::CPU(tensor) => *tensor.shape(),
            #[cfg(feature = "cuda")]
            Self::CUDA(tensor) => *tensor.shape(),
        }
    }

    pub fn load_on<D: ZerosTensor<E> + TensorFromVec<E> + CopySlice<E>>(
        &self,
        device: &D,
    ) -> Tensor<S, E, D> {
        let shape = self.shape();
        let numel = shape.num_elements();

        match self {
            Self::Disk { path, shape } => {
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
            }
            Self::CPU(tensor) => {
                // TODO if `D` is Cpu, we can just clone this
                // let buf = tensor.as_vec();
                // loaded.copy_from(&buf);
                todo!()
            }
            #[cfg(feature = "cuda")]
            Self::CUDA(tensor) => {
                // TODO if `D` is Cuda, we can just clone this
                todo!()
            }
        }
    }
}
