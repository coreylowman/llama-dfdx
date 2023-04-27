use dfdx::{
    shapes::{Shape, Unit},
    tensor::{Cpu, Cuda, DeviceStorage, Tensor, TensorFromVec},
};

pub enum LazyTensor<S: Shape, E: Unit> {
    Disk {
        safetensors_path: String,
        key: &'static str,
        shape: S,
    },
    CPU(Tensor<S, E, Cpu>),
    CUDA(Tensor<S, E, Cuda>),
}

impl<S: Shape, E: Unit> LazyTensor<S, E> {
    pub fn load_into_cpu(&mut self, device: &Cpu) {
        todo!();
    }

    pub fn load_into_cuda(&mut self, device: &Cuda) {
        todo!();
    }
}

impl<S: Shape, E: Unit> LazyTensor<S, E> {
    pub fn load_on<D: TensorFromVec<E>>(&self, device: &D) -> Tensor<S, E, D> {
        todo!()
    }
}
